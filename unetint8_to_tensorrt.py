import os
import torch
import tensorrt as trt
from diffusers import UNet2DConditionModel
from huggingface_hub import snapshot_download
import modelopt.torch.opt as mto
from tqdm import tqdm
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            self._step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            self._step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            return False

class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        return output

def build_engine(
    engine_path: str,
    onnx_path: str,
    input_profiles: dict,
    fp16: bool = True,
    int8: bool = False,
):
    print(f"Building TensorRT engine for {onnx_path}: {engine_path}")

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    if os.path.exists(engine_path):
        print("Engine already exists, skipping build.")
        return

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    success = parser.parse_from_file(onnx_path)
    if not success:
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))
        raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    config = builder.create_builder_config()

    profile = builder.create_optimization_profile()
    for name, (min_shape, opt_shape, max_shape) in input_profiles.items():
        profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)

    config.progress_monitor = TQDMProgressMonitor()

    print("Building engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    print("Engine built successfully.")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Export INT8 UNet to ONNX and/or build TensorRT engine."
    )
    parser.add_argument(
        "--only-onnx",
        action="store_true",
        help="Only export the ONNX model, skip building the TensorRT engine.",
    )
    args = parser.parse_args()

    base_model_id = "socks22/sdxl-wai-nsfw-illustriousv14"
    model_dir = snapshot_download(base_model_id)
    int8_checkpoint_path = os.path.join(model_dir, "unet_int8.safetensors")
    
    output_dir = "/workflow/wai_dmd2_onnx/unet"
    onnx_output_path = os.path.join(output_dir, "model_int8.onnx")
    engine_output_path = os.path.join(output_dir, "model_int8.plan")
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading base UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to("cuda")

    print(f"Restoring INT8 weights from {int8_checkpoint_path}...")
    mto.restore(unet, int8_checkpoint_path)
    unet.eval()
    print("INT8 UNet restored successfully.")

    print(f"Exporting INT8 UNet to ONNX: {onnx_output_path}")
    if os.path.exists(onnx_output_path):
        print("ONNX model already exists, skipping export.")
    else:
        # Dummy inputs for ONNX export
        batch_size = 1
        latent_height = 120
        latent_width = 120
        
        sample = torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float16).to("cuda")
        timestep = torch.tensor(999, dtype=torch.float16).to("cuda")
        encoder_hidden_states = torch.randn(batch_size, 77, 2048, dtype=torch.float16).to("cuda")
        text_embeds = torch.randn(batch_size, 1280, dtype=torch.float16).to("cuda")
        time_ids = torch.randn(batch_size, 6, dtype=torch.float16).to("cuda")
        
        unet_wrapper = UnetWrapper(unet)
        
        input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
        output_names = ["out_sample"]
        dynamic_axes = {
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size"},
            "text_embeds": {0: "batch_size"},
            "time_ids": {0: "batch_size"},
        }
        
        with torch.no_grad():
            torch.onnx.export(
                unet_wrapper,
                (sample, timestep, encoder_hidden_states, text_embeds, time_ids),
                onnx_output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                export_params=True,
            )
        print("ONNX export complete.")

    if args.only_onnx:
        print("Successfully exported ONNX model. Exiting as requested by --only-onnx.")
        return

    print("Building INT8 TensorRT engine...")
    latent_heights = [768 // 8, 1152 // 8, 960 // 8]
    latent_widths = [1152 // 8, 768 // 8, 960 // 8]
    
    min_h, max_h = min(latent_heights), max(latent_heights)
    min_w, max_w = min(latent_widths), max(latent_widths)
    opt_h, opt_w = 960 // 8, 960 // 8
    bs = 1

    unet_input_profiles = {
        "sample": (
            (bs, 4, min_h, min_w),
            (bs, 4, opt_h, opt_w),
            (bs, 4, max_h, max_w),
        ),
        "timestep": ((), (), ()),
        "encoder_hidden_states": ((bs, 77, 2048), (bs, 77, 2048), (bs, 77, 2048)),
        "text_embeds": ((bs, 1280), (bs, 1280), (bs, 1280)),
        "time_ids": ((bs, 6), (bs, 6), (bs, 6)),
    }
    
    build_engine(
        engine_path=engine_output_path,
        onnx_path=onnx_output_path,
        input_profiles=unet_input_profiles,
        fp16=True,
        int8=True,
    )

    print(f"\nINT8 UNet engine saved successfully to {engine_output_path}")

if __name__ == "__main__":
    main() 