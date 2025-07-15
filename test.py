import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import numpy as np

# Simple model with a loop
class SimpleLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Regular model with loop (non-scripted)
class RegularLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x, num_steps):
        for i in range(num_steps):
            x = self.linear(x) + 1.0
        return x

# Test both versions
def test_exports():
    # Create models
    regular_model = SimpleLoop()
    scripted_model = torch.jit.script(regular_model)  # Script the instance
    
    # Sample input
    x = torch.randn(1, 10)
    num_steps = 5
    
    print("Testing TorchScript version:")
    try:
        torch.onnx.export(
            scripted_model,
            (x, num_steps),
            "scripted_loop.onnx",
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Scripted model exported successfully")
    except Exception as e:
        print(f"❌ Scripted model failed: {e}")
    
    print("\nTesting regular model:")
    try:
        torch.onnx.export(
            regular_model,
            (x, num_steps),
            "regular_loop.onnx", 
            input_names=['x', 'num_steps'],
            output_names=['output'],
            dynamic_axes={'x': {0: 'batch_size'}},
            opset_version=11
        )
        print("✅ Regular model exported successfully")
    except Exception as e:
        print(f"❌ Regular model failed: {e}")
    
def build_tensorrt_engine(onnx_file, engine_file):
    """Build TensorRT engine from ONNX model"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build engine
    config = builder.create_builder_config()
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 10), (1, 10), (1, 10))
    profile.set_shape("num_steps", (1,), (1,), (1,))
    config.add_optimization_profile(profile)
    
    try:
        engine = builder.build_engine(network, config)
        if engine:
            with open(engine_file, 'wb') as f:
                f.write(engine.serialize())
            return engine
    except Exception as e:
        print(f"Engine build failed: {e}")
        return None

def test_tensorrt_engines():
    """Test TensorRT engines"""
    print("\n" + "="*50)
    print("TENSORRT ENGINE BUILDING")
    print("="*50)
    
    # Build engines
    scripted_engine = build_tensorrt_engine("scripted_loop.onnx", "scripted_loop.trt")
    regular_engine = build_tensorrt_engine("regular_loop.onnx", "regular_loop.trt")
    
    if scripted_engine:
        print("✅ Scripted TensorRT engine built successfully")
        print(f"   Engine size: {len(open('scripted_loop.trt', 'rb').read())} bytes")
    else:
        print("❌ Scripted TensorRT engine failed")
    
    if regular_engine:
        print("✅ Regular TensorRT engine built successfully")
        print(f"   Engine size: {len(open('regular_loop.trt', 'rb').read())} bytes")
    else:
        print("❌ Regular TensorRT engine failed")
    
    # Analyze engine details
    if scripted_engine:
        print(f"\nScripted engine layers: {scripted_engine.num_layers}")
        print("Layer info:")
        for i in range(scripted_engine.num_layers):
            layer = scripted_engine.get_layer(i)
            print(f"  Layer {i}: {layer.type}")
    
    if regular_engine:
        print(f"\nRegular engine layers: {regular_engine.num_layers}")
        print("Layer info:")
        for i in range(regular_engine.num_layers):
            layer = regular_engine.get_layer(i)
            print(f"  Layer {i}: {layer.type}")

if __name__ == "__main__":
    test_exports()
    test_tensorrt_engines()