from pipeline_seg import StableDiffusionXLSEGPipeline
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch
import argparse
import itertools
import os
from pathlib import Path


pipe = StableDiffusionXLSEGPipeline.from_pretrained(
    "socks22/sdxl-wai-nsfw-illustriousv14",
    torch_dtype=torch.float16
)

device = "cuda"
pipe = pipe.to(device)
prompt = (
    "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, 2girls, "
    "aqua_(konosuba), blue sword, left_side, "
    "megumin, red_sword, right_side, "
    "shiny skin, shiny clothes, looking at viewer, volumetric_lightning, futuristic_city, neon_lights, night"
)

seed = 42


def run_single_inference():
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompts,
        num_inference_steps=8,
        guidance_scale=1.0,
        seg_scale=1.0,
        seg_blur_sigma=50,
        seg_applied_layers=['mid'],
        generator=generator,
    )

    output.images[0].save("output.png")


def run_baseline_inference(benchmark_dir=None):
    print("Running baseline generation...")
    baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
        "socks22/sdxl-wai-nsfw-illustriousv14",
        torch_dtype=torch.float16
    )
    baseline_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(baseline_pipe.scheduler.config)
    baseline_pipe = baseline_pipe.to(device)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = baseline_pipe(
        prompts,
        num_inference_steps=12,
        guidance_scale=1.0,
        generator=generator,
    )

    if benchmark_dir:
        image_path = benchmark_dir / "baseline.png"
    else:
        image_path = Path("baseline_output.png")

    output.images[0].save(image_path)
    print(f"Baseline image saved to {image_path}")
    return {"image_path": image_path.as_posix()}


def run_benchmark():
    benchmark_dir = Path("benchmark_results")
    benchmark_dir.mkdir(exist_ok=True)

    baseline_result = run_baseline_inference(benchmark_dir=benchmark_dir)

    seg_scales = [1.0, 2.0, 3.0, 5.0]
    seg_blur_sigmas = [10.0, 50.0, 200.0, 999.0]
    seg_applied_layers_options = [
        ['down'],
        ['mid'],
        ['up'],
        ['down', 'mid'],
        ['mid', 'up'],
        ['down', 'mid', 'up'],
    ]

    results = []
    
    param_combinations = list(itertools.product(seg_scales, seg_blur_sigmas, seg_applied_layers_options))

    for i, (seg_scale, seg_blur_sigma, seg_applied_layers) in enumerate(param_combinations):
        print(f"Running combination {i+1}/{len(param_combinations)}: seg_scale={seg_scale}, seg_blur_sigma={seg_blur_sigma}, seg_applied_layers={seg_applied_layers}")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        filename_layers = "_".join(seg_applied_layers)
        image_name = f"scale_{seg_scale}_blur_{seg_blur_sigma}_layers_{filename_layers}.png"
        image_path = benchmark_dir / image_name

        output = pipe(
            prompts,
            num_inference_steps=12,
            guidance_scale=1.0,
            seg_scale=seg_scale,
            seg_blur_sigma=seg_blur_sigma,
            seg_applied_layers=seg_applied_layers,
            generator=generator,
        )
        output.images[0].save(image_path)
        
        results.append({
            "seg_scale": seg_scale,
            "seg_blur_sigma": seg_blur_sigma,
            "seg_applied_layers": ", ".join(seg_applied_layers),
            "image_path": image_path.as_posix(),
        })

    generate_html_report([baseline_result] + results, benchmark_dir)


def generate_html_report(results, benchmark_dir):
    html_content = "<html><head><title>Benchmark Results</title>"
    html_content += "<style> table, th, td { border: 1px solid black; border-collapse: collapse; padding: 10px; text-align: center; } </style>"
    html_content += "</head><body><h1>Benchmark Results</h1><table>"
    html_content += "<tr><th>Seg Scale</th><th>Seg Blur Sigma</th><th>Seg Applied Layers</th><th>Image</th></tr>"

    for result in results:
        html_content += "<tr>"
        if "seg_scale" in result:
            html_content += f"<td>{result['seg_scale']}</td>"
            html_content += f"<td>{result['seg_blur_sigma']}</td>"
            html_content += f"<td>{result['seg_applied_layers']}</td>"
        else:
            html_content += '<td colspan="3">Baseline (No SEG)</td>'
        html_content += f'<td><img src="{result["image_path"]}" width="256"></td>'
        html_content += "</tr>"

    html_content += "</table></body></html>"

    report_path = benchmark_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html_content)
    print(f"HTML report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--baseline", action="store_true", help="Run baseline generation only")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.baseline:
        run_baseline_inference()
    else:
        run_single_inference()