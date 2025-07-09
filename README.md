https://docs.nvidia.com/deeplearning/tensorrt/10.12.0/_static/c-api/namespacemembers.html



HOSTNAME ENV?
VAST_TCP_PORT_80
find equivalent for "new_contract" value
CONTAINER_ID=22352052 (?)

trt.init_libnvinfer_plugins(None, "")
https://huggingface.co/SmilingWolf/wd-vit-tagger-v3

optimum-cli export onnx --device cuda --opset 18 --dtype fp16 --no-post-process --no-constant-folding --framework pt --model socks22/sdxl-wai-nsfw-illustriousv14 --task text-to-image wai_dmd2_onnx2

cqyan/hybrid-sd-tinyvae-xl

pip install onnxslim
onnxslim --inspect model.onnx

###### RAM USAGE / SPEED #######
FP16 UNET FP16 VAE Decoder - Height: 832  Width: 1216
Batch 1 Without Tiling: 9891MB 0.67
Batch 1 With Tiling: 7871MB 1.12s (2x time, -2gb of vram for

Batch 2 With Tiling: 7871MB 2.24s (batch size: 2)

###### PIP ######
TMPDIR=/dev/shm/ pip install --no-cache-dir -r requirements.txt && rm -rf /dev/shm/*

apt update -y; apt upgrade -y;
pip install --upgrade pip;
apt install aria2;
aria2c -x 16 -s 16 -k 10M "https://test-storage-pull1123.b-cdn.net/diffusion_pytorch_model.safetensors"


sudo apt install libjemalloc-dev libjemalloc2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python your_training_script.py

###### VLLM ######

pip install vllm flashinfer-python cuda-python==12.9.0

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B-AWQ \
    --quantization awq_marlin \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --host 127.0.0.1 \
    --port 80
    --api-key 123456789
	
https://docs.vllm.ai/en/latest/examples/others/lmcache.html (KVCache in CPU RAM)
https://docs.vllm.ai/en/latest/examples/others/tensorize_vllm_model.html (Fast Loading Of The Model - NOT TENSORRT)

###### VAST AI ######
vastai change bid

show instance
execute
show instances

vastai search offers --interruptible --storage=10 'reliability > 0.98 num_gpus=1 gpu_name=RTX_3060 rented=False' -o "min_bid+"



vastai create instance 19958699 --bid_price 0.039 --image vastai/pytorch:2.6.0-cuda-12.4.1-py312-22.04 --env '-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -p 80:80 -e OPEN_BUTTON_PORT=1111 -e OPEN_BUTTON_TOKEN=1 -e JUPYTER_DIR=/ -e DATA_DIRECTORY=/workspace/ -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard" -e WORKFLOW_GIT_URL=https://github.com/vovler/vastai-sdxl-workflow -e PARENT_HOST=168.119.117.45 -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/vovler/vastai-sdxl-provisioning/refs/heads/main/provisioning.sh' --onstart-cmd 'entrypoint.sh' --disk 10 --jupyter --ssh --direct
{'success': True, 'new_contract': 21963913}

vastai show instance 21964380

vastai change bid 21964380 --price='0.01'


###### Remove Logos ######
https://huggingface.co/spaces/EduardoPacheco/Grounding-Dino-Inference
https://huggingface.co/IDEA-Research/grounding-dino-tiny
patreon, logo, text

https://huggingface.co/spaces/Sanster/Lama-Cleaner-lama
https://github.com/Sanster/IOPaint