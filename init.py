import os
import time
import json
import asyncio
import uuid as uuid_lib
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import torch
from torchvision.io import encode_jpeg
import numpy as np
from run_inference_trt import generate_image, initialize_pipeline

# Global variables
worker_uuid: Optional[str] = None
parent_host: str = ""
public_ip: str = ""
port: str = ""
healthcheck_task: Optional[asyncio.Task] = None

def get_env_values():
    """Get and update environment values"""
    global public_ip, port, parent_host
    public_ip = os.getenv("PUBLIC_IPADDR", "")
    port = os.getenv("VAST_TCP_PORT_80", "")
    parent_host = "http://" + os.getenv("PARENT_HOST", "")
    return public_ip, port, parent_host

def load_uuid() -> Optional[str]:
    """Load UUID from file if it exists"""
    uuid_file = Path("uuid.txt")
    if uuid_file.exists():
        try:
            return uuid_file.read_text().strip()
        except Exception as e:
            print(f"Error reading UUID file: {e}", flush=True)
    return None

def save_uuid(uuid_str: str):
    """Save UUID to file"""
    try:
        Path("uuid.txt").write_text(uuid_str)
        print(f"UUID saved: {uuid_str}", flush=True)
    except Exception as e:
        print(f"Error saving UUID: {e}", flush=True)

async def connect_to_parent() -> Optional[str]:
    """Connect to parent host and get UUID"""
    get_env_values()
    
    if not parent_host or not public_ip or not port:
        print("Missing required environment variables: PARENT_HOST, PUBLIC_IPADDR, VAST_TCP_PORT_80", flush=True)
        return None
    
    payload = {
        "public_ip": public_ip,
        "port": int(port)
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{parent_host}/worker/connect", json=payload)
            response.raise_for_status()
            data = response.json()
            received_uuid = data.get("uuid")
            if received_uuid:
                save_uuid(received_uuid)
                print(f"Connected to parent, received UUID: {received_uuid}", flush=True)
                return received_uuid
            else:
                print("No UUID received from parent", flush=True)
                return None
    except Exception as e:
        print(f"Error connecting to parent: {e}", flush=True)
        return None

async def send_healthcheck():
    """Send healthcheck to parent"""
    if not worker_uuid or not parent_host:
        return
    
    # Update environment values in case they changed
    get_env_values()
    
    payload = {
        "uuid": worker_uuid,
        "public_ip": public_ip,
        "port": int(port)
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{parent_host}/worker/healthcheck", json=payload)
            response.raise_for_status()
            print(f"Healthcheck sent successfully", flush=True)
    except Exception as e:
        print(f"Error sending healthcheck: {e}", flush=True)

async def periodic_healthcheck():
    """Send healthcheck every 5 seconds"""
    while True:
        await asyncio.sleep(5)
        await send_healthcheck()

async def send_task_status(task_id: str, status: str, status_text: Optional[str] = None, image_data: Optional[bytes] = None):
    """Send task status update to parent"""
    if not worker_uuid or not parent_host:
        return
    
    payload = {
        "uuid": worker_uuid,
        "task_id": task_id,
        "status": status
    }
    
    if status_text:
        payload["status_text"] = status_text
    
    # Debug: Check image_data in send_task_status
    print(f"send_task_status - image_data type: {type(image_data)}", flush=True)
    print(f"send_task_status - image_data size: {len(image_data) if image_data else 'None'}", flush=True)
    print(f"send_task_status - image_data is not None: {image_data is not None}", flush=True)
    
    try:
        if image_data:
            # Send with image data as multipart form data
            # Don't specify content-type in files tuple to ensure multipart/form-data is used
            files = {"image": ("test_image.png", image_data)}
            # Convert payload to individual form fields
            form_data = {
                "uuid": worker_uuid,
                "task_id": task_id,
                "status": status
            }
            if status_text:
                form_data["status_text"] = status_text
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                # httpx automatically sets Content-Type to multipart/form-data when files parameter is used
                response = await client.post(f"{parent_host}/worker/task", data=form_data, files=files)
        else:
            # Send JSON only
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{parent_host}/worker/task", json=payload)
        
        response.raise_for_status()
        print(f"Task status sent: {status} for task {task_id}", flush=True)
    except Exception as e:
        print(f"Error sending task status: {e}", flush=True)

async def process_task(task_id: str, task_data: Dict[str, Any]):
    """Process a task with the specified workflow"""
    print(f"Processing task {task_id} with data: {json.dumps(task_data)}", flush=True)
    
    # Send PROCESSING status
    await send_task_status(task_id, "PROCESSING", "Worker is processing the request...")
    
    try:
        # Extract parameters from task_data or use defaults
        prompt = task_data.get("prompt", "")

        # Generate image using run_inference_trt module
        image_data, img = generate_image(
            prompt=prompt
        )
        
        # Minify image to JPEG 85% on GPU, removing metadata
        print("Minifying image to JPEG (85% quality)...", flush=True)
        try:
            # Convert PIL image to uint8 tensor (C, H, W) and move to GPU
            tensor_img_gpu = torch.from_numpy(np.array(img)).permute(2, 0, 1).to("cuda")

            # Encode to JPEG on GPU
            jpeg_tensor_gpu = encode_jpeg(tensor_img_gpu, quality=85)

            # Move back to CPU and get bytes
            image_data = jpeg_tensor_gpu.cpu().numpy().tobytes()
            print(f"Minified image size: {len(image_data)} bytes", flush=True)
        except Exception as e:
            print(f"Could not minify image on GPU, falling back to CPU: {e}", flush=True)
            # Fallback to CPU-based conversion with Pillow
            from io import BytesIO
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            image_data = buffer.getvalue()
        
        # Send DONE status with generated image
        await send_task_status(task_id, "DONE", "Image generated successfully!", image_data=image_data)
        
    except Exception as e:
        print(f"Error processing task: {e}", flush=True)
        await send_task_status(task_id, "ERROR", f"Task processing failed: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global worker_uuid, healthcheck_task
    
    # Startup
    print("Worker starting up...", flush=True)
    
    # Initialize SDXL pipeline
    try:
        print("About to initialize SDXL pipeline...", flush=True)
        initialize_pipeline()
        print("SDXL pipeline initialization completed!", flush=True)
    except Exception as e:
        print(f"Warning: Failed to initialize SDXL pipeline: {e}", flush=True)
    
    # Check if UUID exists
    worker_uuid = load_uuid()
    
    if not worker_uuid:
        print("No UUID found, connecting to parent...", flush=True)
        worker_uuid = await connect_to_parent()
        
        if not worker_uuid:
            print("Failed to get UUID from parent", flush=True)
            yield
            return
    else:
        print(f"Using existing UUID: {worker_uuid}", flush=True)
    
    # Start periodic healthcheck
    healthcheck_task = asyncio.create_task(periodic_healthcheck())
    print("Healthcheck task started", flush=True)
    
    yield
    
    # Shutdown
    if healthcheck_task:
        healthcheck_task.cancel()
        try:
            await healthcheck_task
        except asyncio.CancelledError:
            pass

# Initialize FastAPI app with lifespan
app = FastAPI(title="Worker Service", lifespan=lifespan)

@app.post("/pipeline/create_task")
async def create_task(task_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Handle incoming task creation requests"""
    task_id = str(uuid_lib.uuid4())
    
    # Print the encoded JSON string as requested
    print(f"Received task: {json.dumps(task_data)}", flush=True)
    
    # Start processing in background
    background_tasks.add_task(process_task, task_id, task_data)
    
    # Return STARTED status with TaskId
    return JSONResponse(
        content={
            "status": "STARTED",
            "task_id": task_id
        }
    )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "uuid": worker_uuid}

def main():
    """Main function to run the FastAPI server"""
    get_env_values()
    
    if not port:
        print("VAST_TCP_PORT_80 environment variable not set", flush=True)
        return
    
    try:
        port_int = 80
        print(f"Starting worker server on port {port_int}", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port_int)
    except ValueError:
        print(f"Invalid port value: {port}", flush=True)

if __name__ == "__main__":
    main()
