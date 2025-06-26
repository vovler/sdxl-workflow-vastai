import os
import time
import json
import asyncio
import uuid as uuid_lib
from pathlib import Path
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Worker Service")

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
            print(f"Error reading UUID file: {e}")
    return None

def save_uuid(uuid_str: str):
    """Save UUID to file"""
    try:
        Path("uuid.txt").write_text(uuid_str)
        print(f"UUID saved: {uuid_str}")
    except Exception as e:
        print(f"Error saving UUID: {e}")

async def connect_to_parent() -> Optional[str]:
    """Connect to parent host and get UUID"""
    get_env_values()
    
    if not parent_host or not public_ip or not port:
        print("Missing required environment variables: PARENT_HOST, PUBLIC_IPADDR, VAST_TCP_PORT_80")
        return None
    
    payload = {
        "PublicIP": public_ip,
        "Port": port
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{parent_host}/worker/connect", json=payload)
            response.raise_for_status()
            data = response.json()
            received_uuid = data.get("uuid")
            if received_uuid:
                save_uuid(received_uuid)
                print(f"Connected to parent, received UUID: {received_uuid}")
                return received_uuid
            else:
                print("No UUID received from parent")
                return None
    except Exception as e:
        print(f"Error connecting to parent: {e}")
        return None

async def send_healthcheck():
    """Send healthcheck to parent"""
    if not worker_uuid or not parent_host:
        return
    
    # Update environment values in case they changed
    get_env_values()
    
    payload = {
        "UUID": worker_uuid,
        "PublicIP": public_ip,
        "Port": port
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{parent_host}/worker/healthcheck", json=payload)
            response.raise_for_status()
            print(f"Healthcheck sent successfully")
    except Exception as e:
        print(f"Error sending healthcheck: {e}")

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
        "UUID": worker_uuid,
        "TaskID": task_id,
        "Status": status
    }
    
    if status_text:
        payload["StatusText"] = status_text
    
    try:
        if image_data:
            # Send with image data
            files = {"image": ("test_image.png", image_data, "image/png")}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{parent_host}/worker/task", data=payload, files=files)
        else:
            # Send JSON only
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{parent_host}/worker/task", json=payload)
        
        response.raise_for_status()
        print(f"Task status sent: {status} for task {task_id}")
    except Exception as e:
        print(f"Error sending task status: {e}")

async def process_task(task_id: str, task_data: Dict[str, Any]):
    """Process a task with the specified workflow"""
    print(f"Processing task {task_id} with data: {json.dumps(task_data)}")
    
    # Send PROCESSING status after 2 seconds
    await asyncio.sleep(2)
    await send_task_status(task_id, "PROCESSING", "Worker is processing the request...")
    
    # Send DONE status with image after another 2 seconds
    await asyncio.sleep(2)
    
    # Load and send the test image
    try:
        image_path = Path("test_image.png")
        if image_path.exists():
            image_data = image_path.read_bytes()
            await send_task_status(task_id, "DONE", image_data=image_data)
        else:
            print("Warning: test_image.png not found")
            await send_task_status(task_id, "DONE")
    except Exception as e:
        print(f"Error processing task: {e}")
        await send_task_status(task_id, "ERROR", f"Task processing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize worker on startup"""
    global worker_uuid, healthcheck_task
    
    print("Worker starting up...")
    
    # Check if UUID exists
    worker_uuid = load_uuid()
    
    if not worker_uuid:
        print("No UUID found, connecting to parent...")
        worker_uuid = await connect_to_parent()
        
        if not worker_uuid:
            print("Failed to get UUID from parent")
            return
    else:
        print(f"Using existing UUID: {worker_uuid}")
    
    # Start periodic healthcheck
    healthcheck_task = asyncio.create_task(periodic_healthcheck())
    print("Healthcheck task started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if healthcheck_task:
        healthcheck_task.cancel()
        try:
            await healthcheck_task
        except asyncio.CancelledError:
            pass

@app.post("/pipeline/create_task")
async def create_task(task_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Handle incoming task creation requests"""
    task_id = str(uuid_lib.uuid4())
    
    # Print the encoded JSON string as requested
    print(f"Received task: {json.dumps(task_data)}")
    
    # Start processing in background
    background_tasks.add_task(process_task, task_id, task_data)
    
    # Return STARTED status with TaskId
    return JSONResponse(
        content={
            "Status": "STARTED",
            "TaskId": task_id
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
        print("VAST_TCP_PORT_80 environment variable not set")
        return
    
    try:
        port_int = int(port)
        print(f"Starting worker server on port {port_int}")
        uvicorn.run(app, host="0.0.0.0", port=port_int)
    except ValueError:
        print(f"Invalid port value: {port}")

if __name__ == "__main__":
    main()
