import os
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import uvicorn
from urllib.parse import quote
import requests

# The base directory for serving files. This will be configured at runtime
# based on command-line arguments or environment variables.
BASE_DIR = None

app = FastAPI()

@app.get("/{file_path:path}")
async def serve_path(file_path: str = ""):
    """
    Serves a file or provides a browsable directory listing.
    This single endpoint handles both file downloads and directory browsing.
    """
    print(f"--- New Request ---")
    print(f"Received request for path: '{file_path}'")
    try:
        # Construct the full, absolute path for the requested file or directory.
        requested_path = BASE_DIR.joinpath(file_path).resolve()
        print(f"Resolved requested path to: {requested_path}")

        # Security check: Prevent directory traversal attacks.
        # Ensure the resolved path is within the designated BASE_DIR.
        print(f"Checking if {requested_path} is within {BASE_DIR.resolve()}")
        if BASE_DIR.resolve() not in requested_path.parents and requested_path != BASE_DIR.resolve():
             print("Security check FAILED: Path is outside the base directory.")
             raise HTTPException(status_code=403, detail="Forbidden: Access denied.")
        print("Security check PASSED.")

    except Exception as e:
         print(f"Error resolving path or security check failed: {e}")
         # Broad exception to catch potential resolution errors.
         raise HTTPException(status_code=404, detail="File or directory not found.")

    print(f"Checking existence of: {requested_path}")
    if not requested_path.exists():
        print("Path does NOT exist.")
        raise HTTPException(status_code=404, detail="File or directory not found")
    print("Path exists.")

    if requested_path.is_dir():
        print("Path is a directory. Generating directory listing.")
        # Generate an HTML page with a list of directory contents.
        # The page will be titled with the current directory path.
        html_content = f"<html><head><title>Index of /{file_path}</title></head><body><h1>Index of /{file_path}</h1><ul>"
        
        # Add a link to the parent directory, if not in the root directory.
        if requested_path != BASE_DIR.resolve():
            parent_path = Path(file_path).parent
            parent_url = f"/{parent_path}" if str(parent_path) != '.' else '/'
            html_content += f'<li><a href="{parent_url}">..</a></li>'

        # List all items in the directory, with links for navigation/download.
        for item in sorted(requested_path.iterdir()):
            # Create a URL-safe path for the href attribute.
            url_path = quote(str(item.relative_to(BASE_DIR)))
            # Append a slash to directory names for clarity.
            display_name = item.name + ("/" if item.is_dir() else "")
            html_content += f'<li><a href="/{url_path}">{display_name}</a></li>'
        
        html_content += "</ul></body></html>"
        return HTMLResponse(content=html_content)
    
    elif requested_path.is_file():
        print("Path is a file. Serving file for download.")
        # Serve the file for download.
        return FileResponse(requested_path)
    
    else:
        print("Path is not a file or directory (e.g., symlink).")
        # This case handles other path types, like symlinks, which are not supported.
        raise HTTPException(status_code=400, detail="Unsupported path type.")

if __name__ == "__main__":
    # Set up argument parser to handle the directory path.
    parser = argparse.ArgumentParser(
        description="A simple HTTP server for serving files from a specified directory."
    )
    parser.add_argument(
        "directory",
        help="The local directory path to serve files from."
    )
    args = parser.parse_args()


    BASE_DIR = Path(args.directory)

    # Before starting the server, check if the target directory exists and is a directory.
    if not BASE_DIR.is_dir():
        print(f"Error: The specified path '{BASE_DIR}' is not a valid directory.")
        exit(1)

    # Get the external port from an environment variable.
    port = os.getenv("VAST_TCP_PORT_80", "")
    if not port:
        print("Error: The 'VAST_TCP_PORT_80' environment variable is not set.")
        exit(1)

    # Fetch the public IP address.
    try:
        response = requests.get("https://api.ipify.org/?format=json")
        response.raise_for_status()
        public_ip = response.json()["ip"]
    except requests.RequestException as e:
        print(f"Error: Could not fetch public IP address.")
        print(f"Details: {e}")
        exit(1)
    
    print(f"Serving files from: {BASE_DIR.resolve()}")
    print("Starting server on http://0.0.0.0:80")
    print(f"You can access it on http://{public_ip}:{port}")
    
    try:
        # Attempt to run the server on the privileged port 80.
        uvicorn.run(app, host="0.0.0.0", port=80)
    except (PermissionError, OSError) as e:
        # Fail gracefully if port 80 is not available.
        print(f"\nERROR: Could not bind to port 80. Please ensure it is not in use and you have sufficient permissions.")
        print(f"Details: {e}")
        exit(1)
