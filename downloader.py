import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import uvicorn
from urllib.parse import quote

# Define the directory to be served. The path can be configured via an
# environment variable `WAI_DMD2_ONNX_DIR` for flexibility across different
# operating systems (like Windows for development and Linux for production).
# If the environment variable is not set, it defaults to the Linux production path.
BASE_DIR_PATH = os.environ.get("WAI_DMD2_ONNX_DIR", "/workflow/wai_dmd2_onnx")
BASE_DIR = Path(BASE_DIR_PATH)

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
    # Before starting the server, check if the target directory exists.
    if not BASE_DIR.exists() or not BASE_DIR.is_dir():
        print(f"WARNING: The directory '{BASE_DIR.resolve()}' does not exist.")
        print("Please ensure the path is correct or set the 'WAI_DMD2_ONNX_DIR' environment variable.")
    
    print(f"Serving files from: {BASE_DIR.resolve()}")
    print("Starting server on http://0.0.0.0:80")
    
    try:
        # Attempt to run the server on the privileged port 80.
        uvicorn.run(app, host="0.0.0.0", port=80)
    except (PermissionError, OSError):
        # If port 80 is unavailable (e.g., due to permissions), fall back to port 8000.
        print("\nERROR: Permission denied to bind to port 80.")
        print("This is common on Unix-like systems where ports below 1024 require root access.")
        print("Falling back to port 8000.")
        print("Server starting on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
