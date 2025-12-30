import os
import sys
import time
import subprocess
from pathlib import Path

try:
    from PIL import ImageGrab
    import win32gui
    import win32con
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "pywin32"])
    from PIL import ImageGrab
    import win32gui
    import win32con


def maximize_window_by_pid(pid, max_wait=2):
    """Find and maximize the window associated with a process ID."""
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                hwnds.append(hwnd)
        return True
    
    import win32process
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        if hwnds:
            # Maximize the first window found
            hwnd = hwnds[0]
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        time.sleep(0.1)
    return False


def screenshot_mdl_files(folder_path):
    """
    For each .mdl file in the given folder:
    1. Launch med.exe with the file as argument
    2. Maximize the window
    3. Wait 1 second
    4. Take a screenshot and save as PNG with the same name
    """
    med_exe = r"C:\Program Files (x86)\GStudio8\med.exe"
    
    if not os.path.exists(med_exe):
        print(f"Error: med.exe not found at {med_exe}")
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        return
    
    mdl_files = list(folder.glob("*.mdl"))
    
    if not mdl_files:
        print(f"No .mdl files found in {folder_path}")
        return
    
    print(f"Found {len(mdl_files)} .mdl file(s)")
    
    for mdl_file in mdl_files:
        print(f"Processing: {mdl_file.name}")
        
        # Launch med.exe with the mdl file as argument
        process = subprocess.Popen([med_exe, str(mdl_file)])
        
        # Wait a bit for the window to appear, then maximize it
        time.sleep(0.5)
        if maximize_window_by_pid(process.pid):
            print(f"Window maximized")
        else:
            print(f"Warning: Could not find window to maximize")
        
        # Wait 1 second for the application to fully render
        time.sleep(1)
        
        # Take screenshot of the entire screen
        screenshot = ImageGrab.grab()
        
        # Save with same name but .png extension
        png_path = mdl_file.with_suffix(".png")
        screenshot.save(png_path, "PNG")
        print(f"Saved screenshot: {png_path.name}")
        
        # Close the process
        process.terminate()
        process.wait()
        
        # Small delay before next file
        time.sleep(0.5)
    
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python screenshot_mdl.py <folder_path>")
        print("Example: python screenshot_mdl.py C:\\Models")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    screenshot_mdl_files(folder_path)
