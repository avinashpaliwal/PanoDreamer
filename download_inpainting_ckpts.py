"""
Download inpainting checkpoints from Google Drive.

The original 3d-moments download.sh has broken links, so we provide
an alternative download from a public Google Drive folder.

Usage:
    python download_inpainting_ckpts.py
"""

import os
import gdown

# Google Drive folder with inpainting checkpoints
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ShKK1nTPj-ZSTYu_2g6Ix9XltN5lVMEC"

# Expected checkpoint files
CHECKPOINTS = {
    "depth-model.pth": None,  # Will be filled with file IDs if needed
    "color-model.pth": None,
}

def download_checkpoints(output_dir="inpainting_ckpts"):
    """
    Download inpainting checkpoints from Google Drive.
    
    Args:
        output_dir: Directory to save checkpoints
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Downloading inpainting checkpoints to {output_dir}/")
    print(f"[INFO] Source: {GDRIVE_FOLDER_URL}")
    
    try:
        # Download entire folder
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        print(f"[INFO] Successfully downloaded checkpoints to {output_dir}/")
        
        # Verify files
        for ckpt_name in CHECKPOINTS.keys():
            ckpt_path = os.path.join(output_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                print(f"[INFO] ✓ {ckpt_name} ({size_mb:.1f} MB)")
            else:
                print(f"[WARNING] ✗ {ckpt_name} not found")
        
    except Exception as e:
        print(f"[ERROR] Failed to download checkpoints: {e}")
        print("[INFO] Please download manually from:")
        print(f"       {GDRIVE_FOLDER_URL}")
        print(f"       and place files in {output_dir}/")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download inpainting checkpoints")
    parser.add_argument("--output_dir", type=str, default="inpainting_ckpts",
                        help="Output directory for checkpoints")
    args = parser.parse_args()
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown is not installed")
        print("[INFO] Install with: pip install gdown")
        exit(1)
    
    success = download_checkpoints(args.output_dir)
    
    if success:
        print("[INFO] Done! You can now run ldi_generation.py with inpainting support.")
    else:
        print("[INFO] Download failed. Please download manually.")
