"""
Download pre-trained models from Hugging Face Hub
This script downloads the entire model repository structure for WMH segmentation.
"""

from huggingface_hub import snapshot_download
import os

# Repository information
REPO_ID = "Bawil/wmh_leverage_normal_abnormal_segmentation"
LOCAL_DIR = "models"

def download_models():
    """Download entire model repository from Hugging Face Hub"""
    
    print(f"Downloading entire repository from {REPO_ID}...")
    print("This may take a few minutes depending on your connection speed.\n")
    
    try:
        # Download entire repository structure
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            repo_type="model"
        )
        print("\n✓ Successfully downloaded all models!")
        print(f"✓ Models saved to: {os.path.abspath(LOCAL_DIR)}")
        
        # Display downloaded structure
        print("\nDownloaded structure:")
        for root, dirs, files in os.walk(LOCAL_DIR):
            level = root.replace(LOCAL_DIR, '').count(os.sep)
            indent = '  ' * level
            print(f'{indent}{os.path.basename(root)}/')
            sub_indent = '  ' * (level + 1)
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    print(f'{sub_indent}{file}')
        
    except Exception as e:
        print(f"\n✗ Error downloading repository: {str(e)}")
        print("\nPlease ensure you have:")
        print("1. Internet connection")
        print("2. Installed huggingface_hub: pip install huggingface_hub")
        print(f"3. Access to the repository: https://huggingface.co/{REPO_ID}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_models()
    if success:
        print("\n" + "="*60)
        print("Download complete! You can now use the models.")
        print("="*60)
