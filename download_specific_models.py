"""
Download specific models from Hugging Face Hub
Use this script if you only need certain model architectures or scenarios.
"""

from huggingface_hub import hf_hub_download
import os
import sys

# Repository information
REPO_ID = "Bawil/wmh_leverage_normal_abnormal_segmentation"

# Available models (update based on your actual structure)
AVAILABLE_MODELS = {
    "1": {
        "name": "unet",
        "scenarios": {
            "1": "unet/models/scenario1_binary_model.h5",
            "2": "unet/models/scenario2_multiclass_model.h5"
        }
    },
    "2": {
        "name": "attention_unet",
        "scenarios": {
            "1": "attention_unet/models/scenario1_binary_model.h5",
            "2": "attention_unet/models/scenario2_multiclass_model.h5"
        }
    },
    "3": {
        "name": "deeplabv3plus",
        "scenarios": {
            "1": "deeplabv3plus/models/scenario1_binary_model.h5",
            "2": "deeplabv3plus/models/scenario2_multiclass_model.h5"
        }
    },
    "4": {
        "name": "transunet",
        "scenarios": {
            "1": "transunet/models/scenario1_binary_model.h5",
            "2": "transunet/models/scenario2_multiclass_model.h5"
        }
    }
}

def display_menu():
    """Display available models"""
    print("\n" + "="*60)
    print("Available Models:")
    print("="*60)
    for key, model in AVAILABLE_MODELS.items():
        print(f"\n{key}. {model['name']}")
        for s_key, s_path in model['scenarios'].items():
            print(f"   - Scenario {s_key}: {os.path.basename(s_path)}")
    print("\n" + "="*60)

def download_specific_model(model_id, scenario_id=None):
    """Download a specific model and scenario"""
    
    if model_id not in AVAILABLE_MODELS:
        print(f"Error: Model ID '{model_id}' not found!")
        return False
    
    model_info = AVAILABLE_MODELS[model_id]
    
    # If scenario not specified, download all scenarios for this model
    if scenario_id is None:
        scenarios_to_download = model_info['scenarios'].items()
    elif scenario_id in model_info['scenarios']:
        scenarios_to_download = [(scenario_id, model_info['scenarios'][scenario_id])]
    else:
        print(f"Error: Scenario '{scenario_id}' not found for {model_info['name']}!")
        return False
    
    print(f"\nDownloading {model_info['name']}...")
    
    for s_id, file_path in scenarios_to_download:
        try:
            print(f"\n  Downloading Scenario {s_id}...")
            
            # Create local directory structure
            local_path = os.path.join("models", os.path.dirname(file_path))
            os.makedirs(local_path, exist_ok=True)
            
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            print(f"  ✓ Downloaded to: {downloaded_path}")
            
        except Exception as e:
            print(f"  ✗ Error downloading: {str(e)}")
            return False
    
    return True

def interactive_download():
    """Interactive mode for downloading models"""
    
    display_menu()
    
    print("\nOptions:")
    print("- Enter model number (1-4) to download all scenarios for that model")
    print("- Enter 'model_number,scenario_number' to download specific scenario (e.g., '1,1')")
    print("- Enter 'all' to download all models")
    print("- Enter 'q' to quit")
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            break
        
        if choice == 'all':
            print("\nDownloading all models...")
            for model_id in AVAILABLE_MODELS.keys():
                download_specific_model(model_id)
            print("\n✓ All downloads complete!")
            break
        
        if ',' in choice:
            parts = choice.split(',')
            if len(parts) == 2:
                model_id, scenario_id = parts[0].strip(), parts[1].strip()
                if download_specific_model(model_id, scenario_id):
                    print(f"\n✓ Successfully downloaded {AVAILABLE_MODELS[model_id]['name']} - Scenario {scenario_id}")
            else:
                print("Invalid format! Use 'model_number,scenario_number' (e.g., '1,1')")
        else:
            if download_specific_model(choice):
                print(f"\n✓ Successfully downloaded all scenarios for {AVAILABLE_MODELS[choice]['name']}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        # Command line mode
        if sys.argv[1] == 'all':
            for model_id in AVAILABLE_MODELS.keys():
                download_specific_model(model_id)
        elif ',' in sys.argv[1]:
            model_id, scenario_id = sys.argv[1].split(',')
            download_specific_model(model_id.strip(), scenario_id.strip())
        else:
            download_specific_model(sys.argv[1])
    else:
        # Interactive mode
        interactive_download()
