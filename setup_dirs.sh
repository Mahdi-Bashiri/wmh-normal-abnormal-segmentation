#!/bin/bash
# Setup model directory structure for Git
# This creates the directory structure while keeping models folder tracked in Git

echo "Creating model directory structure..."

# Create main models directory
mkdir -p models

# Create subdirectories for each model architecture
mkdir -p models/unet/models
mkdir -p models/attention_unet/models
mkdir -p models/deeplabv3plus/models
mkdir -p models/transunet/models

# Create .gitkeep files to track empty directories
touch models/.gitkeep
touch models/unet/.gitkeep
touch models/unet/models/.gitkeep
touch models/attention_unet/.gitkeep
touch models/attention_unet/models/.gitkeep
touch models/deeplabv3plus/.gitkeep
touch models/deeplabv3plus/models/.gitkeep
touch models/transunet/.gitkeep
touch models/transunet/models/.gitkeep

echo "✓ Directory structure created!"
echo ""
echo "Structure:"
echo "models/"
echo "├── unet/models/"
echo "├── attention_unet/models/"
echo "├── deeplabv3plus/models/"
echo "└── transunet/models/"
echo ""
echo "Run 'python download_models.py' to download all trained models."
