# Incorporating Normal Periventricular Changes for Enhanced Pathological White Matter Hyperintensity Segmentation: On Multi-Class Deep Learning Approaches

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![Paper Status](https://img.shields.io/badge/paper-under%20review-blue.svg)](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation)
[![Models on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Bawil/wmh_leverage_normal_abnormal_segmentation)

## Overview

This repository implements a novel **three-class deep learning approach** for white matter hyperintensity (WMH) segmentation that explicitly distinguishes between **normal aging-related hyperintensities** and **pathologically significant lesions**. Our method addresses a fundamental limitation of current automated segmentation approaches that suffer from false positive detection in periventricular regions.

### Key Innovation

Traditional WMH segmentation methods use binary classification (background vs abnormal WMH), treating all hyperintensities equally. Our three-class approach introduces an intermediate category:

- **Class 0**: Background
- **Class 1**: Normal WMH (periventricular normal changes (aging-related or CSF-contaminated))  
- **Class 2**: Abnormal WMH (pathologically significant lesions)

This enables models to learn distinguishing features between normal aging-related changes and pathologically significant lesions, directly addressing the clinical challenge of false positive detection in periventricular regions.

### Performance Highlights

**U-Net (Best Performing Architecture)**
- **+27.1% increase** in Dice coefficient (0.768 vs 0.497)
- **2.4 Hausdorff Distance enhancement** (27.4 vs 29.8)
- **Statistical significance**: p < 0.0001, Cohen's d = 0.5643

**Key Findings Across All Models**
- All four architectures showed measurable improvements with three-class training
- Traditional CNN-based models benefited more than transformer-based approaches
- Maintained clinical feasibility with 1.5-second processing time (for 40-slice workflows)

## Architecture Comparison

We evaluated four modern baseline deep learning architectures:

| Architecture | Scenario 1 (Binary) | Scenario 2 (Three-Class) | Improvement | Statistical Significance |
|--------------|---------------------|---------------------------|-------------|-------------------------|
| **U-Net** | Dice: 0.497 | Dice: 0.768 | **+0.271** | p < 0.0001 |
| **Attention U-Net** | Dice: 0.486 | Dice: 0.740 | **+0.253** | p < 0.0001 |
| **DeepLabV3Plus** | Dice: 0.374 | Dice: 0.586 | **+0.212** | p < 0.0001 |
| **Trans-U-Net** | Dice: 0.510 | Dice: 0.700 | **+0.190** | p < 0.0001 |

## Dataset and Methodology

### Clinical Dataset
- **Local Dataset: 100 MS patients** (2,000 FLAIR MRI images)
- **Demographics**: 26 males, 74 females (age range 18-68 years)
- **Scanner**: 1.5-Tesla TOSHIBA Vantage
- **Public Dataset: MSSEG2016: 15 patients** (750 FLAIR MRI images)
- **Expert Annotations**: Board-certified neuroradiologists (20+ years experience)

### Training Configuration
- **Data Split**: 80/10/10 local patients, 9/3/3 public patients (train/validation/test)
- **Preprocessing**: Noise reduction, Intensity normalization, 256×256 resizing
- **Training**: 50 epochs, Adam optimizer, early stopping
- **Loss Functions**: 
  - Scenario 1: Weighted binary cross-entropy
  - Scenario 2: Weighted categorical cross-entropy

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation.git
cd wmh-normal-abnormal-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pre-trained Models

All pre-trained models are hosted on Hugging Face. Download them before running experiments:

```bash
python download_models.py  # Downloads all 8 models (~2.5 GB)
```

See the [Pre-trained Models](#pre-trained-models) section below for more download options.

### Basic Usage

```python
import sys
sys.path.append('src')

from models.unet import UNet
from preprocessing.data_loader import load_flair_image
from training.predict import predict_wmh

# Load and preprocess FLAIR image
image_path = "path/to/your/flair.nii.gz"
preprocessed_image = load_flair_image(image_path)

# Load pre-trained model (three-class)
model = UNet(input_shape=(256, 256, 1), num_classes=3)
model.load_weights('models/unet/models/scenario2_multiclass_model.h5')

# Run prediction
prediction = predict_wmh(model, preprocessed_image)

# Results contain:
# - Class 0: Background
# - Class 1: Normal WMH (periventricular)
# - Class 2: Abnormal WMH (pathological lesions)
```

## Pre-trained Models

Due to the large size of the trained model files (>300MB each), all models are hosted on **Hugging Face Hub** 🤗 for easy access and version control.

**🤗 Hugging Face Repository:** [Bawil/wmh_leverage_normal_abnormal_segmentation](https://huggingface.co/Bawil/wmh_leverage_normal_abnormal_segmentation)

### Quick Download

#### Option 1: Download All Models (Recommended)
```bash
pip install huggingface_hub
python download_models.py
```

This downloads all 8 pre-trained models (~2.5 GB total) into the proper directory structure.

#### Option 2: Download Specific Models
```bash
# Interactive mode with menu
python download_specific_models.py

# Command line examples:
python download_specific_models.py 1      # Download all U-Net models
python download_specific_models.py 1,1    # Download U-Net Scenario 1 only
python download_specific_models.py all    # Download everything
```

### Available Model Architectures

We provide pre-trained weights for all four architectures evaluated in our study:

| Model | Parameters | Scenario 1 (Binary) | Scenario 2 (Three-Class) |
|-------|-----------|---------------------|---------------------------|
| **U-Net** | 31.1M | ✓ Available | ✓ Available (Recommended) |
| **Attention U-Net** | 31.4M | ✓ Available | ✓ Available |
| **DeepLabV3Plus** | 40.3M | ✓ Available | ✓ Available |
| **TransUNet** | 38.7M | ✓ Available | ✓ Available |

### Model Repository Structure

```
models/
├── unet/models/
│   ├── scenario1_binary_model.h5          # Binary classification (Dice: 0.497)
│   └── scenario2_multiclass_model.h5      # Three-class (Dice: 0.768) ⭐ Best
├── attention_unet/models/
│   ├── scenario1_binary_model.h5          # Binary classification (Dice: 0.486)
│   └── scenario2_multiclass_model.h5      # Three-class (Dice: 0.740)
├── deeplabv3plus/models/
│   ├── scenario1_binary_model.h5          # Binary classification (Dice: 0.374)
│   └── scenario2_multiclass_model.h5      # Three-class (Dice: 0.586)
└── transunet/models/
    ├── scenario1_binary_model.h5          # Binary classification (Dice: 0.510)
    └── scenario2_multiclass_model.h5      # Three-class (Dice: 0.700)
```

### Training Scenarios

Each architecture is trained under two scenarios:

**Scenario 1: Binary Classification**
- **Classes**: Background (0) vs Abnormal WMH (1)
- **Use Case**: Traditional WMH segmentation baseline
- **Training Data**: Abnormal WMH samples only

**Scenario 2: Three-Class Classification** (Recommended)
- **Classes**: Background (0), Normal WMH (1), Abnormal WMH (2)
- **Use Case**: Enhanced pathological lesion detection with reduced false positives
- **Training Data**: Both normal periventricular changes and pathological lesions
- **Improvement**: Up to +27.1% Dice coefficient increase

### Model Details

- **Framework**: TensorFlow 2.11 / Keras
- **Format**: HDF5 (.h5)
- **Input Shape**: 256×256×1 (grayscale FLAIR MRI)
- **Output**: 
  - Scenario 1: 256×256×2 (background, abnormal)
  - Scenario 2: 256×256×3 (background, normal, abnormal)
- **Training Dataset**: 100 MS patients (2,000 FLAIR images) + MSSEG2016 (15 patients)
- **Validation**: Patient-level stratified cross-validation

### Usage Example

```python
from tensorflow.keras.models import load_model
import numpy as np

# Load the best performing model (U-Net Scenario 2)
model = load_model('models/unet/models/scenario2_multiclass_model.h5')

# Prepare your preprocessed FLAIR image (256x256x1)
# input_image shape: (batch_size, 256, 256, 1)

# Run inference
predictions = model.predict(input_image)

# Get class predictions
predicted_classes = np.argmax(predictions, axis=-1)
# Class 0: Background
# Class 1: Normal WMH (periventricular)
# Class 2: Abnormal WMH (pathological lesions)

# Extract only pathological lesions (Class 2)
abnormal_wmh_mask = (predicted_classes == 2).astype(np.uint8)
```

### Storage Requirements

- **Total repository size**: ~2.5 GB
- **Per model file**: ~300-400 MB
- **Total models**: 8 (4 architectures × 2 scenarios)
- **Disk space needed**: Minimum 3-4 GB (including extraction)

### Performance Comparison

| Architecture | Scenario 1 Dice | Scenario 2 Dice | Improvement | p-value |
|--------------|-----------------|-----------------|-------------|---------|
| U-Net ⭐ | 0.497 | **0.768** | **+54.5%** | <0.0001 |
| Attention U-Net | 0.486 | 0.740 | +52.1% | <0.0001 |
| TransUNet | 0.510 | 0.700 | +37.3% | <0.0001 |
| DeepLabV3Plus | 0.374 | 0.586 | +56.7% | <0.0001 |

⭐ **Recommended**: U-Net with Scenario 2 (three-class) for best performance

### Troubleshooting

**Issue: Download fails or times out**
```bash
# Try manual download from Hugging Face web interface
# Or download specific models instead of all at once
python download_specific_models.py 1  # Just U-Net first
```

**Issue: Out of memory when loading model**
```python
# Use model quantization or load on CPU
import tensorflow as tf
with tf.device('/CPU:0'):
    model = load_model('models/unet/models/scenario2_multiclass_model.h5')
```

**Issue: Wrong predictions**
- Ensure input is properly preprocessed (normalized, resized to 256×256)
- Check input shape: (batch_size, 256, 256, 1)
- Verify pixel values are in correct range [0, 1] or normalized

### Model Provenance

All models were trained with:
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: 
  - Scenario 1: Weighted binary cross-entropy
  - Scenario 2: Weighted categorical cross-entropy
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 8
- **Hardware**: NVIDIA RTX 3060 (12GB)
- **Training Time**: 2-3 hours per model
- **Validation Strategy**: Patient-level stratified 80/10/10 split

## Repository Structure

```
wmh-normal-abnormal-segmentation/
├── article_tables_figures/
│   ├── Figure_1.tif
│   ├── Figure_2.tif
│   ├── Figure_3.tif
│   ├── Figure_4.tif
│   ├── Table_1.png
│   ├── Table_2.png
│   └── Table_3.png
├── src/
│   ├── models/                    # Model architectures
│   │   ├── unet.py
│   │   ├── attention_unet.py
│   │   ├── deeplabv3plus.py
│   │   ├── transunet.py
│   │   ├── losses.py
│   │   └── metrics.py
│   ├── preprocessing/             # Data preprocessing
│   │   ├── preprocessing.py
│   │   └── normalization.py
│   ├── training/                  # Training framework
│   │   └── train.py
│   └── inference/                 # Prediction tools
│       └── predict.py
├── models/                        # Pre-trained models (download via HF)
│   ├── unet/models/
│   ├── attention_unet/models/
│   ├── deeplabv3plus/models/
│   └── transunet/models/
├── data/                          # Sample data
│   ├── test/
│   └── train/
├── results/                       # Evaluation results
├── docs/                          # Documentation
├── download_models.py             # Download all models from HF
├── download_specific_models.py    # Download specific models
└── setup_dirs.sh                  # Setup directory structure
```

## Clinical Impact

### Addressing Current Limitations
- **False Positive Reduction**: Significant decrease in periventricular false positives
- **Clinical Applicability**: Enhanced diagnostic accuracy for routine clinical use
- **Expert Validation**: Reduces manual radiologist correction requirements
- **Disease Burden Quantification**: More accurate longitudinal monitoring

### Clinical Validation Results
- **Precision Improvement**: Up to 0.271 enhancement in pathological lesion detection (Dice)
- **Specificity Enhancement**: Better discrimination between normal and abnormal hyperintensities
- **Processing Efficiency**: 3.5x increase in inference time but remains clinically feasible (1.5 seconds total for 40-slices workflow)

## Statistical Analysis

Our comprehensive evaluation includes:

- **Paired t-tests** for performance comparison between scenarios
- **Cohen's d effect size** calculations for practical significance
- **95% confidence intervals** for all metrics
- **Wilcoxon signed-rank test** for non-parametric validation

### Key Statistical Findings
- **U-Net**: Cohen's d = 0.5643 (large effect size)
- **Attention U-Net**: Cohen's d = 0.4419 (medium effect size)
- **DeepLabV3Plus**: Cohen's d = 0.5653 (large effect size)
- **Trans-U-Net**: Cohen's d = 0.4778 (medium effect size)

## Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or equivalent
- **CPU**: Intel Core i7-7700K (8 cores) or equivalent
- **RAM**: 64GB DDR4 (minimum 16GB)
- **Software**: TensorFlow 2.11, CUDA 11.8, Python 3.9

### Computational Performance
- **Training Time**: 2-3 hours per model per scenario
- **Inference Time**: 1.5 seconds per case (including preprocessing)
- **Memory Usage**: Batch size 8 (GPU memory constrained)
- **Parameter Count**: 31.0M (U-Net) to 38.7M (TransUNet)

### Sample Data
We provide anonymized sample FLAIR images with corresponding ground truth annotations for:
- Normal WMH in periventricular regions
- Abnormal WMH (MS lesions)
- Expert annotation examples

## Research Applications

### Academic Use
- **Reproducible Research**: Complete implementation with documented methodology
- **Baseline Comparisons**: Standardized evaluation framework
- **Method Development**: Foundation for advanced WMH segmentation approaches
- **Cross-validation**: Patient-level stratified evaluation

### Clinical Translation
- **Workflow Integration**: Compatible with standard clinical FLAIR protocols
- **Real-time Processing**: Suitable for same-session clinical decision making
- **Multi-center Validation**: Framework for broader clinical studies
- **Quality Assurance**: Systematic performance monitoring tools

## Future Directions

### Immediate Extensions
- **Multi-center Validation**: Evaluation across different scanner types and protocols
- **Longitudinal Studies**: Disease progression monitoring capabilities
- **Multi-modal Integration**: Incorporation of additional MRI sequences
- **Uncertainty Quantification**: Confidence measures for clinical predictions

### Advanced Development
- **Model Compression**: Optimization for clinical deployment
- **Active Learning**: Strategies for continuous model improvement
- **Clinical Decision Support**: Integration with automated reporting systems
- **Real-world Validation**: Prospective clinical studies

## Documentation

- **[Installation Guide](docs/INSTALLATION.md)**: Detailed setup instructions
- **[Usage Tutorial](docs/USAGE.md)**: Step-by-step usage guide  
- **[Methodology](docs/METHODOLOGY.md)**: Detailed three-class approach explanation
- **[Clinical Guide](docs/CLINICAL_GUIDE.md)**: Clinical interpretation and validation
- **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Common issues and solutions

## Contributing

We welcome contributions from the research and clinical communities! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute
- **Bug Reports**: Use GitHub issues for bug reports
- **Feature Requests**: Suggest enhancements via issues
- **Code Contributions**: Submit pull requests with improvements
- **Clinical Validation**: Share results from your clinical datasets
- **Documentation**: Help improve documentation and examples

## Citation

If you use this work in your research, please cite:

```bibtex
@article{bawil2025wmh,
  title={Incorporating Normal Periventricular Changes for Enhanced Pathological White Matter Hyperintensity Segmentation: On Multi-Class Deep Learning Approaches},
  author={Bawil, Mahdi Bashiri and Shamsi, Mousa and Jafargholkhanloo, Ali Fahmi and Bavil, Abolhassan Shakeri and Jafargholkhanloo, Ali Fahmi},
  journal={},
  year={2025},
  note={Code: https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation, Models: https://huggingface.co/Bawil/wmh_leverage_normal_abnormal_segmentation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Golgasht Medical Imaging Center**, Tabriz, Iran for providing the clinical dataset
- **Expert neuroradiologist** for manual annotations and clinical validation
- **Sahand University of Technology** and **Tabriz University of Medical Sciences** for institutional support
- **Ethics Committee** for approval (IR.TBZMED.REC.1402.902)

## Contact and Support

- **Primary Author**: Mahdi Bashiri Bawil (m_bashiri99@sut.ac.ir)
- **Repository**: [https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation)
- **Models**: [https://huggingface.co/Bawil/wmh_leverage_normal_abnormal_segmentation](https://huggingface.co/Bawil/wmh_leverage_normal_abnormal_segmentation)
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation/discussions)

---

**Keywords**: White matter hyperintensities, deep learning, medical image segmentation, FLAIR MRI, multi-class classification, U-Net, pathological segmentation, neuroimaging, clinical validation
