[README.md](https://github.com/user-attachments/files/22440966/README.md)
# Leveraging Normal White Matter Hyperintensity Context for Enhanced Pathological Segmentation via Multi-Class Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.11](https://img.shields.io/badge/TensorFlow-2.11-orange.svg)](https://tensorflow.org/)
[![Medical Imaging](https://img.shields.io/badge/domain-Medical%20Imaging-green.svg)](https://github.com/topics/medical-imaging)
[![Paper Status](https://img.shields.io/badge/paper-under%20review-blue.svg)](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation)

## Overview

This repository implements a novel **three-class deep learning approach** for white matter hyperintensity (WMH) segmentation that explicitly distinguishes between **normal aging-related hyperintensities** and **pathologically significant lesions**. Our method addresses a fundamental limitation of current automated segmentation approaches that suffer from false positive detection in periventricular regions.

### Key Innovation

Traditional WMH segmentation methods use binary classification (background vs abnormal WMH), treating all hyperintensities equally. Our three-class approach introduces an intermediate category:

- **Class 0**: Background
- **Class 1**: Normal WMH (periventricular normal changes (aging-related))  
- **Class 2**: Abnormal WMH (pathologically significant lesions)

This enables models to learn distinguishing features between normal aging-related changes and pathologically significant lesions, directly addressing the clinical challenge of false positive detection in periventricular regions.

### Performance Highlights

**U-Net (Best Performing Architecture)**
- **55.6% improvement** in Dice coefficient (0.693 vs 0.443)
- **131% precision enhancement** (0.662 vs 0.287)
- **Statistical significance**: p < 0.0001, Cohen's d = 0.971

**Key Findings Across All Models**
- All four architectures showed measurable improvements with three-class training
- Traditional CNN-based models benefited more than transformer-based approaches
- Maintained clinical feasibility with 1.5-second processing time

## Architecture Comparison

We evaluated four state-of-the-art deep learning architectures:

| Architecture | Scenario 1 (Binary) | Scenario 2 (Three-Class) | Improvement | Statistical Significance |
|--------------|---------------------|---------------------------|-------------|-------------------------|
| **U-Net** | Dice: 0.443 | Dice: 0.693 | **+55.6%** | p < 0.0001 |
| **Attention U-Net** | Dice: 0.410 | Dice: 0.665 | **+62.3%** | p < 0.0001 |
| **DeepLabV3Plus** | Dice: 0.379 | Dice: 0.553 | **+46.0%** | p < 0.0001 |
| **Trans-U-Net** | Dice: 0.619 | Dice: 0.686 | **+10.8%** | p = 0.250 |

## Dataset and Methodology

### Clinical Dataset
- **100 MS patients** (1,974 FLAIR MRI images)
- **Demographics**: 26 males, 74 females (age range 18-68 years)
- **Scanner**: 1.5-Tesla TOSHIBA Vantage
- **Expert Annotations**: Board-certified neuroradiologist (20+ years experience)
- **Inter-observer Agreement**: κ = 0.8 (substantial agreement)

### Training Configuration
- **Data Split**: 80/10/10 patients (train/validation/test)
- **Preprocessing**: Intensity normalization, brain extraction, 256×256 resizing
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
model.load_weights('models/unet/scenario2_multiclass/model_weights.h5')

# Run prediction
prediction = predict_wmh(model, preprocessed_image)

# Results contain:
# - Class 0: Background
# - Class 1: Normal WMH (periventricular)
# - Class 2: Abnormal WMH (pathological lesions)
```

### Pre-trained Models

We provide pre-trained models for all four architectures in both scenarios:

```
models/
├── unet/
│   ├── scenario1_binary/          # Binary classification
│   └── scenario2_multiclass/      # Three-class classification (recommended)
├── attention_unet/
├── deeplabv3plus/
└── transunet/
```

## Repository Structure

```
wmh-normal-abnormal-segmentation/
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
│   │   ├── data_loader.py
│   │   └── normalization.py
│   ├── training/                  # Training framework
│   │   ├── train.py
│   │   ├── config.py
│   │   └── scenario_manager.py
│   ├── inference/                 # Prediction tools
│   │   ├── predict.py
│   │   └── batch_predict.py
│   └── evaluation/                # Evaluation framework
│       ├── evaluator.py
│       ├── statistical_analysis.py
│       └── visualization.py
├── models/                        # Pre-trained models
├── data/                          # Sample data
├── results/                       # Evaluation results
└── docs/                          # Documentation
```

## Clinical Impact

### Addressing Current Limitations
- **False Positive Reduction**: Significant decrease in periventricular false positives
- **Clinical Applicability**: Enhanced diagnostic accuracy for routine clinical use
- **Expert Validation**: Reduces manual radiologist correction requirements
- **Disease Burden Quantification**: More accurate longitudinal monitoring

### Clinical Validation Results
- **Precision Improvement**: Up to 131% enhancement in pathological lesion detection
- **Specificity Enhancement**: Better discrimination between normal and abnormal hyperintensities
- **Processing Efficiency**: 3.5x increase in inference time but remains clinically feasible (1.5 seconds total)

## Statistical Analysis

Our comprehensive evaluation includes:

- **Paired t-tests** for performance comparison between scenarios
- **Cohen's d effect size** calculations for practical significance
- **95% confidence intervals** for all metrics
- **Wilcoxon signed-rank test** for non-parametric validation

### Key Statistical Findings
- **U-Net**: Cohen's d = 0.971 (large effect size)
- **Attention U-Net**: Cohen's d = 0.814 (large effect size)
- **DeepLabV3Plus**: Cohen's d = 0.503 (medium effect size)
- **Trans-U-Net**: Cohen's d = 0.123 (small effect size, not significant)

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
- **Parameter Count**: 31.0M (U-Net) to 40.3M (DeepLabV3Plus)

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
  title={Leveraging Normal White Matter Hyperintensity Context for Enhanced Pathological Segmentation via Multi-Class Deep Learning},
  author={Bawil, Mahdi Bashiri and Shamsi, Mousa and Bavil, Abolhassan Shakeri and Jafargholkhanloo, Ali Fahmi},
  journal={Iranian International Conference on Biomedical Engineering},
  year={2025},
  note={Code available at: https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation}
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
- **Issues**: [GitHub Issues](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Mahdi-Bashiri/wmh-normal-abnormal-segmentation/discussions)

---

**Keywords**: White matter hyperintensities, deep learning, medical image segmentation, FLAIR MRI, multi-class classification, U-Net, pathological segmentation, neuroimaging, clinical validation
