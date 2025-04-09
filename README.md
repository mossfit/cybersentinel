# CyberSentinel: A Transparent Defense Framework for Malware Detection

This repository is our research paper implementation **CyberSentinel**, a sophisticated malware detection framework leveraging deep learning techniques, particularly designed for high-stakes operational environments. CyberSentinel employs a dual-branch architecture combining DenseNet with advanced spatial attention modules and auxiliary attention mechanisms, optimized through Quantum Swarm Hyperparameter Optimization (QSHO).

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Contributions](#key-contributions)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [References](#references)
- [License](#license)

## Overview
CyberSentinel transforms binary malware files into image representations to leverage advanced image classification models. Its two-branch network utilizes asymmetric spatial attention and an auxiliary attention branch, enhancing feature extraction and malware classification accuracy significantly.

## Project Structure
```
CyberSentinel/
├── dataset/
│   ├── raw_pe_files/
│   └── images/
├── preprocessing/
│   ├── pe_image_conversion.py
│   └── data_augmentation.py
├── networks/
│   ├── backbone.py
│   ├── asam_module.py
│   ├── auxiliary_branch.py
│   └── cybersentinel_model.py
├── optimization/
│   └── quantum_swarm.py
├── training/
│   ├── trainer.py
│   └── checkpoints/
├── evaluation/
│   ├── metrics.py
│   └── visualizations.py
├── utils/
│   ├── data_loader.py
│   └── config.py
├── main.py
└── requirements.txt
```

## Key Contributions
- **Two-Branch Attention Network**: Utilizes DenseNet121 backbone integrated with Asymmetric Spatial Attention Modules (ASAM).
- **Auxiliary Attention Branch**: Specifically captures features missed due to obfuscation and artifacts in malware binaries.
- **Quantum Swarm Hyperparameter Optimization (QSHO)**: Efficiently optimizes hyperparameters to enhance model performance.
- **Robust Evaluation**: Demonstrated superior performance across Malimg, Microsoft Malware, and BODMAS-14 datasets.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- scikit-learn
- Matplotlib
- Pillow
- tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Setup and Installation
Clone this repository:

```bash
git clone https://github.com/mossfit/CyberSentinel.git
cd CyberSentinel
```

## Data Preparation
Convert Portable Executable (PE) malware files into images:
- Place PE files into `dataset/raw_pe_files/`
- Run:
- 
```bash
python preprocessing/pe_image_conversion.py
```
This generates RGB images in `dataset/images/`.

## Model Training
Train the CyberSentinel model:
```bash
python main.py
```
Adjust training parameters within `utils/config.py`.

## Evaluation and Visualization
Evaluate model performance and visualize results:
```bash
python evaluation/visualizations.py
```
This provides accuracy, precision, recall, F1-score, confusion matrices, and ROC curves.

## References
Basak, M.; Han, M.-M. CyberSentinel: A Transparent Defense Framework for Malware Detection in High-Stakes Operational Environments. *Sensors* **2024**, *24*, 3406. [doi:10.3390/s24113406](https://doi.org/10.3390/s24113406)

## License
![MIT](#MIT)
