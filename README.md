# PhysioTransformer GUI

**Interactive Lactate Threshold Predictor (Demo)**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🖥️ Overview

An interactive desktop application for non-invasive lactate threshold ($HR_{2\text{mmol}\cdot\text{L}^{-1}}$) estimation using the PhysioTransformer deep learning model. 

Enter heart rate and performance data from an incremental exercise test to receive an instant non-invasive LT prediction with embedded attention weight visualization.

> **Note:** This GUI is a demonstration and screening tool. For rigorous scientific evaluation, cross-validation metrics, and ablation results, please refer to the main research repository:  
> 👉 [mehrnaz-ai/PhysioTransformer](https://github.com/mehrnaz-ai/PhysioTransformer)

---

## 📄 Paper & Preprint

> **Eskandarisani, M., & Daryanoosh, F.**  
> *Can AI Predict Lactate Thresholds Without Blood Samples? Deep Learning Succeeds Within Sports but Fails Across Movement Modalities.*  
> **Status:** Under Review   
> **Preprint DOI:** [10.21203/rs.3.rs-10811779/v1](https://doi.org/10.21203/rs.3.rs-10811779/v1)[cite: 48]

---

## ✨ Features

- **4 Endurance Sports Supported:** Running ($n=278$), Cycling ($n=279$), Rowing ($n=209$), and Kayaking ($n=53$).
- **Real-Time Prediction:** Enter stage-wise heart rate and workload sequences for immediate inference.
- **Attention Visualization:** Visualizes learned attention weights to highlight which test stages the model relies on most.
- **Modern Dark Theme:** Clean, professional cross-platform user interface.
- **Safe Checkpoint Verification:** Clear safeguards and warnings if model weights are missing.
- **Pre-filled Example Data:** Ships with a valid running test sequence for immediate out-of-the-box testing.

---

## ⚙️ Installation & Setup

### Requirements
- Python 3.11
- PyTorch 2.0.1 + dependencies (see `requirements.txt`)

### Quick Start Commands
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mehrnaz-ai/PhysioTransformer-GUI.git](https://github.com/mehrnaz-ai/PhysioTransformer-GUI.git)
   cd PhysioTransformer-GUI
