# 🔬 Research & Development Log

This directory archives the experimental notebooks, exploratory data analysis (EDA), and data pipelines that demonstrate the engineering process behind PulsePod.

> **Note:** These notebooks contain the original research code, training logs, and loss curves. They serve as proof of work for the results presented in the project report.

## 📂 Notebooks Overview

### 1. `01_exploratory_data_analysis.ipynb`
* **Focus:** Initial investigation of the MIT-BIH and PTB-XL databases.
* **Key Findings:**
    Visualized the **Generalization Gap** and domain shift between ambulatory (MIT-BIH) and clinical (PTB-XL) signals.
    * Analyzed the class imbalance (Normal vs. Abnormal) across 48 records.

### 2. `02_dataset_creation_pipeline.ipynb`
**Focus:** Building the "Unified Dataset".
* **Methodology:**
    **Preprocessing:** Applied **NeuroKit2** for R-peak detection and beat segmentation.
    **Standardization:** Fixed window segmentation (187 samples) to align diverse signal sources.
    **Merging:** Combined 54,680 MIT-BIH beats with 250,948 PTB-XL beats.

### 3. `03_data_augmentation_stratification.ipynb`
* **Focus:** Solving class imbalance and enhancing robustness.
* **Techniques:**
    **Stratified Splitting:** Ensured a consistent 70/15/15 split across both datasets.
    **Augmentation:** Applied **Gaussian noise** and **time-shifting** to the 'Abnormal' class to prevent overfitting.

### 4. `04_model_training_experiments.ipynb`
* **Focus:** Training and validation of Deep Learning architectures.
* **Experiments:**
    **Architectures:** 1D-CNN, CNN-LSTM, and Spiking Neural Networks (SNN).
    **Optimization:** Used **Focal Loss** to penalize misclassifications in the minority class.
    **Results:** Contains the generated loss curves and confusion matrices validating the 82% generalization score on PTB-XL.

---

## 🛠️ Usage
These notebooks are intended for **educational and reproducibility purposes**.
* To view the training history and graphs, open the files in GitHub (the outputs are preserved).
* To run them locally, ensure you have the research dependencies installed (see `requirements_cnn.txt`).