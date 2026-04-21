# Rethinking Cross-View Variance: A Preventative Denoising Strategy for FastGS

## 📖 Project Overview
This repository contains the official implementation for the DSAI 5207 final project: **"Rethinking Cross-View Variance: A Preventative Denoising Strategy for FastGS"**.

Building upon the ultra-fast training framework of FastGS, we introduce a **Variance-Aware Densification (VAD)** strategy. By integrating Welford's online algorithm, our method tracks the cross-view L1 loss variance in real-time during the forward pass with minimal memory overhead. It uses this variance as a precise noise indicator to intercept and prevent the generation of multi-view inconsistent geometric noise ("floaters") at their source.

---

## 📂 Repository Structure
The repository is organized to facilitate easy navigation and straightforward reproducibility of the core results:

```text
VAD-FastGS/
├── train.py                  # Main training script incorporating the VAD pipeline
├── scene/
│   └── gaussian_model.py     # Gaussian model definition (contains Welford states and VAD pruning logic)
├── utils/
│   ├── fast_utils.py         # Variance-based continuous error mapping and metric calculation
│   └── visualize.py          # Variance heatmap generation (used for report visualizations)
├── submodules/               # Customized CUDA rasterization and KNN dependencies
├── train_shiny.sh            # Core reproduction script for the ShinyBlender dataset
├── full_eval.py              # Full evaluation script (computes PSNR, SSIM, LPIPS)
└── render.py                 # Novel view rendering script
```

---

## ⚙️ 1. Setup & Dependencies

### Hardware Requirements
* **OS:** Ubuntu 20.04 / 22.04 or Windows 11
* **GPU:** Minimum 8GB VRAM (All experiments in the report were conducted on a single NVIDIA RTX 4060Ti)
* **CUDA:** >= 11.8

### Installation Instructions
We strongly recommend using Conda to manage your environment:

```bash
# 1. Create and activate the Conda environment
conda create -n vad_fastgs python=3.8
conda activate vad_fastgs

# 2. Install PyTorch (Adjust the CUDA version based on your system)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install required Python packages
pip install plyfile tqdm scipy wandb opencv-python matplotlib

# 4. Compile and install custom C++/CUDA submodules (Crucial step)
pip install ./submodules/diff-gaussian-rasterization_fastgs
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

---

## 📊 2. Data Preparation

Our experiments and evaluations are conducted on the **ShinyBlender dataset**, which features highly reflective materials and challenging view-dependent effects, making it ideal for testing artifact reduction and specularity preservation.

1. **Download:** Download the ShinyBlender dataset from the [official Ref-NeRF release page](https://github.com/google-research/multinerf).
2. **Organization:** Extract the dataset into a `data/` directory outside the main code repository. Ensure the structure looks like this:

```text
<Your_Data_Path>/ShinyBlender/
    ├── ball/
    ├── car/
    ├── coffee/
    ├── helmet/
    ├── teapot/
    └── toaster/
```

---

## 🚀 3. Running & Reproducing Results

All quantitative results presented in the project report (specifically **Table 1**) can be fully reproduced using the provided scripts.

### 3.1 One-Click Reproduction (Table 1)
To reproduce the experimental results regarding **#Gaussians**, **Training Time**, **PSNR**, **SSIM**, and **LPIPS** across all evaluated scenes, run the provided bash script:

```bash
# IMPORTANT: Open train_shiny.sh and update the path to match your <Your_Data_Path>/ShinyBlender/ directory before running.
bash train_shiny.sh
```
*Note: This script will sequentially train the model on the Ball, Car, Coffee, Helmet, Teapot, and Toaster scenes.*

### 3.2 Manual Training for a Single Scene
To run the training pipeline on a specific scene (e.g., the `toaster` scene), use the following command:

```bash
python train.py -s <Your_Data_Path>/ShinyBlender/toaster -m outputs/toaster_vad
```

### 3.3 Rendering and Quantitative Evaluation
Once training is complete, you can render the test set and compute the quantitative metrics:

```bash
# Render test views
python render.py -m outputs/toaster_vad

# Compute PSNR, SSIM, and LPIPS
python full_eval.py -m outputs/toaster_vad
```
The final evaluation metrics will be saved to `outputs/toaster_vad/results.json`.

---

## 🔍 4. Code Mapping to the Report

To assist reviewers in verifying the implementation, here is how the core components in the code map to the sections in the project report (*Rethinking Cross-View Variance*):

* **Incremental Cross-View Variance Tracking (Section 3.1):**
  * State initialization (`error_running_mean`, `error_running_M2`, `view_observe_count`): Located in `scene/gaussian_model.py` within the `training_setup()` method.
  * Welford's online update logic: Located in `utils/fast_utils.py` within the `compute_gaussian_score_fastgs()` function.
* **Variance-Aware Densification (VAD) (Section 3.2):**
  * The preventative noise filtering mechanism and normalized variance threshold logic ($\tau_{var}=0.8$): Implemented in `scene/gaussian_model.py` within the `densify_and_prune_fastgs()` function.
* **Empirical Visualizations (Figure 1 & Figure 2):**
  * The code responsible for computing true variance and generating the 2D variance heatmaps for the report is located in `utils/visualize.py` within the `compute_variance_and_save_heatmap()` function.

---
