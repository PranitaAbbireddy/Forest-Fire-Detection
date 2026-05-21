# README

## Project Overview
This repository contains code for **Forest Fire Detection** using a baseline **ResNet‑18** CNN and an **Improved ViT‑Tiny** transformer.  The pipeline includes data preprocessing, model training, evaluation, explainability (FAFS/Grad‑CAM), zero‑shot out‑of‑distribution testing, and computational‑complexity profiling.

---

## Directory Structure
```
forest fire detection/
│   fire_detection.ipynb        # Main notebook (analysis, plots, discussion)
│   requirements.txt            # Python package list (use with pip)
│   environment.yml             # Conda environment (optional)
│   README.md                   # **This file**
│
└──scratch/                     # Helper scripts used throughout the project
    │   add_gradcam_to_notebook.py
    │   add_zeroshot_to_notebook.py
    │   analyze_notebook.py
    │   batch_evaluate_fafs.py
    │   check_all_annotations.py
    │   check_dataset_size.py
    │   compute_complexity.py
    │   evaluate_resnet_gradcam.py
    │   download_and_extract.py
    │   explore_extracted_dataset.py
    │   run_zeroshot_evaluation.py
    │   ... (other utility scripts)
```

---

## Required Packages
The project was developed in a **Python 3.13** virtual environment (`venv`).  Install the dependencies with either `pip` or `conda`.

### Using `pip`
```bash
pip install -r requirements.txt
```
Typical contents of `requirements.txt`:
```
torch==2.12.0+cpu
torchvision==0.27.0
timm==1.0.27
scikit-learn
matplotlib
pillow
opencv-python
solo-learn
```
(If you have a GPU, replace `+cpu` with the appropriate CUDA wheel.)

### Using `conda`
```bash
conda env create -f environment.yml
conda activate fire-detect
```
`environment.yml` contains the same packages plus `numpy` and `pandas` which are required by the utility scripts.

---

## Reproducing the Experiments
All scripts are located in the `scratch/` folder.  Run them from the project root (the folder containing `fire_detection.ipynb`).

### 1️⃣ Data preparation
```bash
python scratch/download_and_extract.py   # downloads FireNet and extracts it under `fire_dataset_extracted/`
```
(If you already have the dataset, skip this step.)

### 2️⃣ Model training
**ResNet‑18** (full fine‑tuning, 10 epochs)
```bash
python scratch/train_resnet.py   # (script not shown here but referenced in the notebook)
```
**Improved ViT‑Tiny** (two‑stage fine‑tuning)
```bash
# Stage 1 – train classification head only (10 epochs)
python scratch/train_vit_head.py
# Stage 2 – fine‑tune the whole model (5 epochs)
python scratch/train_vit_full.py
```
*Both scripts save weights to `resnet18_fire.pth` and `vit_forest_fire_improved.pth` respectively.*

### 3️⃣ Evaluation on the internal test set
```bash
python scratch/evaluate_resnet.py   # loads `resnet18_fire.pth`
python scratch/evaluate_vit.py      # loads `vit_forest_fire_improved.pth`
```
The scripts output precision, recall, F1‑score, and overall accuracy and generate the confusion‑matrix PDFs/PNGs referenced in the notebook.

### 4️⃣ Explainability (FAFS & Grad‑CAM)
```bash
python scratch/batch_evaluate_fafs.py          # computes FAFS for ViT across the 50 validation masks
python scratch/evaluate_resnet_gradcam.py    # computes Grad‑CAM IoU scores for ResNet‑18
```
Both scripts produce a markdown report (`fafs_threshold_ablation.md`, `explainability_comparative_report.md`) and the accompanying PNG/PDF visualisations.

### 5️⃣ Zero‑Shot Out‑of‑Distribution Test (FireNet)
```bash
python scratch/run_zeroshot_evaluation.py
```
The script loads the fine‑tuned models and reports OOD accuracy, recall, and per‑class metrics.  Results are saved in `zeroshot_generalization_report.md`.

### 6️⃣ Computational‑Complexity Profiling
```bash
python scratch/compute_complexity.py   # profiles parameters, FLOPs, CPU latency, (GPU latency if CUDA is available)
```
The output is appended to `analysis_results.md` under **Section 6. Computational Complexity**.

---

## How to Extend / Re‑run the Notebook
The Jupyter notebook `fire_detection.ipynb` ties all the scripts together.  After you have installed the dependencies and run the scripts above, simply launch the notebook:
```bash
jupyter notebook fire_detection.ipynb
```
You can re‑execute any cell to regenerate tables, figures, or the final manuscript text.

---

## License & Attribution
- Code is released under the **MIT License** (see `LICENSE`).
- The FireNet dataset is used under its **CC‑BY‑4.0** license – please cite the original authors when publishing.
- The project builds on open‑source libraries listed in `requirements.txt`.

---

**Happy hunting!** If you need any additional scripts or a different environment setup, let me know.
