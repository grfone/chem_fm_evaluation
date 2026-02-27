# CCS Prediction Pipeline

This repository contains a Python-based pipeline for processing chemical datasets, generating molecular embeddings (UniMol, ChemBERTa, MolFormer, MolMIM, and fingerprints), training machine learning models for Collision Cross Section (CCS) prediction, computing statistics, and generating visualizations.

The pipeline automates the full workflow: dataset download, format unification, embedding generation, classification, model training (LR and MLP), statistical analysis, and visualization.

> **Note:** GPU resources are strongly recommended. Some steps (e.g., training) can be very time-consuming.

---

## Prerequisites

- **Python** 3.12+ (via Conda)
- **Conda**
- **Docker**
- **NVIDIA GPU** with CUDA
- **NGC API Key** (for MolMIM)

---

## Installation

### Clone the Repository
git clone https://github.com/grfone/metlin_foundation.git

cd metlin_foundation

### Set Up Conda Environment
conda env create -f environment.yml  
conda activate pytorch+_env

### Set Up Docker for MolMIM
- Install Docker
- Install NVIDIA Container Toolkit
- Set environment variables:

export NGC_API_KEY=your_ngc_api_key_here  
export LOCAL_NIM_CACHE=/path/to/cache

Docker image:
nvcr.io/nim/nvidia/molmim:1.0.0

---

### Usage

Run the full pipeline:
python main.py

### What main.py Does

1. Environment setup (KERAS_BACKEND=torch)
2. Dataset download & extraction
3. Format unification and filtering
4. Dataset preparation
5. Embedding generation
6. Compound classification (ClassyFire)
7. Model training (5-fold CV)
8. Statistical analysis (LaTeX output)
9. Visualization (UMAP)

---

## Outputs

- resources/ – processed datasets & embeddings
- results/ – model metrics, predictions, statistics, etc.

Subsequent runs skip completed steps automatically.

---

## Troubleshooting

- Docker failures: check docker logs <container_id> and GPU access
- Memory errors: reduce batch sizes in src/Encoders.py
- ClassyFire API issues: automatic fallback enabled
- Conda issues:
conda env create -f environment.yml --solver=libmamba

---

## Project Structure
```
metlin_foundation/
├── main.py
├── environment.yml
├── src/
│   ├── FileManager.py
│   ├── Encoders.py
│   ├── CCSTrainer.py
│   ├── CCSTrainer_extra/
│   ├── Statistics.py
│   └── Visualization.py
├── resources/
└── results/
```
---

## Contributing

- Fork the repository
- Create a feature branch
- Submit a pull request

---

## License

Open Access.

---

## Acknowledgments

- RDKit, UniMol, Hugging Face, PyTorch, Keras
- Public datasets: CCSBase, METLIN, SMRT, in house μ<sub>eff</sub> library
- Cheminformatics research for CCS prediction