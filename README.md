# Geographical Origin Subdivision of Environmental Soil Samples

This repository contains source codes used for the geographical origin subdivision and classification of environmental soil samples in South Korea.  
The approach integrates unsupervised clustering and supervised classification using multivariate geochemical variables and strontium isotope ratios.

The repository includes three main Python scripts:
- `Autoencoder_BayesianOptimization.py`: Dimensionality reduction using autoencoder with Bayesian hyperparameter optimization.
- `Clustering.py`: Hierarchical clustering of the latent features to identify origin subdivisions.
- `Classification.py`: Supervised validation of clustering results using Random Forest classification based on geochemical and isotopic features.

The machine learning analysis was conducted using Python 3.12 and relevant packages including `scikit-learn`, `keras`, `tensorflow`, and `matplotlib`.

---

## Files Included

| File Name                           | Description |
|------------------------------------|-------------|
| `Autoencoder_BayesianOptimization.py` | Autoencoder architecture and Bayesian optimization for latent feature extraction |
| `Clustering.py`                    | Agglomerative clustering and visualization of origin subdivisions |
| `Classification.py`               | Classification of clustered origins for validation using Random Forest |
| `sample_data.csv`                 | Example dataset with anonymized structure identical to original |
| `data_description.txt`            | Description of data columns, units, and measurement methods |
| `README.md`                        | Repository description, instructions, and citation guide |

---

## Data

Due to data sensitivity, the full dataset used in this study is not publicly available.  
To support transparency and reproducibility, the following are provided:

- `sample_data.csv`: A synthetic dataset with the same structure and format as the original data
- `data_description.txt`: Contains metadata including variable names, units, descriptions, and data types

---

## Software Metadata

- **Software name**: Python  
- **Programming language**: Python (version 3.12)  
- **Developers**: Subi Lee, Jina Jeong  
- **Contact**: jeong.j@knu.ac.kr

---

## Citation

If you use this repository or its components, please cite:

> Lee et al. (2025).  
> *Utilizing deep learning-based feature engineering for effective geographical origin subdivision and classification of environmental soil samples in South Korea.*  
> *Ecological Informatics* (under review).

---

## Reproducibility

This repository adheres to the open science guidelines promoted by *Ecological Informatics*:

- Transparent and modular analysis pipeline  
- Reproducible code with version-controlled scripts  
- Documented data structure for synthetic and real datasets  
- Sample input data and metadata files included for demonstration  
- Full pipeline tested with Python 3.12 environment

If you have questions or seek collaboration, please contact: jeong.j@knu.ac.kr

---
