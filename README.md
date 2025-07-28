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

| File Name                          | Description |
|-----------------------------------|-------------|
| `Autoencoder_BayesianOptimization.py` | Autoencoder architecture and Bayesian optimization for latent feature extraction |
| `Clustering.py`                   | Agglomerative clustering and visualization of origin subdivisions |
| `Classification.py`              | Supervised classification of origin clusters using Random Forest |
| `README.md`                       | Description of repository contents, usage, and citation |

---

## Software Metadata

- **Software name**: Python  
- **Programming language**: Python (version 3.12)  
- **Developers**: Subi Lee, Jina Jeong  
- **Contact**: jeong.j@knu.ac.kr

---

## Citation

If you use this code, please cite:

> Lee et al. (2025).  
> *Utilizing deep learning-based feature engineering for effective geographical origin subdivision and classification of environmental soil samples in South Korea.*  
> Ecological Informatics (under review).

---

## Data Availability

Due to data sensitivity, the full dataset used in this study is not publicly shared.  
However, to support reproducibility, the following files are included:

- `sample_data.csv`: anonymized example dataset with the same structure as the original data  
- `data_description.txt`: description of each variable (unit, meaning, data type, etc.)

---

## Reproducibility

This repository adheres to the open science principles recommended by *Ecological Informatics*, including:
- Transparent methodology
- Reproducible analysis with open-source code
- Documentation of data structure and processing logic

For questions or collaboration inquiries, please contact: jeong.j@knu.ac.kr

---
