import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram

# Fix random seed
seed_number = 42
np.random.seed(seed_number)
random.seed(seed_number)
tf.random.set_seed(seed_number)

# Load and preprocess data
os.chdir('./data')
data = pd.read_csv('input_data.csv', encoding='cp949')
data.columns = ['EF_ratio', 'Sr', 'Soil_ratio', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P', 'Ig.loss']
composition_data = ['Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P', 'Ig.loss']
attribute = ['EF_ratio', 'Sr', 'Soil_ratio', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P']

# Drop missing values
forest = data.dropna(axis=0)

# Transform Sr ratio (Box-Cox for EF, Soil; log for Sr)
Sr_data = forest[['EF_ratio', 'Sr', 'Soil_ratio']]
EF_fit = stats.boxcox(Sr_data['EF_ratio'])[0]
Soil_fit = stats.boxcox(Sr_data['Soil_ratio'])[0]
Sr_data['Sr'] = pd.to_numeric(Sr_data['Sr'], errors='coerce')
Sr_log = np.log1p(Sr_data['Sr'])

# Log-ratio transform for compositional data
main_element = forest[composition_data].to_numpy()
geo_means = np.exp(np.mean(np.log(main_element), axis=1))
log_ratios = np.log(main_element / geo_means[:, np.newaxis])
M = pd.DataFrame(log_ratios, columns=composition_data).drop(['Ig.loss'], axis=1)

# Combine features
X_transform = pd.concat([
    pd.DataFrame(EF_fit, columns=['EF_ratio']),
    pd.DataFrame(Sr_log, columns=['Sr']),
    pd.DataFrame(Soil_fit, columns=['Soil_ratio']),
    M.reset_index(drop=True)
], axis=1).dropna()

X_train, X_test = train_test_split(X_transform, test_size=0.2, random_state=42)

# Normalize
scaler_std = StandardScaler().fit(X_train)
X_train = scaler_std.transform(X_train)
scaler_minmax = MinMaxScaler((0, 1)).fit(X_train)
X_train = scaler_minmax.transform(X_train)
X_train = pd.DataFrame(X_train, columns=attribute)

# Result collector
result_table = []

# Clustering & evaluation function
def clustering_result_from_data(X_data, method_label):
    k_range = range(3, 10)
    silhouette_scores = []
    distortions = []
    db_scores = []

    for k in k_range:
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(X_data)

        centroids = np.array([X_data[labels == i].mean(axis=0) for i in range(k)])
        distortions.append(np.sum(cdist(X_data, centroids, 'euclidean').min(axis=1)) / len(X_data))

        silhouette_scores.append(silhouette_score(X_data, labels))
        db_scores.append(davies_bouldin_score(X_data, labels))

    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    best_silhouette = max(silhouette_scores)
    db_at_best_k = db_scores[silhouette_scores.index(max(silhouette_scores))]

    result_table.append({
        'Method': method_label,
        'Best_k': best_k,
        'Silhouette_Score': round(best_silhouette, 4),
        'Davies_Bouldin_Index': round(db_at_best_k, 4)
    })

# 1. Original (13D)
clustering_result_from_data(X_train, method_label='Original')

# 2. PCA (3, 4, 5D)
for dim in [3, 4, 5]:
    X_pca = PCA(n_components=dim).fit_transform(X_train)
    clustering_result_from_data(X_pca, method_label=f'PCA_{dim}')

# 3. Autoencoder (2D, 3D, 4D)
ae_models = {
    'AE_2': 'autoencoder_model_2D_2layer.h5',
    'AE_3': 'autoencoder_model_3D_2layer.h5',
    'AE_4': 'autoencoder_model_4D_2layer.h5',
}

for label, path in ae_models.items():
    ae_model = load_model(path, compile=False)
    encoder = Model(inputs=ae_model.input, outputs=ae_model.get_layer('encoder_layer').output)
    X_encoded = encoder.predict(X_train)
    clustering_result_from_data(X_encoded, method_label=label)

# Show result summary
results_df = pd.DataFrame(result_table)
print(results_df)

# Store k-wise scores for plotting
silhouette_by_k = {'Original': [], 'PCA_3': [], 'AE_2': []}
dbi_by_k = {'Original': [], 'PCA_3': [], 'AE_2': []}

def collect_scores_across_k(X_data, method_label):
    ks = list(range(3, 11))
    sils, dbs = [], []

    for k in ks:
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(X_data)

        sil_score = silhouette_score(X_data, labels)
        dbi_score = davies_bouldin_score(X_data, labels)

        sils.append(sil_score)
        dbs.append(dbi_score)

    silhouette_by_k[method_label] = sils
    dbi_by_k[method_label] = dbs

# Collect scores for each method
collect_scores_across_k(X_train, 'Original')

X_pca3 = PCA(n_components=3).fit_transform(X_train)
collect_scores_across_k(X_pca3, 'PCA_3')  # changed to PCA_3

ae_model = load_model('autoencoder_model_2D_2layer.h5', compile=False)
encoder = Model(inputs=ae_model.input, outputs=ae_model.get_layer('encoder_layer').output)
X_ae2 = encoder.predict(X_train)
collect_scores_across_k(X_ae2, 'AE_2')

# Plotting like Figure 4
def plot_score_across_k(score_dict, dbi_dict):
    ks = list(range(3, 11))

    plt.figure(figsize=(8, 10))

    # Silhouette Score
    plt.subplot(2, 1, 1)
    for label, scores in score_dict.items():
        if not scores:
            continue
        style = {
            'Original': ':',
            'PCA_3': '--',
            'AE_2': '-'
        }.get(label, '-')
        plt.plot(ks, scores, style, label=f'Case {list(score_dict.keys()).index(label)+1} ({label})')
    plt.title('(a) Silhouette score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.legend()

    # Davies-Bouldin Index
    plt.subplot(2, 1, 2)
    for label, scores in dbi_dict.items():
        if not scores:
            continue
        style = {
            'Original': ':',
            'PCA_3': '--',
            'AE_2': '-'
        }.get(label, '-')
        plt.plot(ks, scores, style, label=f'Case {list(score_dict.keys()).index(label)+1} ({label})')
    plt.title('(b) Davies-Bouldin score')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Davies-Bouldin score')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run final plot
plot_score_across_k(silhouette_by_k, dbi_by_k)