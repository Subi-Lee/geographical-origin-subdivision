import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import randint
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Dropout
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.utils import to_categorical

seed_number = 42

np.random.seed(seed_number)
random.seed(seed_number)
tf.random.set_seed(seed_number)

print(os.getcwd())
os.chdir('./data')

# Load and preprocess data
data = pd.read_csv('input_data_label.csv', encoding='cp949')
data.columns = ['EF_ratio', 'Sr', 'Soil_ratio', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P', 'Ig.loss', 'label']
attribute = ['EF_ratio', 'Sr', 'Soil_ratio', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P']
composition_data = ['Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P', 'Ig.loss']

# 결측치 행 전체 제거
forest = data.dropna(axis=0).to_numpy()
forest = pd.DataFrame(forest)
forest.columns = ['EF_ratio', 'Sr', 'Soil_ratio', 'Si', 'Al', 'Fe', 'Ca', 'Mg', 'K', 'Na', 'Ti', 'Mn', 'P', 'Ig.loss', 'label']
print(len(forest))

# target index
target = forest['label'].astype(int)
X = forest.drop(['label'], axis=1)
Y = target

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

# Normalize
scaler_std = StandardScaler().fit(X_transform)
X_scaled = scaler_std.transform(X_transform)
scaler_minmax = MinMaxScaler((0, 1)).fit(X_scaled)
X_scaled = scaler_minmax.transform(X_scaled)
X_scaled = pd.DataFrame(X_scaled, columns=attribute)

# label 별로 10개씩 제외
X_label = pd.DataFrame(X_scaled)
X_label['label'] = Y

def drop_random_10(group, random_seed=42):
    to_drop = group.sample(n=10, random_state=random_seed)
    return group.drop(to_drop.index), to_drop

# 순서대로 10개씩 제외한 데이터, 각 라벨별 10개씩 데이터
reduced_groups = []
dropped_samples = []

for _, group in X_label.groupby('label'):
    reduced_group, dropped = drop_random_10(group)
    reduced_groups.append(reduced_group)
    dropped_samples.append(dropped)

# 훈련을 위한 각 라벨 별 10개씩 제외한 데이터
reduced_label = pd.concat(reduced_groups)
X_train = reduced_label.drop(['label'], axis=1)
Y_train = reduced_label['label']

# 검증을 위한 각 라벨 별 10개 데이터
selected_label = pd.concat(dropped_samples)
X_validation = selected_label.drop(['label'], axis=1)
Y_validation = selected_label['label']

# RF 모델 적용 전 Oversampling - SMOTE
print('SMOTE Oversampling')
smote = SMOTE(random_state=seed_number)
X_SMOTE, Y_SMOTE = smote.fit_resample(X_train, Y_train)
print('Distribution of labels after SMOTE:')
print(Y_SMOTE.value_counts())

# 랜덤포레스트 모델 설정
rf = RandomForestClassifier(random_state=seed_number)

## Optimization ------------------------------------------------------
## RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(30, 500),  # Or use a list like before
    'max_depth': [None, 5, 10, 15, 20, 30],  # Or use randint for a range
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                   n_iter=1000, scoring='f1_micro', cv=5, n_jobs=-1, verbose=2, random_state=42)

# Model training
random_search.fit(X_SMOTE, Y_SMOTE)

# 최적의 파라미터와 성능 결과 확인
best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Applying the best parameters to the Random Forest model
rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_SMOTE, Y_SMOTE)

print('---------------Validation data---------------')
# Model evaluation on validation data
y_pred = rf_best.predict(X_validation)
accuracy = accuracy_score(Y_validation, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(Y_validation, y_pred))

print('[Optimized rf_best] Accuracy:', accuracy_score(Y_validation, rf_best.predict(X_validation)))

# Confusion matrix
conf_matrix = confusion_matrix(Y_validation, y_pred)
print("Confusion Matrix:")
print(conf_matrix)