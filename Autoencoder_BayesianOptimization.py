import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bayes_opt import BayesianOptimization

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

# Remove rows with missing values
forest = data.dropna(axis=0)

# Box-Cox transformation for Sr ratio data
Sr_data = forest[['EF_ratio', 'Sr', 'Soil_ratio']]
EF_fit = stats.boxcox(Sr_data['EF_ratio'])[0]
Soil_fit = stats.boxcox(Sr_data['Soil_ratio'])[0]

# Log transformation for Sr
Sr_data['Sr'] = pd.to_numeric(Sr_data['Sr'], errors='coerce')
Sr_log = np.log1p(Sr_data['Sr'])

# Compositional Log-ratio transformation for compositional data
main_element = forest[composition_data].to_numpy()
geo_means = np.exp(np.mean(np.log(main_element), axis=1))
log_ratios = np.log(main_element / geo_means[:, np.newaxis])
M = pd.DataFrame(log_ratios, columns=composition_data).drop(['Ig.loss'], axis=1)

# Combine all features
X_transform = pd.concat([
    pd.DataFrame(EF_fit, columns=['EF_ratio']),
    pd.DataFrame(Sr_log, columns=['Sr']),
    pd.DataFrame(Soil_fit, columns=['Soil_ratio']),
    M.reset_index(drop=True)
], axis=1).dropna()

print(X_transform.isnull().sum())

# Train/test split (keep original order for location)
X_train, X_test = train_test_split(X_transform, test_size=0.2, random_state=42)

# Standardization and Normalization
scaler_std = StandardScaler().fit(X_train)
X_train = scaler_std.transform(X_train)
X_test = scaler_std.transform(X_test)

scaler_minmax = MinMaxScaler((0, 1)).fit(X_train)
X_train = scaler_minmax.transform(X_train)
X_test = scaler_minmax.transform(X_test)

# Fixing the Input Dimension
num_inputs = X_train.shape[1]

# Autoencoder Definition Function
def autoencoder(nodes_1, nodes_2, learning_rate, num_middle):
    A = Input(shape=(num_inputs,))

    # Encoder
    encoded = Dense(int(nodes_1))(A)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    encoded = Dense(int(nodes_2))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU()(encoded)

    # Bottleneck
    bottleneck = Dense(int(num_middle), activation='linear', name='encoder_layer')(encoded)

    # Decoder
    decoded = Dense(int(nodes_2))(bottleneck)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    decoded = Dense(int(nodes_1))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU()(decoded)

    # Output
    B = Dense(num_inputs, activation='linear')(decoded)

    model = Model(inputs=A, outputs=B)
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse')

    return model

# Training and MSE Evaluation Function
def train_autoencoder_return_mse(nodes_1, nodes_2, learning_rate, batch_size, epochs, num_middle):
    # 정수 변환
    nodes_1 = int(nodes_1)
    nodes_2 = int(nodes_2)
    batch_size = int(batch_size)
    epochs = int(epochs)

    # Return a large loss value when constraints are violated
    if not (num_middle < nodes_2 < nodes_1 < 13):
        return -1e6  # Abnormal Value (to Avoid Being Minimized)

    # Model Training and MSE Calculation
    model = autoencoder(nodes_1, nodes_2, learning_rate, num_middle)
    model.fit(X_train, X_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=0,
              shuffle=False,
              validation_data=(X_test, X_test))
    pred = model.predict(X_test, verbose=0)
    mse = mean_squared_error(X_test, pred)
    return -mse

# Bayesian Optimization Function
def optimize_autoencoder(bottleneck_dim):
    def bo_function(nodes_1, nodes_2, learning_rate, batch_size, epochs):
        return train_autoencoder_return_mse(nodes_1, nodes_2, learning_rate, batch_size, epochs, bottleneck_dim)

    # Define a Valid Range According to the Bottleneck Size
    pbounds = {
        'nodes_1': (bottleneck_dim + 2, 12),  # max 12 < 13
        'nodes_2': (bottleneck_dim + 1, 11),
        'learning_rate': (0.0001, 0.1),
        'batch_size': (16, 128),
        'epochs': (100, 2000),
    }

    optimizer = BayesianOptimization(f=bo_function, pbounds=pbounds, random_state=42)
    optimizer.maximize(init_points=5, n_iter=20)

    print(f"\n Bottleneck = {bottleneck_dim} 최적 결과")
    print(optimizer.max)
    return optimizer

# Run: Hyperparameter Optimization for Bottleneck Sizes Ranging from 2 to 5
optimizers = {}
for dim in [2, 3, 4, 5]:
    print(f"\n==== Bottleneck Dimension: {dim} ====")
    optimizers[dim] = optimize_autoencoder(dim)

# Print Summary of Results
for dim in optimizers:
    print(f"\n Bottleneck {dim} ➜ Best MSE: {-optimizers[dim].max['target']:.5f}")
    print(f" Hyperparameters: {optimizers[dim].max['params']}")