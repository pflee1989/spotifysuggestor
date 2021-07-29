import pandas as pd
import numpy as np
import glob
import os
from os import listdir
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import tensorflow as tf
import sklearn 
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model, Sequential, save_model, load_model
from tensorflow.config import list_logical_devices
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


def build_model(input):
    """
    Autoencoder with Nearest Neighbors Model for song suggestions
    """
    df = pd.read_csv('suggestor/clean_norm_reduced_US.csv')
    data = df.select_dtypes('number')
    data = data.to_numpy()

    # load LeakyRelu model configuration
    model = keras.models.load_model("finalized_spotify_model_leaky")

    # encoded data for KNN model
    encoded_data_leaky = model.encoder(data)

    # nearest neighbors
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree',
                           radius=1, n_jobs=-1)
    knn.fit(encoded_data_leaky)

    # query the data for knn
    input_index = input

    _, ind_leaky = knn.kneighbors([encoded_data_leaky[input_index]])

    # configure nn_song_index Nearest Neighbor result output
    mask = ['combined', 'url']
    df_result = df[mask]
    nn_song_index = ind_leaky.flat[:].tolist()
    df_result.iloc[nn_song_index]

    # creating output with url for each track
    output = df_result.iloc[nn_song_index]
    return (output)


def to_list(df):
    """
    Create list of track & artists
    """
    combined = df['combined']
    track_artist = combined.tolist()
    return track_artist
