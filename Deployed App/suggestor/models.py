import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors



def build_model(input):
    """
    Nearest Neighbors Model for song suggestions
    """
    df = pd.read_csv('suggestor/clean_US.csv')
    # configure song index Nearest Neighbor result output
    mask = ['combined', 'url']
    df_result = df[mask]
    # set up data DataFrame for numerical features only
    data = df.select_dtypes('number')
    def normalize_column(col):
        """
        function to normalize data in numerical features
        """
        max_d = data[col].max()
        min_d = data[col].min()
        data[col] = (data[col] - min_d)/(max_d - min_d)
    num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = data.select_dtypes(include=num_types)
    for col in num.columns:
        normalize_column(col)
    # nearest neighbors configuration
    knn = NearestNeighbors(algorithm='ball_tree',
                           radius=1.75, leaf_size =20, metric='euclidean', n_neighbors=12, n_jobs=-1)
    knn.fit(data)
    # query point to sample from
    input_index = input
    # vectorize data for processing and cast to a list
    data_vector = [data.iloc[input_index].values]
    nn_distance, nn_indices = knn.kneighbors(data_vector)
    indices = nn_indices.flat[0:7].tolist()
    # generate output
    output = df_result.iloc[indices]
    return (output)


def to_list(df):
    """
    Create list of track & artists
    """
    combined = df['combined']
    track_artist = combined.tolist()
    return track_artist
