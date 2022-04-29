import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# random seed is N number
random.seed(18402254)
# Load data
data = pd.read_csv('musicData.csv')

#missing data
missing = data.isnull().sum()
print(missing)

# imputing missing numeric data
data['instance_id'] = data['instance_id'].fillna(data['instance_id'].mean())
data['popularity'] = data['popularity'].fillna(data['popularity'].mean())
data['acousticness'] = data['acousticness'].fillna(data['acousticness'].mean())
data['danceability'] = data['danceability'].fillna(data['danceability'].mean())
data['duration_ms'] = data['duration_ms'].fillna(data['duration_ms'].mean())
data['energy'] = data['energy'].fillna(data['energy'].mean())
data['instrumentalness'] = data['instrumentalness'].fillna(data['instrumentalness'].mean())
data['liveness'] = data['liveness'].fillna(data['liveness'].mean())
data['loudness'] = data['loudness'].fillna(data['loudness'].mean())
data['speechiness'] = data['speechiness'].fillna(data['speechiness'].mean())
data['valence'] = data['valence'].fillna(data['valence'].mean())

# delete missing data
data = data.dropna()
missing = data.isnull().sum()

# replace categorical data with numerical data
key = data['key'].unique()
data['key'] = data['key'].replace(key, range(len(key)))
data['mode'] = data['mode'].replace(['minor', 'major'], [0, 1])
music_genere = data['music_genre'].unique()
data['music_genre'] = data['music_genre'].replace(music_genere, range(len(music_genere)))

# drop unnecessary columns
preprocessed_data = data.drop('artist_name', axis=1)
preprocessed_data = preprocessed_data.drop('track_name', axis=1)
preprocessed_data = preprocessed_data.drop('obtained_date', axis=1)
print(preprocessed_data.head())

# classifications
X = []
Y = []
for i in range(len(music_genere)):
    temp = preprocessed_data[preprocessed_data['music_genre'] == i]
    X.append(temp.drop('music_genre', axis=1))
    Y.append(temp['music_genre'])

# model by genre
