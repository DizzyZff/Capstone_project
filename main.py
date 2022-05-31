import random
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
from IPython import display
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.cluster import KMeans
# random seed is N number
random.seed(18402254)
# Load data
"""data = pd.read_csv('musicData.csv')
# replace string to number except for null
artist_name = data['artist_name'].dropna()
data['artist_name'] = artist_name.replace(artist_name.unique(), np.arange(0, len(artist_name.unique())))
song_name = data['track_name'].dropna()
data['track_name'] = song_name.replace(song_name.unique(), np.arange(0, len(song_name.unique())))
key = data['key'].dropna().unique()
data['key'] = data['key'].replace(key, np.arange(0, len(key)))
mode = data['mode'].dropna().unique()
data['mode'] = data['mode'].replace(mode, np.arange(0, len(mode)))
obtained_date = data['obtained_date'].dropna().unique()
data['obtained_date'] = data['obtained_date'].replace(obtained_date, np.arange(0, len(obtained_date)))
music_genre = data['music_genre'].dropna().unique()
data['music_genre'] = data['music_genre'].replace(music_genre, np.arange(0, len(music_genre)))

data.to_csv('musicData_processed.csv', index=False)
"""

pre_data = pd.read_csv('musicData_processed.csv')
pre_data = pre_data.dropna()

for c in pre_data.columns:
    pre_data[c] = pre_data[c].replace("?", -1)

X = pre_data.drop(['music_genre'], axis=1)
y = pre_data['music_genre']


#pca
pca = PCA(n_components=3)
pca.fit(X)
X_pca = pca.transform(X)

plt.figure(figsize=(10, 10))
sns.scatterplot(X_pca[:, 0], X_pca[:, 1],
                hue=y,
                palette=sns.color_palette("hls", len(y.unique())))
plt.show()

ax = plt.figure(figsize=(10, 10)).gca(projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
           c=y,
           cmap=plt.cm.get_cmap('jet', len(y.unique())))
plt.show()

#kmeans
kmeans = KMeans(n_clusters=len(y.unique()))
kmeans.fit(X_pca)
y_kmeans = kmeans.predict(X_pca)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 10))
sns.scatterplot(X_pca[:, 0], X_pca[:, 1],
                hue=y_kmeans,
                palette=sns.color_palette("hls", len(y.unique())))
sns.scatterplot(centroids[:, 0], centroids[:, 1],
                marker='*',
                color='black',
                s=200)

plt.show()

"""
Make sure to do the following train/test split:
For *each* genre, use 500 randomly picked songs for the 
test set and the other 4500 songs from that genre for the
training set. So the complete test set will be 5000x1 
randomly picked genres(one per song, 500 from each genre). 
Use all the other data in the training set and make sure there is no leakage. 
"""
"""X_train, X_test, y_train, y_test = [], [], [], []
for i in range(0,10):
    #randomly pick 500 songs from each genre
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_pca[y==i], y_kmeans[y==i], test_size=0.1, random_state=random.randint(0,100))
    X_train.append(X_train_temp)
    X_test.append(X_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)

X_train = np.concatenate(X_train)
X_test = np.concatenate(X_test)
y_train = np.concatenate(y_train)
y_test = np.concatenate(y_test)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)"""

#train/test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.1, random_state=random.randint(0,100))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# fnn with 10 hidden layers
class fnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(fnn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        out = self.relu(out)
        out = self.fc10(out)
        return out


model = fnn(3, 10, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1)%1000 == 0:
        print('Epoch [{}/5000], Loss: {:.4f}'.format(epoch+1, loss.item()))


# evaluate
y_pred = model(X_test)
y_pred = torch.max(y_pred, 1)[1]
print('Accuracy of the network on the test images: %d %%' % (100 * torch.sum(y_pred == y_test).item() / y_test.size(0)))

#to binary
y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_pred[:, 0])
roc_auc = roc_auc_score(y_test[:, 0], y_pred[:, 0])
print('AUC: %.2f' % roc_auc)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

