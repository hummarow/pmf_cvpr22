import numpy as np

from sklearn.manifold import TSNE
from os import listdir

param = []

files = [f for f in listdir('./params')]

for file in files:
    with open(file, 'r') as fp:
        params.append(fp.read())

tsne = TSNE(perplexity=5)
tsne.fit_transform(np.array(params))
