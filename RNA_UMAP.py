# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:52:20 2023

@author: Tony Kelly
used Allen brain cortical dataset contributed by berenslab (https://github.com/berenslab/rna-seq-tsne/blob/master/demo.ipynb)
and used umap see umap_numbers (from https://umap-learn.readthedocs.io/en/latest/basic_usage.html)

"""

# Prepare

# %matplotlib notebook

import numpy as np
import pylab as plt
import seaborn as sns; sns.set()
import pandas as pd

import scipy
import matplotlib
from scipy import sparse
import sklearn
from sklearn.decomposition import PCA


# this tsne didnt work for me
# # import tsne
# import sys; sys.path.append('/home/localadmin/github/FIt-SNE')
# from fast_tsne import fast_tsne

# print('Python version', sys.version)
print('Numpy', np.__version__)
print('Matplotlib', matplotlib.__version__)
print('Seaborn', sns.__version__)
print('Pandas', pd.__version__)
print('Scipy', scipy.__version__)
print('Sklearn', sklearn.__version__)

# %% Load the Allen institute data. This takes a few minutes
# %time

# Using Pandas to load these files in one go eats up a lot of RAM. 
# So we are doing it in chunks
# converting each chunk to the sparse matrix format on the fly.


def sparseload(filenames):
    genes = []
    sparseblocks = []
    areas = []
    cells = []
    for chunk1,chunk2 in zip(pd.read_csv(filenames[0], chunksize=1000, index_col=0, na_filter=False),
                             pd.read_csv(filenames[1], chunksize=1000, index_col=0, na_filter=False)):
        if len(cells)==0:
            cells = np.concatenate((chunk1.columns, chunk2.columns))
            areas = [0]*chunk1.columns.size + [1]*chunk2.columns.size
        
        genes.extend(list(chunk1.index))
        sparseblock1 = sparse.csr_matrix(chunk1.values.astype(float))
        sparseblock2 = sparse.csr_matrix(chunk2.values.astype(float))
        sparseblock = sparse.hstack((sparseblock1,sparseblock2), format='csr')
        sparseblocks.append([sparseblock])
        print('.', end='', flush=True)
    print(' done')
    counts = sparse.bmat(sparseblocks)
    return (counts.T, np.array(genes), cells, np.array(areas))

#exon matrix is rows: gene id (see gene csv) and columns: samples
filenames = ['D:/RNAseq_datasets/berenslab/data/tasic-nature/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_exon-matrix.csv','D:/RNAseq_datasets/berenslab/data/tasic-nature/mouse_ALM_gene_expression_matrices_2018-06-14/mouse_ALM_2018-06-14_exon-matrix.csv']

counts, genes, cells, areas = sparseload(filenames)

genesDF = pd.read_csv('D:/RNAseq_datasets/berenslab/data/tasic-nature/mouse_VISp_gene_expression_matrices_2018-06-14/mouse_VISp_2018-06-14_genes-rows.csv')
ids     = genesDF['gene_entrez_id'].tolist()
symbols = genesDF['gene_symbol'].tolist()
id2symbol = dict(zip(ids, symbols))
genes = np.array([id2symbol[g] for g in genes])

clusterInfo = pd.read_csv('D:/RNAseq_datasets/berenslab/data/tasic-nature/sample_heatmap_plot_data.csv')
goodCells  = clusterInfo['sample_name'].values
ids        = clusterInfo['cluster_id'].values
labels     = clusterInfo['cluster_label'].values
colors     = clusterInfo['cluster_color'].values

clusterNames  = np.array([labels[ids==i+1][0] for i in range(np.max(ids))])
clusterColors = np.array([colors[ids==i+1][0] for i in range(np.max(ids))])
clusters   = np.copy(ids) - 1

ind = np.array([np.where(cells==c)[0][0] for c in goodCells])
counts = counts[ind, :]

tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas, 
             'clusterColors': clusterColors, 'clusterNames': clusterNames}
counts = []

print('Number of cells:', tasic2018['counts'].shape[0])
print('Number of cells from ALM:', np.sum(tasic2018['areas']==0))
print('Number of cells from VISp:', np.sum(tasic2018['areas']==1))
print('Number of clusters:', np.unique(tasic2018['clusters']).size)
print('Number of genes:', tasic2018['counts'].shape[1])
print('Fraction of zeros in the data matrix: {:.2f}'.format(
    tasic2018['counts'].size/np.prod(tasic2018['counts'].shape)))


# %%Feature selection
# %time

def nearZeroRate(data, threshold=0):
    zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
    return zeroRate

def meanLogExpression(data, threshold=0, atleast=10):
    nonZeros = np.squeeze(np.array((data>threshold).sum(axis=0)))
    N = data.shape[0]
    A = data.multiply(data>threshold)
    A.data = np.log2(A.data)
    meanExpr = np.zeros(data.shape[1]) * np.nan
    detected = nonZeros >= atleast
    meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (nonZeros[detected]/N)
    return meanExpr
    
def featureSelection(meanLogExpression, nearZeroRate, yoffset=.02, decay=1.5, n=3000):
    low = 0; up=10    
    nonan = ~np.isnan(meanLogExpression)
    xoffset = 5
    for step in range(100):
        selected = np.zeros_like(nearZeroRate).astype(bool)
        selected[nonan] = nearZeroRate[nonan] > np.exp(-decay*meanLogExpression[nonan] + xoffset) + yoffset
        if np.sum(selected) == n:
            break
        elif np.sum(selected) < n:
            up = xoffset
            xoffset = (xoffset + low)/2
        else:
            low = xoffset
            xoffset = (xoffset + up)/2
    return selected

x = meanLogExpression(tasic2018['counts'], threshold=32)  # Get mean log non-zero expression of each gene
y = nearZeroRate(tasic2018['counts'], threshold=32)       # Get near-zero frequency of each gene
selectedGenes = featureSelection(x, y, n=3000)            # Adjust the threshold to select 3000 genes

plt.figure(figsize=(6,3))
plt.scatter(x[~selectedGenes], y[~selectedGenes], s=1)
plt.scatter(x[selectedGenes],  y[selectedGenes], s=1, color='r')
plt.xlabel('Mean log2 nonzero expression')
plt.ylabel('Frequency of\nnear-zero expression')
plt.ylim([0,1])
plt.tight_layout()

# %% Create logCPM and pca of logCPM from selected genes
# %time

counts3k = tasic2018['counts'][:, selectedGenes]  # Feature selection

librarySizes = tasic2018['counts'].sum(axis=1)    # Compute library sizes
CPM = counts3k / librarySizes * 1e+6              # Library size normalisation

logCPM = np.log2(CPM + 1)                         # Log-transformation

logCPM=np.asarray(logCPM)                       #np array 

pca = PCA(n_components=50, svd_solver='full').fit(logCPM)   # PCA

flipSigns = np.sum(pca.components_, axis=1) < 0             # fix PC signs
X = pca.transform(logCPM)
X[:, flipSigns] *= -1

print('Shape of the resulting matrix:', X.shape, '\n')

# Principal component analysis

plt.figure(figsize=(4,4))
plt.scatter(X[:,0], X[:,1], s=1, color=tasic2018['clusterColors'][tasic2018['clusters']])
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()

#NOW HAVE X (pca transform of log CPM) and logCPM 
#tSNE on pca. but use from sklearn.manifold import TSNE.
# orig fast tSNE not available 
# import sys; sys.path.append('/home/localadmin/github/FIt-SNE')
# from fast_tsne import fast_tsne

# %% Run UMAP on RNAseq data and plot
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

reducer = umap.UMAP(random_state=42)
#run on logCPM data
reducer.fit(logCPM)
embedding = reducer.transform(logCPM)
assert(np.all(embedding == reducer.embedding_))
embedding.shape

plt.figure(figsize=(4,4))
plt.scatter(embedding[:, 0], embedding[:, 1], s=1, color=tasic2018['clusterColors'][tasic2018['clusters']], cmap='Spectral')
plt.title('UMAP')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.tight_layout()

# %% select cells (of the 2382 cell) that have gene of interest (total 45768 genes)

GOI = 'Htr2c'
indices = [i for i, s in enumerate(genes) if GOI in s]
#now fing indices in magenes
mapGenes=np.where(selectedGenes)[0]

GOI_cells=logCPM[:,np.where(mapGenes==indices)[0][0]]

#colour scale based on cell expression
normalized_arr = preprocessing.normalize([GOI_cells])
scaler = preprocessing.MinMaxScaler()
GOI_colour_scale = scaler.fit_transform(GOI_cells.reshape(-1, 1))
 
plt.figure(figsize=(4,4))
plt.scatter(X[GOI_cells>0,0], X[GOI_cells>0,1], s=1, c='r', alpha=GOI_colour_scale) #Htr2c_colour_scale[Htr2c>0]
plt.scatter(X[GOI_cells==0,0], X[GOI_cells==0,1], s=1, c='k', alpha=0.01) #Htr2c_colour_scale[Htr2c>0]

plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()

plt.figure(figsize=(8,4))

plt.subplot(121)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.scatter(embedding[:, 0], embedding[:, 1], s=1, color=tasic2018['clusterColors'][tasic2018['clusters']])
plt.title('Full dataset UMAP')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.subplot(122)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.scatter(embedding[GOI_cells>0,0], embedding[GOI_cells>0,1], s=1, c=tasic2018['clusterColors'][tasic2018['clusters']][GOI_cells>0], alpha=GOI_colour_scale[GOI_cells>0]) #Htr2c_colour_scale[Htr2c>0]
plt.scatter(embedding[GOI_cells==0,0], embedding[GOI_cells==0,1], s=1, c='k', alpha=0.01) #Htr2c_colour_scale[Htr2c>0]

plt.title('mapped Htr2c cells')
plt.gca().get_xaxis().set_ticklabels([])
plt.gca().get_yaxis().set_ticklabels([])

plt.tight_layout()

#list of cell clustes positive for GOI
GOI_CellClusters=clusterInfo[GOI_cells>0].drop_duplicates(subset=['cluster_id'])