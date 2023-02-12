# CortexRNA-seqUMAP
example of exploring open RNAseq dataset

This repository uses open RNAseq dataset from Allen brain institute provided by Berenslab (see https://github.com/berenslab/rna-seq-tsne/blob/master/demo.ipynb)
and used umap (from https://umap-learn.readthedocs.io/en/latest/basic_usage.html)

Download the data from here: https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq and unpack

To get the information about cluster colors and labels (sample_heatmap_plot_data.csv), open the interactive data browser http://celltypes.brain-map.org/rnaseq/mouse/v1-alm, go to "Sample Heatmaps", click "Build Plot!" and then "Download data as CSV". Or the heatmap file is also provided in the heatmap folder.

Cells expressing a gene of interest are mapped onto the exisiting cell clusters. The Htr2c gene expressed in a sub-population of ventral CA3 hippocampal cells is much more widely expressed in the adult cortical dataset. 
