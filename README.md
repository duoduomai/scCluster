# scCluster
![framework](https://github.com/duoduomai/scCluster/assets/77309033/2cd2c2a0-ac54-496d-97c9-fdb58cfee15f)
# Overview
The emergence of single-cell RNA sequencing (scRNA-seq) technology has enabled researchers to study cellular heterogeneity at the single-cell level, in which identifying subpopulations of cells with different functions is an important research problem in scRNA-seq data analysis.
Existing cell subpopulation identification methods mainly rely on gene expression features, while ignoring the genomic information enriched in the raw sequencing data.
This work proposes an end-to-end deep learning model, named scCluster, for stratifying cell subpopulations of cancer tissues based on integrating the gene expression features and the expressed single nucleotide polymorphism (eSNP) features both derived from the raw scRNA-seq.
scCluster utilizes a two-stage training strategy to fuse the dual modalities and consequently improve the cell subpopulation identification. Experimental results reveal its proficiency in accurately identifying cell subpopulations in multiple cancer tissues.

# Software dependencies

scanpy==1.9.3
torch==2.0.1
numpy==1.24.3 

#
