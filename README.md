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

pandas==2.0.1

scikit-learn==1.2.2

keras==2.9.0

scipy==1.10.1

# Usage

```
python ./code/run_scCluster.py  # set "data_name" to "Patel" in run_scCluster.py
```
Set data_name in run_scCluster.py as the target of the data before running. 

Running requires three files: the gene expression file, eSNP file, and label file (only used for calculating evaluation metrics) for the dataset. 

The final output reports the clustering performance, that is, the clustering indicator results between the predicted results and the real labels, and can output the trained cell embedding representation as needed. 

## Input-demo

Patel.txt -----gGene expression data from the Patel dataset.

Patel_snp.txt ----- eSNP data from the Patel dataset.

Patel_truelabels.csv ----- truelabels from the Patel dataset.

## Output-demo

ACC = 0.9853

NMI = 0.9583

AMI = 0.9575

ARI = 0.9652

Save the cell embeddings as Patel.npy (optionaly).


