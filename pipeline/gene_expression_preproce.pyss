import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler

file_path="../data/XXX.csv"
data = pd.read_csv(file_path, index_col=0, header=0)
data = pd.DataFrame(data)
scaler=StandardScaler()
data_scaler=pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
adata = sc.AnnData(data_scaler)
adata.raw = adata

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=20000)
adata = adata[:, adata.var['highly_variable']]
#sc.pp.scale(adata, max_value=3)
np.save("../XXX.txt", adata.X, delimiter=',')

print("Finished.")
