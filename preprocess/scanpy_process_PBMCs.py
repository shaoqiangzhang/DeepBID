import numpy as np
import pandas as pd
import scanpy as sc

adata1 = sc.read_10x_mtx( "3pV1/filtered_matrices_mex/hg19", var_names='gene_symbols', cache=True)
adata2 = sc.read_10x_mtx( "3pV2/filtered_gene_bc_matrices/GRCh38", var_names='gene_symbols', cache=True)
adata5 = sc.read_10x_mtx( "5p/filtered_gene_bc_matrices/GRCh38", var_names='gene_symbols', cache=True)

data1=sc.AnnData(adata1, dtype=np.float32)
sc.pp.filter_genes(data1, min_cells=3)

sc.pp.normalize_total(data1, target_sum=1e4) 
sc.pp.log1p(data1) 
sc.pp.highly_variable_genes(data1, n_top_genes=1000)

data2=sc.AnnData(adata2, dtype=np.float32)
sc.pp.filter_genes(data2, min_cells=3)

sc.pp.normalize_total(data2, target_sum=1e4) 
sc.pp.log1p(data2) 
sc.pp.highly_variable_genes(data2, n_top_genes=1000)

data5=sc.AnnData(adata5, dtype=np.float32)
sc.pp.filter_genes(data5, min_cells=3)

sc.pp.normalize_total(data5, target_sum=1e4) 
sc.pp.log1p(data5) 
sc.pp.highly_variable_genes(data5, n_top_genes=1000)

highvar = data1.var.highly_variable | data2.var.highly_variable | data5.var.highly_variable

data_all = sc.AnnData.concatenate(data1,data2,data5, join ='outer')

data_all = data_all[:, highvar] 

alldata=data_all.to_df().fillna(0.0)

alldata.to_csv("10xPBMCs3data.csv")

