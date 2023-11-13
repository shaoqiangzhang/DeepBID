import numpy as np
import pandas as pd
import scanpy as sc

adata1=pd.read_csv("pancreas_multi_celseq_expression_matrix.txt",header=0, index_col=0, delim_whitespace=True)
adata2=pd.read_csv("pancreas_multi_celseq2_expression_matrix.txt",header=0, index_col=0, delim_whitespace=True)
adata3=pd.read_csv("pancreas_multi_fluidigmc1_expression_matrix.txt",header=0, index_col=0, delim_whitespace=True)
adata4=pd.read_csv("pancreas_multi_smartseq2_expression_matrix.txt",header=0, index_col=0, delim_whitespace=True)
adata5=pd.read_csv("pancreas_human.expressionMatrix.txt",header=0, index_col=0, delim_whitespace=True)

adata1 = adata1.add_prefix("celseq_")
adata2 = adata2.add_prefix("celseq2_")
adata3 = adata3.add_prefix("c1_")
adata4 = adata4.add_prefix("smartseq_")
adata5 = adata5.add_prefix("tenx_")

adata1=adata1.T # transpose adata if it is gene*cellmore
data1=sc.AnnData(adata1, dtype=np.float32)
#sc.pp.filter_genes(data1, min_cells=3)
sc.pp.normalize_total(data1, target_sum=1e4) 
sc.pp.log1p(data1) 
sc.pp.highly_variable_genes(data1, n_top_genes=1000)

adata2=adata2.T # transpose adata if it is gene*cell
data2=sc.AnnData(adata2, dtype=np.float32)
sc.pp.normalize_total(data2, target_sum=1e4) 
sc.pp.log1p(data2) 
sc.pp.highly_variable_genes(data2, n_top_genes=1000)

adata3=adata3.T # transpose adata if it is gene*cell
data3=sc.AnnData(adata3, dtype=np.float32)
sc.pp.normalize_total(data3, target_sum=1e4) 
sc.pp.log1p(data3) 
sc.pp.highly_variable_genes(data3, n_top_genes=1000)

adata4=adata4.T # transpose adata if it is gene*cell
data4=sc.AnnData(adata4, dtype=np.float32)
sc.pp.normalize_total(data4, target_sum=1e4) 
sc.pp.log1p(data4) 
sc.pp.highly_variable_genes(data4, n_top_genes=1000)

adata5=adata5.T # transpose adata if it is gene*cell
data5=sc.AnnData(adata5, dtype=np.float32)
sc.pp.normalize_total(data5, target_sum=1e4) 
sc.pp.log1p(data5) 
sc.pp.highly_variable_genes(data5, n_top_genes=1000)

highvar = data1.var.highly_variable | data2.var.highly_variable | data3.var.highly_variable | data4.var.highly_variable | data5.var.highly_variable 

data_all = sc.AnnData.concatenate(data1,data2,data3,data4,data5, join ='outer')

data_all = data_all[:, highvar] 

alldata=data_all.to_df().fillna(0.0)

alldata.to_csv("pancreas5data.csv")





