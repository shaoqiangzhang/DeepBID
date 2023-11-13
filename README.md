# DeepBID
Deep Batch Integration and Denoise of Single-Cell RNA-seq Data

## Dependency

Programs implemented by pytorch reqiure: 

- python 3.6
- pytorch 1.7.1
- numpy
- scikit-learn 
- scipy
- munkres
- pandas


## Brief Introduction

- DeepBID.py  the main code for single-cell RNA-seq data
- utils.py functions used in experiemnts.
- metrics.py codes for evaluation of clustering results. 
- layers.py function of NB and ZINB

Samples to run the code is given as follows

```python

if __name__ == '__main__':
    data, labels ,csv_batch = BID_data, BID_labels,BID_batches
    data = data.T
    for lam1 in [0.01,0.05,0.1,0.5,1,5,10]:
        print('lam1={}'.format(lam1))
        bid = DeepBID(data, labels, BID_batches,[data.shape[0], 1000, 500,200], lam1 = lam1,lam2=0.001,gamma=1,sigma=1,kl1=0.1, kl2=0.1, nb=1,  batch_size=128, lr=10**-5)
        bid.run()

```


## Citations




## Thanks

Thanks to 

- Rui Zhang, Xuelong Li, Hongyuan Zhang, and Feiping Nie, "Deep Fuzzy K-Means with Adaptive Loss and Entropy Regularization," *IEEE Transactions on Fuzzy Systems*, DOI:10.1109/TFUZZ.2019.2945232.

- Liang Chen,Weinan Wang,Yuyao Zhai & Minghua Deng.(2020).Deep soft K-means clustering with self-training for single-cell RNA sequence data. NAR Genomics and Bioinformatics(2). doi:10.1093/nargab/lqaa039.

- Tian Tian,Ji Wan,Qi Song & Zhi Wei.(2019).Clustering single-cell RNA-seq data with a model-based deep learning approach. Nature Machine Intelligence(4). doi:10.1038/s42256-019-0037-0.

Part of their codes were used in our project. 
