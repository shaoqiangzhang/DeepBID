from multiprocessing import reduction
import torch
from os.path import join
import numpy as np
import csv
import utils
import pandas as pd
from metrics import cal_clustering_metric
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import scipy.io as scio
import random
import warnings
import torch.nn.functional as F
from torch.nn import Parameter
from time import time
import torch.nn as nn
from layers import NBLoss, MeanAct, DispAct
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings('ignore')


raw_data = pd.read_csv("dataset/DC_pre.csv")
BID_data = np.array(raw_data)
raw_label = pd.read_csv("dataset/DC_celltype.csv")
BID_label = np.array(raw_label)
BID_labels = [int(x) for item in BID_label for x in item] #
raw_batch = pd.read_csv("dataset/DC_batch.csv")
BID_batch = np.array(raw_batch)
BID_batches = [int(x) for item in BID_batch for x in item] #



class Dataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[:, idx], idx
    
    def __len__(self):
        return self.X.shape[1]

class PretrainDoubleLayer(torch.nn.Module):
    def __init__(self, X, dim, device, act, batch_size=128, lr=10**-5,layers=None):
        super(PretrainDoubleLayer, self).__init__()
        self.X = X
        self.dim = dim
        self.lr = lr
        self.device = device
        self.enc = torch.nn.Linear(X.shape[0], self.dim)
        self.dec = torch.nn.Linear(self.dim, X.shape[0])
        self.batch_size = batch_size
        self.act = act

    def forward(self, x):
        if self.act is not None:
            z = self.act(self.enc(x))
            return z, self.act(self.dec(z))
        else:
            z = self.enc(x)
            return z, self.dec(z)

    def _build_loss(self, x, recons_x):
        size = x.shape[0]
        return torch.norm(x-recons_x, p='fro')**2 / size 
        
    def run(self):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        loss = 0
        for epoch in range(10):
            for i, batch in enumerate(train_loader):
                x, _ = batch
                optimizer.zero_grad()
                _, recons_x = self(x)
                loss = self._build_loss(x, recons_x)
                loss.backward()
                optimizer.step()
            print('epoch-{}: loss={}'.format(epoch, loss.item()))
        Z, _ = self(self.X.t())
        return Z.t()

class DeepBID(torch.nn.Module):
    def __init__(self, X, labels, your_batch , layers=None,lam1=1.0,lam2=0.001, sigma=None, gamma=1.0, lr=10**-3, kl1=1, kl2=1, nb=1, device=None, batch_size=128,
                 torch_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(DeepBID, self).__init__()
        if layers is None:
            layers = [X.shape[0], 512, 300]
        if device is None:
            device = torch_device
        self.layers = layers
        self.count = 0
        self.device = device
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        self.X = X.to(device)
        x_mean, x_std = torch.mean(self.X, dim=0), torch.std(self.X, dim=0)
        x_norm = (self.X - x_mean) / x_std
        self.X = x_norm
        self.csv_batch = your_batch
        self.n_batch = len(np.unique(self.csv_batch))
        self.labels = labels
        self.gamma = gamma
        self.lam1 = lam1
        self.lam2 = lam2
        self.sigma = sigma
        self.t_alpha = 1.0
        self.kl1 = kl1
        self.kl2 = kl2
        self.nb = nb
        self.batch_size = batch_size
        self.n_clusters = len(np.unique(self.labels))
        self.lr = lr
        self.n = X.shape[1] // batch_size +2
        self.encoder = torch.nn.Linear(self.layers[0], self.layers[1])
        self.decoder = torch.nn.Linear(self.layers[1], self.layers[0])
        self._dec_mean = torch.nn.Sequential(torch.nn.Linear(X.shape[0], X.shape[0]), MeanAct())
        self._dec_disp = torch.nn.Sequential(torch.nn.Linear(X.shape[0], X.shape[0]), DispAct())
        self.nb_loss = NBLoss().to(self.device)
        self._build_up()

    def _build_up(self):
        self.act = torch.tanh
        self.enc1 = torch.nn.Linear(self.layers[0], self.layers[1])
        self.enc2 = torch.nn.Linear(self.layers[1], self.layers[2])
        self.dec1 = torch.nn.Linear(self.layers[2], self.layers[1])
        self.dec2 = torch.nn.Linear(self.layers[1], self.layers[0])

    def forward(self, x):
        z = self.act(self.enc1(x))
        z = self.act(self.enc2(z))
        recons_x = self.act(self.dec1(z))
        recons_x = self.act(self.dec2(recons_x))
        return z, recons_x

    def _build_loss(self, z, x, d, u, recons_x):
        size = x.shape[0]
        loss = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size
        t = d*u  # t: m * c
        distances = utils.distance(z.t(), self.centroids)
        loss = (self.lam1 / 2 * torch.trace(distances.t().matmul(t)) / size)
        loss += self.lam2 * (self.enc1.weight.norm()**2 + self.enc1.bias.norm()**2) / size
        loss += self.lam2 * (self.enc2.weight.norm()**2 + self.enc2.bias.norm()**2) / size
        loss += self.lam2 * (self.dec1.weight.norm()**2 + self.dec1.bias.norm()**2) / size
        loss += self.lam2 * (self.dec2.weight.norm()**2 + self.dec2.bias.norm()**2) / size
        return loss

    def cul_batch_kl(self,csv_batch,cu):
        self.to(self.device)
        csv_batch = np.array(csv_batch)
        bt = torch.from_numpy(csv_batch)
        bt_count = torch.bincount(bt)
        qt = bt_count/bt.numel()
        B = torch.Tensor(self.n_clusters,bt.numel()) #4*5   
        B.copy_(bt)
        B = B.t()
        B = B.to(self.device)
        Z = torch.zeros(bt.numel(),self.n_clusters).to(self.device)
        su = self.U.sum(axis = 0)
        tensor_list = list()
        for i in range(0,self.n_batch):
            u0 = torch.where(B==i,cu,Z)
            su0 = u0.sum(axis = 0)
            pb0 = torch.div(su0,su)
            tensor_list.append(pb0)
        pb = torch.stack(tensor_list)
        Q = torch.Tensor(self.n_clusters,qt.numel())
        Q = Q.copy_(qt).t().to(self.device)
        kl = pb * torch.log(torch.div(pb+1e-6,Q+1e-6))
        kl_sum = kl.sum().item()    
        return kl_sum
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return torch.div(p.t(),p.sum(1)+1).t()

    def cal_latent(self,hidden, alpha):
        sum_y = torch.sum(torch.square(hidden), axis=1)
        num = -2.0 *torch.mm(hidden, hidden.t()) + torch.reshape(sum_y, [-1, 1]) + sum_y
        num = num / alpha
        num = torch.pow(1.0 + num, -(alpha + 1.0) / 2.0)
        zerodiag_num = num - torch.diag(torch.diag(num))
        latent_p = (zerodiag_num.t() / torch.sum(zerodiag_num, axis=1)).t()
        return num, latent_p
    
    def target_dis(self,latent_p):
        latent_q = torch.pow(latent_p, 2).t() / torch.sum(latent_p, axis = 1)
        latent_q = latent_q.t()
        return (latent_q.t() / torch.sum(latent_q, axis = 1)).t()
    
    def forwardAE(self, x):
        x=x.T
        h = self.act(self.encoder(x))
        h = self.act(self.decoder(h))
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        return _mean, _disp


    def run(self):
        self.to(self.device)
        self.pretrain()
        Z, _ = self(self.X.t())
        Z = Z.t().detach()
        idx = random.sample(list(range(Z.shape[0])), self.n_clusters)
        self.centroids = Z[:, idx] + 10 ** -8
        self._update_U(Z)
        print('Starting training......')
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.parameters(),lr = self.lr)
        loss = 0
        t0 = time()
        cu = self.U[self.n_clusters, :]
        index = []
        for epoch in range(25):
            D = self._update_D(Z)
            for i, batch in enumerate(train_loader):
                x, idx = batch
                idx_n = idx.numpy()
                index.extend(idx_n)
                optimizer.zero_grad()
                z, recons_x = self(x)
                d = D[idx, :]
                u = self.U[idx, :]
                self.loss = self._build_loss(z, x, d, u, recons_x)
                _, y_pred = self.U.max(dim=1)
                meanbatch, dispbatch= self.forwardAE(self.X)
                input = self.X.T
                self.loss_nb = self.nb_loss(input,meanbatch, dispbatch)
                self.num, self.latent_p = self.cal_latent(recons_x, self.t_alpha)
                self.latent_q = self.target_dis(self.latent_p)
                self.latent_p = self.latent_p + torch.diag(torch.diag(self.num))
                self.latent_q = self.latent_q + torch.diag(torch.diag(self.num))
                self.cross_entropy = -torch.sum(self.latent_q * torch.log(self.latent_p))
                self.entropy = -torch.sum(self.latent_q * torch.log(self.latent_q))
                self.loss_kl1 = self.cross_entropy - self.entropy
                self.loss_kl2 = self.cul_batch_kl(self.csv_batch,cu)
                self.loss_kl1 = self.kl1 * self.loss_kl1
                self.loss_kl2 = self.kl2 * self.loss_kl2
                self.loss_nb = self.nb * self.loss_nb
                self.loss_sum = self.loss + self.loss_kl1 + self.loss_kl2 + self.loss_nb
                self.loss_sum.backward()
                optimizer.step()
                
            Z, _ = self(self.X.t())
            Z = Z.t().detach()
            self.clustering(Z, 1)
            _, y_pred = self.U.max(dim=1)
            y_pred = y_pred.detach().cpu() + 1
            y_pred = y_pred.numpy()
            self.ARI = np.around(adjusted_rand_score(self.labels, y_pred),6)
            self.NMI = np.around(normalized_mutual_info_score(self.labels, y_pred), 6)
            print('epoch-{}, loss_sum={:.5f},loss={:.5f},loss_kl1={:.5f},loss_kl2={:.5f},loss_nb={:.5f},NMI={:.5f}, ARI={:.5f}'.format(epoch, self.loss_sum,self.loss,self.loss_kl1,self.loss_kl2,self.loss_nb,  self.NMI,self.ARI))
            
        
    def pretrain(self):
        string_template = 'Start pretraining-{}......'
        print(string_template.format(1))
        pre1 = PretrainDoubleLayer(self.X, self.layers[1], self.device, self.act, lr=self.lr)
        Z = pre1.run()
        self.enc1.weight = pre1.enc.weight
        self.enc1.bias = pre1.enc.bias
        self.dec2.weight = pre1.dec.weight
        self.dec2.bias = pre1.dec.bias
        print(string_template.format(2))

        pre2 = PretrainDoubleLayer(Z.detach(), self.layers[2], self.device, self.act, lr=self.lr)
        pre2.run()
        self.enc2.weight = pre2.enc.weight
        self.enc2.bias = pre2.enc.bias
        self.dec1.weight = pre2.dec.weight
        self.dec1.bias = pre2.dec.bias

    def _update_D(self, Z):
        if self.sigma is None:
            return torch.ones([Z.shape[1], self.centroids.shape[1]]).to(self.device)
        else:
            distances = utils.distance(Z, self.centroids, False)
            return (1 + self.sigma) * (distances + 2 * self.sigma) / (2 * (distances + self.sigma))

    def clustering(self, Z, max_iter=1):
        for i in range(max_iter):
            D = self._update_D(Z)
            T = D * self.U
            self.centroids = Z.matmul(T) / T.sum(dim=0).reshape([1, -1])
            self._update_U(Z)

    def _update_U(self, Z):
        if self.sigma is None:
            distances = utils.distance(Z, self.centroids, False)
        else:
            distances = self.adaptive_loss(utils.distance(Z, self.centroids, False), self.sigma)
        U = torch.exp(-distances / self.gamma)
        self.U = U / U.sum(dim=1).reshape([-1, 1])

    def adaptive_loss(self,D, sigma):
        return (1 + sigma) * D * D / (D + sigma)

if __name__ == '__main__':
    data, labels ,csv_batch = BID_data, BID_labels,BID_batches
    data = data.T
    for lam1 in [0.01,0.05,0.1,0.5,1,5,10]:
        print('lam1={}'.format(lam1))
        bid = DeepBID(data, labels, BID_batches,[data.shape[0], 1000, 500,200], lam1 = lam1,lam2=0.001,gamma=1,sigma=1,kl1=0.1, kl2=0.1, nb=1,  batch_size=128, lr=10**-5)
        bid.run()
        

