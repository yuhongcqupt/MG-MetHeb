import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
import sys

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, xsd=None, xsw=None,pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        self.has_xd = xsd is not None
        self.has_xw = xsw is not None
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            if self.has_xd:
                xd_padding = np.repeat(xsd[-1:], num_padding, axis=0)
                xsd = np.concatenate([xsd, xd_padding], axis=0)
            if self.has_xw:
                xw_padding = np.repeat(xsw[-1:], num_padding, axis=0)
                xsw = np.concatenate([xsw, xw_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // batch_size)
        self.xs = xs
        self.ys = ys
        self.xsd, self.xsw = xsd, xsw

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.xs, self.ys = self.xs[permutation], self.ys[permutation]
        if self.has_xd:
            self.xsd = self.xsd[permutation]
        if self.has_xw: self.xsw = self.xsw[permutation]
    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start = self.batch_size * self.current_ind
                end = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start:end, ...]
                y_i = self.ys[start:end, ...]
                if self.has_xd and self.has_xw:
                    xd_i = self.xsd[start:end, ...]
                    xw_i = self.xsw[start:end, ...]
                    yield (x_i, xd_i, xw_i, y_i)
                elif self.has_xd:
                    xd_i = self.xsd[start:end, ...]
                    yield (x_i, xd_i, y_i)
                else:
                    yield (x_i, y_i)
                self.current_ind += 1
        return _wrapper()

class StandardScaler():

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std  = np.array(std)


        self.std = np.where(self.std == 0, 1e-6, self.std)

    def transform(self, data):
        data = np.asarray(data)

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        data = np.asarray(data)
        return (data * self.std) + self.mean



def sym_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):

    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype, area='SF'):
    if area == 'Seoul':
        adj_mx_O = pd.read_csv(pkl_filename[0], header=None, dtype=np.float64).values + np.identity(304)
        adj_mx_I = pd.read_csv(pkl_filename[1], header=None, dtype=np.float64).values + np.identity(304)
        adj = [asym_adj(adj_mx_O), asym_adj(adj_mx_I)]
    else:
        if area == 'Seattle':
            adj_mx = np.load(pkl_filename)
        else:
            sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)

        if adjtype == "scalap":
            adj = [calculate_scaled_laplacian(adj_mx)]
        elif adjtype == "normlap":
            adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            adj = [sym_adj(adj_mx)]
        elif adjtype == "transition":
            adj = [asym_adj(adj_mx)]
        elif adjtype == "doubletransition":
            adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
        elif adjtype == "identity":
            adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"

    # return sensor_ids, sensor_id_to_ind, adj
    return adj


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None):
    data = {}
    for category in ['train','val','test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))

        data['x_' + category]  = cat_data['x'].astype(np.float32)
        data['y_' + category]  = cat_data['y'].astype(np.float32)


        data['xd_' + category] = cat_data['x_daily'].astype(np.float32)  if ('x_daily' in cat_data.files)  else None
        data['xw_' + category] = cat_data['x_weekly'].astype(np.float32) if ('x_weekly' in cat_data.files) else None


    F = 6  

    x_train = data['x_train']
    mean = x_train[..., :F].mean(axis=(0,1,2))
    std  = x_train[..., :F].std(axis=(0,1,2))
    scaler = StandardScaler(mean=mean, std=std)

    for category in ['train','val','test']:
        data['x_' + category][..., :F]  = scaler.transform(data['x_' + category][..., :F])
        if data['xd_' + category] is not None:
            data['xd_' + category][..., :F] = scaler.transform(data['xd_' + category][..., :F])
        if data['xw_' + category] is not None:
            data['xw_' + category][..., :F] = scaler.transform(data['xw_' + category][..., :F])



    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size,
                                    xsd=data['xd_train'], xsw=data['xw_train'])
    data['val_loader']   = DataLoader(data['x_val'], data['y_val'], valid_batch_size,
                                    xsd=data['xd_val'], xsw=data['xw_val'])
    data['test_loader']  = DataLoader(data['x_test'], data['y_test'], test_batch_size,
                                    xsd=data['xd_test'], xsw=data['xw_test'])

    data['scaler'] = scaler

 
    x = cat_data['x'][:, -1, :, 0].transpose()

    xmin = np.min(x, axis=1, keepdims=True)
    xmax = np.max(x, axis=1, keepdims=True)
    rng  = xmax - xmin
    rng[rng == 0] = 1.0
    xn = (x - xmin) / rng

    norm = np.sqrt(np.sum(xn ** 2, axis=1, keepdims=True))
    norm[norm == 0] = 1.0
    xn = xn / norm
    xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)

    adpmat = np.matmul(xn, xn.transpose())
    adp_thresh = 0.95
    adpmat[adpmat < adp_thresh] = 0.0

    return data, adpmat
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


