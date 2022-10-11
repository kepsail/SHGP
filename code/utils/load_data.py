import numpy as np
import scipy.sparse as sp
import torch
import pickle
import torch.nn.functional as F



def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent = (1.0 - train_percent) / 2
    idx_train = torch.LongTensor(rand_idx[int(label_shape * 0.0): int(label_shape * train_percent)])
    idx_val = torch.LongTensor(
        rand_idx[int(label_shape * train_percent): int(label_shape * (train_percent + val_percent))])
    idx_test = torch.LongTensor(rand_idx[int(label_shape * (train_percent + val_percent)): int(label_shape * 1.0)])
    return idx_train, idx_val, idx_test



def load_odbmag_4017(train_percent):
    path='../data/mag/'
    feats = np.load(path+'feats.npz', allow_pickle=True)
    p_ft = feats['p_ft']
    a_ft = feats['a_ft']
    i_ft = feats['i_ft']
    f_ft = feats['f_ft']

    ft_dict = {}
    ft_dict['p'] = torch.FloatTensor(p_ft)
    ft_dict['a'] = torch.FloatTensor(a_ft)
    ft_dict['i'] = torch.FloatTensor(i_ft)
    ft_dict['f'] = torch.FloatTensor(f_ft)

    p_label = np.load(path+'p_label.npy', allow_pickle=True)
    p_label = torch.LongTensor(p_label)

    idx_train_p, idx_val_p, idx_test_p = train_val_test_split(p_label.shape[0], train_percent)

    label = {}
    label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]

    sp_a_i = sp.load_npz(path+'norm_sp_a_i.npz')
    sp_i_a = sp.load_npz(path+'norm_sp_i_a.npz')
    sp_a_p = sp.load_npz(path+'norm_sp_a_p.npz')
    sp_p_a = sp.load_npz(path+'norm_sp_p_a.npz')
    sp_p_f = sp.load_npz(path+'norm_sp_p_f.npz')
    sp_f_p = sp.load_npz(path+'norm_sp_f_p.npz')
    sp_p_cp = sp.load_npz(path+'norm_sp_p_cp.npz')
    sp_cp_p = sp.load_npz(path+'norm_sp_cp_p.npz')

    adj_dict = {'p': {}, 'a': {}, 'i': {}, 'f': {}}
    adj_dict['a']['i'] = sp_coo_2_sp_tensor(sp_a_i.tocoo())
    adj_dict['a']['p'] = sp_coo_2_sp_tensor(sp_a_p.tocoo())
    adj_dict['i']['a'] = sp_coo_2_sp_tensor(sp_i_a.tocoo())
    adj_dict['f']['p'] = sp_coo_2_sp_tensor(sp_f_p.tocoo())
    adj_dict['p']['a'] = sp_coo_2_sp_tensor(sp_p_a.tocoo())
    adj_dict['p']['f'] = sp_coo_2_sp_tensor(sp_p_f.tocoo())
    adj_dict['p']['citing_p'] = sp_coo_2_sp_tensor(sp_p_cp.tocoo())
    adj_dict['p']['cited_p'] = sp_coo_2_sp_tensor(sp_cp_p.tocoo())

    return label, ft_dict, adj_dict

def load_imdb_3228(train_percent):
    data_path = '../data/imdb/imdb3228.pkl'

    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)

        m_label = label['m'][0]
        idx_train_m, idx_val_m, idx_test_m = train_val_test_split(m_label.shape[0], train_percent)
        label['m'] = [m_label, idx_train_m, idx_val_m, idx_test_m]

        adj_dict['m']['a'] = adj_dict['m']['a'].to_sparse()
        adj_dict['m']['u'] = adj_dict['m']['u'].to_sparse()
        adj_dict['m']['d'] = adj_dict['m']['d'].to_sparse()

        adj_dict['a']['m'] = adj_dict['a']['m'].to_sparse()
        adj_dict['u']['m'] = adj_dict['u']['m'].to_sparse()
        adj_dict['d']['m'] = adj_dict['d']['m'].to_sparse()

    return label, ft_dict, adj_dict


def load_acm_4025(train_percent):
    data_path = '../data/acm/acm4025.pkl'
    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)

        p_label = label['p'][0]
        idx_train_p, idx_val_p, idx_test_p = train_val_test_split(p_label.shape[0], train_percent)
        label['p'] = [p_label, idx_train_p, idx_val_p, idx_test_p]

        adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
        adj_dict['p']['l'] = adj_dict['p']['l'].to_sparse()

        adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
        adj_dict['l']['p'] = adj_dict['l']['p'].to_sparse()

    return label, ft_dict, adj_dict


def load_dblp_4057(train_percent):
    data_path = '../data/dblp/dblp4057.pkl'
    with open(data_path, 'rb') as in_file:
        (label, ft_dict, adj_dict) = pickle.load(in_file)
        labels = {}
        for key in label.keys():
            key_label = label[key][0]
            idx_train, idx_val, idx_test = train_val_test_split(key_label.shape[0], train_percent)
            labels[key] = [key_label, idx_train, idx_val, idx_test]

        adj_dict['p']['a'] = adj_dict['p']['a'].to_sparse()
        adj_dict['p']['c'] = adj_dict['p']['c'].to_sparse()
        adj_dict['p']['t'] = adj_dict['p']['t'].to_sparse()

        adj_dict['a']['p'] = adj_dict['a']['p'].to_sparse()
        adj_dict['c']['p'] = adj_dict['c']['p'].to_sparse()
        adj_dict['t']['p'] = adj_dict['t']['p'].to_sparse()

    return labels, ft_dict, adj_dict

def load_data(dataset,train_percent):
    if dataset=="mag":
        label, ft_dict, adj_dict=load_odbmag_4017(train_percent)
    if dataset=="dblp":
        label, ft_dict, adj_dict=load_dblp_4057(train_percent)
    if dataset=="acm":
        label, ft_dict, adj_dict=load_acm_4025(train_percent)
    if dataset=="imdb":
        label, ft_dict, adj_dict=load_imdb_3228(train_percent)

    return label, ft_dict, adj_dict


