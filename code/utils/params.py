import argparse
import sys


def mag_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--dataset', type=str, default="mag")
    parser.add_argument('--target_type', type=str, default="p")
    parser.add_argument('--train_percent', type=float, default=0.08 )
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=[256,512])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=10)
    parser.add_argument('--compress_ratio', type=int, default=0.01)    
    
    args, _ = parser.parse_known_args()
    #30
    return args


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--train_percent', type=float, default=0.08 )
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=[256,256])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--target_type', type=str, default="p")
    parser.add_argument('--lr', type=float, default=0.008)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=10)
    parser.add_argument('--compress_ratio', type=int, default=0.05)  

    args, _ = parser.parse_known_args()
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--train_percent', type=float, default=0.08)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=[512,256])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--target_type', type=str, default="a")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=20)
    parser.add_argument('--compress_ratio', type=int, default=0.02)    
    
    args, _ = parser.parse_known_args()
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_true")
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--train_percent', type=float, default=0.08 )
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=[256,512])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--target_type', type=str, default="m")
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)
    parser.add_argument('--warm_epochs', type=int, default=50)
    parser.add_argument('--compress_ratio', type=int, default=0.06)    

    args, _ = parser.parse_known_args()
    return args


def set_params(dataset):
    if dataset == "mag":
        args = mag_params()
    if dataset == "acm":
        args = acm_params()
    if dataset == "dblp":
        args = dblp_params()
    if dataset == "imdb":
        args = imdb_params()
    return args
