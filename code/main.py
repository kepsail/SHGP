import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import load_data, set_params
from utils.evaluate import evaluate
from utils.cluster import kmeans
from module.att_lpa import *
from module.att_hgcn import ATT_HGCN
import warnings
import pickle as pkl
import os
import random
import time
warnings.filterwarnings('ignore')



dataset="mag"
args = set_params(dataset)
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")
train_percent=args.train_percent



# random seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



def train():
    epochs =args.epochs 
    label, ft_dict, adj_dict = load_data(dataset,train_percent)
    target_type = args.target_type
    num_cluster = int(ft_dict[target_type].shape[0]*args.compress_ratio) # compress the range of initial pseudo-labels. 
    num_class = np.unique(label[target_type][0]).shape[0]
    init_pseudo_label=0
    pseudo_pseudo_label = 0

    print('number of classes: ', num_cluster, '\n')
    layer_shape = []
    input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
    hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in args.hidden_dim]
    output_layer_shape = dict.fromkeys(ft_dict.keys(), num_cluster)

    layer_shape.append(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.append(output_layer_shape)

    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    model = ATT_HGCN(
        net_schema=net_schema,
        layer_shape=layer_shape,
        label_keys=list(label.keys()),
        type_fusion=args.type_fusion,
        type_att_size=args.type_att_size,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    if args.cuda and torch.cuda.is_available():
        model.cuda()
        for k in ft_dict:
            ft_dict[k] = ft_dict[k].cuda()
        for k in adj_dict:
            for kk in adj_dict[k]:
                adj_dict[k][kk] = adj_dict[k][kk].cuda()
        for k in label:
            for i in range(len(label[k])):
                label[k][i] = label[k][i].cuda()
    
    best = 1e9
    loss_list=[]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, embd, attention_dict = model(ft_dict, adj_dict)
        target_embd = embd[target_type]

        if epoch==0:
            init_pseudo_label = init_lpa(adj_dict,ft_dict,target_type,num_cluster)
            pseudo_label_dict = init_pseudo_label
        elif epoch < args.warm_epochs:
            pseudo_label_dict=init_pseudo_label
        else:
            pseudo_label_dict = att_lpa(adj_dict,init_pseudo_label,attention_dict,target_type,num_cluster)
            init_pseudo_label=pseudo_label_dict
        label_predict=torch.argmax(pseudo_label_dict[target_type], dim=1)
        logits = F.log_softmax(logits[target_type], dim=1)
        loss_train = F.nll_loss(logits, label_predict.long().detach())
        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train)
        if loss_train < best:
            best = loss_train

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train.item()),
        )

    #evaluate
    logits, embd, _ = model(ft_dict, adj_dict)
    target_embd = embd[target_type]
    label_target = label[target_type]
    true_label = label_target[0]
    idx_train = label_target[1]
    idx_val = label_target[2]
    idx_test = label_target[3]

    evaluate(target_embd, idx_train, idx_val, idx_test, true_label, num_class, isTest=True)



if __name__ == '__main__':
    train()
