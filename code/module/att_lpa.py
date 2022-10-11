import torch.nn.functional as F
import copy
import torch



def gen_rand_label(ft_dict,num_cluster):
    rand_label_dict = dict.fromkeys(ft_dict.keys())
    for k in ft_dict:
        rand_label = torch.randint(num_cluster, (ft_dict[k].shape[0],))
        rand_label = F.one_hot(rand_label, num_cluster).type(dtype=torch.float32)
        rand_label_dict[k] = rand_label

    return rand_label_dict



def cat_homo_adj(adj_dict):
    for k in adj_dict:
        print(k)

    return None



def lpa(init_label_dict,adj_dict,target_type,num_cluster,max_iter=1000):
    pseudo_label_dict = copy.deepcopy(init_label_dict)
    for k in pseudo_label_dict:
        pseudo_label_dict[k] = pseudo_label_dict[k].cuda()
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_label_list=[]
    soft_label=0
    for i in range(max_iter):
        for k in net_schema:
            k_nbs_label_list = []
            for kk in net_schema[k]:
                try:
                    soft_label = torch.spmm(adj_dict[k][kk], pseudo_label_dict[kk])
                except KeyError as ke:
                    soft_label = torch.spmm(adj_dict[k][kk], pseudo_label_dict[k])
                finally:
                    k_nbs_label_list.append(soft_label)
            new_k_label = torch.cat([nb_label.unsqueeze(1) for nb_label in k_nbs_label_list], 1)

            new_k_label = new_k_label.sum(1)
            new_k_label = torch.argmax(new_k_label, dim=1)
            new_k_label = F.one_hot(new_k_label, num_cluster).type(dtype=torch.float32)
            pseudo_label_dict[k] = new_k_label
            if k==target_type:
                target_label_list.append(new_k_label)
        if len(target_label_list)>1:
            if target_label_list[-2].equal(target_label_list[-1]):
                break

    return pseudo_label_dict



def init_lpa(adj_dict,ft_dict,target_type,num_cluster):
    run_num = 1
    for i in range(run_num):
        init_label_dict = gen_rand_label(ft_dict, num_cluster)
        pseudo_label_dict = lpa(init_label_dict,adj_dict,target_type,num_cluster)
        
    return pseudo_label_dict



def att_lpa(adj_dict,init_pseudo_label,attention_dict,target_type,num_cluster,max_iter=1000):

    pseudo_label_dict = copy.deepcopy(init_pseudo_label)
    current_label_dict=copy.deepcopy(init_pseudo_label)

    for k in pseudo_label_dict:
        pseudo_label_dict[k] = pseudo_label_dict[k].cuda()
    for k in current_label_dict:
        current_label_dict[k] = current_label_dict[k].cuda()
    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    target_label_list=[]
    soft_label=0
    for _ in range(max_iter):
        for m in range(len(attention_dict)):
            for k in net_schema:
                k_nbs_label_list = []
                k_nbs_label_list.append(pseudo_label_dict[k])
                for kk in net_schema[k]:
                    try:
                        soft_label = torch.spmm(adj_dict[k][kk], current_label_dict[kk])
                    except KeyError as ke:
                        soft_label = torch.spmm(adj_dict[k][kk], current_label_dict[k])
                    finally:
                        k_nbs_label_list.append(soft_label)
                pseudo_label_dict[k]=torch.cat([nb_label.unsqueeze(1) for nb_label in k_nbs_label_list], 1).mul(attention_dict[m][k].unsqueeze(-1)).sum(1)

            for k in net_schema:
                current_label_dict[k]=pseudo_label_dict[k]

        for k in net_schema:
            new_k_label = torch.argmax(pseudo_label_dict[k], dim=1)
            if k==target_type:
                target_label_list.append(new_k_label)
            pseudo_label_dict[k] = F.one_hot(new_k_label, num_cluster).type(dtype=torch.float32)
            current_label_dict[k]=pseudo_label_dict[k]
        if len(target_label_list)>1:
            if target_label_list[-2].equal(target_label_list[-1]):
                break
                
    return pseudo_label_dict