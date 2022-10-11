import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import HeteGCNLayer



class ATT_HGCN(nn.Module):

	def __init__(self, net_schema, layer_shape, label_keys, type_fusion='att', type_att_size=64):
		super(ATT_HGCN, self).__init__()
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)

		self.embd2class = nn.ParameterDict()
		self.bias = nn.ParameterDict()
		self.label_keys = label_keys
		self.layer_shape=layer_shape
		for k in label_keys:
			self.embd2class[k] = nn.Parameter(torch.FloatTensor(layer_shape[-2][k], layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)
	def ini_embd2class(self):
		for k in self.label_keys:
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)

	def forward(self, ft_dict, adj_dict):
		attention_list=[]
		x_dict,attention_dict = self.hgc1(ft_dict, adj_dict)
		attention_list.append((attention_dict))
		x_dict= self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)

		x_dict,attention_dict= self.hgc2(x_dict, adj_dict)
		attention_list.append((attention_dict))

		logits = {}
		embd = {}
		for k in self.label_keys:
			embd[k] = x_dict[k]
			logits[k] = torch.mm(x_dict[k], self.embd2class[k]) + self.bias[k]
		return logits, embd,attention_list


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.elu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict
