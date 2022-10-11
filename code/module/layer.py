import torch
import torch.nn as nn
import torch.nn.functional as F



class HeteGCNLayer(nn.Module):

	def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
		super(HeteGCNLayer, self).__init__()
		self.net_schema = net_schema
		self.in_layer_shape = in_layer_shape
		self.out_layer_shape = out_layer_shape

		self.hete_agg = nn.ModuleDict()
		for k in net_schema:
			self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape[k], type_fusion, type_att_size)


	def forward(self, x_dict, adj_dict):
		attention_dict={}
		ret_x_dict = {}
		for k in self.hete_agg.keys():
			ret_x_dict[k],attention_dict[k] = self.hete_agg[k](x_dict, adj_dict[k])

		return ret_x_dict,attention_dict



class HeteAggregateLayer(nn.Module):
	
	def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
		super(HeteAggregateLayer, self).__init__()
		
		self.nb_list = nb_list
		self.curr_k = curr_k
		self.type_fusion = type_fusion
		self.W_rel = nn.ParameterDict()
		for k in nb_list:
			try:
				self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[k], out_shape))
			except KeyError as ke:
				self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[self.curr_k], out_shape))
			finally:
				nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)
		
		self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape[curr_k], out_shape))
		nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

		self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
		nn.init.xavier_uniform_(self.bias.data, gain=1.414)

		if type_fusion == 'att':
			self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
			self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
			nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
			self.w_att = nn.Parameter(torch.FloatTensor(2*type_att_size, 1))
			nn.init.xavier_uniform_(self.w_att.data, gain=1.414)


	def forward(self, x_dict, adj_dict):
		attention_curr_k=0
		self_ft = torch.mm(x_dict[self.curr_k], self.w_self)
		
		nb_ft_list = [self_ft]
		nb_name = [self.curr_k+'_self']
		for k in self.nb_list:
			try:
				nb_ft = torch.mm(x_dict[k], self.W_rel[k])
			except KeyError as ke:
				nb_ft = torch.mm(x_dict[self.curr_k], self.W_rel[k])
			finally:
				nb_ft = torch.spmm(adj_dict[k], nb_ft)
				nb_ft_list.append(nb_ft)
				nb_name.append(k)

		if self.type_fusion == 'mean':
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)
			attention=[]
		elif self.type_fusion == 'att':
			att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
			att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
			att_input = torch.cat([att_keys, att_query], 1)
			att_input = F.dropout(att_input, 0.5, training=self.training)
			e = F.elu(torch.matmul(att_input, self.w_att))
			attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0,1), dim=1)   #4025*3
			agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
			
		output = agg_nb_ft + self.bias

		return output,attention