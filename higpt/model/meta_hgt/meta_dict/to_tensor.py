import torch
import os.path as osp
data_dir = '/root/paddlejob/workspace/env_run/output/HeteGPT/hetegpt/model/meta_hgt/meta_dict'

dsname = 'imdb'

node_feas_dict = torch.load(osp.join(data_dir, dsname, 'node_type.pt'))

for k, v in node_feas_dict.items():
    node_feas_dict[k] = torch.Tensor(v)

edge_feas_dict = torch.load(osp.join(data_dir, dsname, 'edge_type.pt'))

for k, v in edge_feas_dict.items():
    edge_feas_dict[k] = torch.Tensor(v)

torch.save(node_feas_dict, osp.join(data_dir, dsname, 'node_type.pt'))
torch.save(edge_feas_dict, osp.join(data_dir, dsname, 'edge_type.pt'))