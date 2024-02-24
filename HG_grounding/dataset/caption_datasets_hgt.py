"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from dataset.base_dataset_graph import BaseDataset
from PIL import Image
import torch
from torch.utils.data.dataloader import default_collate
import math


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["graph"],
                "caption": ann["caption"],
                "graph": sample["graph"],
            }
        )

class HeteCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, graph_processor, text_processor, datasets_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(graph_processor, text_processor, datasets_root, ann_paths)

        self.graph_ids = {}
        n = 0
        for ann in self.annotation:
            g_id = ann["graph_id"]
            if g_id not in self.graph_ids.keys():
                self.graph_ids[g_id] = n
                n += 1
        # handle graph 
        self.node_type_feas_dict_dblp = torch.load('./models/meta_hgt/meta_dict/dblp/node_type.pt')
        for k in self.node_type_feas_dict_dblp.keys(): 
            self.node_type_feas_dict_dblp[k] = torch.Tensor(self.node_type_feas_dict_dblp[k])
        self.edge_type_feas_dict_dblp = torch.load('./models/meta_hgt/meta_dict/dblp/edge_type.pt')
        for k in self.edge_type_feas_dict_dblp.keys(): 
            self.edge_type_feas_dict_dblp[k] = torch.Tensor(self.edge_type_feas_dict_dblp[k])

        self.node_type_feas_dict_acm = torch.load('./models/meta_hgt/meta_dict/acm/node_type.pt')
        for k in self.node_type_feas_dict_acm.keys(): 
            self.node_type_feas_dict_acm[k] = torch.Tensor(self.node_type_feas_dict_acm[k])
        self.edge_type_feas_dict_acm = torch.load('./models/meta_hgt/meta_dict/acm/edge_type.pt')
        for k in self.edge_type_feas_dict_acm.keys(): 
            self.edge_type_feas_dict_acm[k] = torch.Tensor(self.edge_type_feas_dict_acm[k])

        self.node_type_feas_dict_imdb = torch.load('./models/meta_hgt/meta_dict/imdb/node_type.pt')
        for k in self.node_type_feas_dict_imdb.keys(): 
            self.node_type_feas_dict_imdb[k] = torch.Tensor(self.node_type_feas_dict_imdb[k])
        self.edge_type_feas_dict_imdb = torch.load('./models/meta_hgt/meta_dict/imdb/edge_type.pt')
        for k in self.edge_type_feas_dict_imdb.keys(): 
            self.edge_type_feas_dict_imdb[k] = torch.Tensor(self.edge_type_feas_dict_imdb[k])
        

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        
        ann = self.annotation[index]

        dsname = ann['graph_id'].split('_')[0]

        # print(self.datasets_root, dsname, ann["graph"])

        graph_path = os.path.join(ann["graph"]['graph'])
        graph_dict = torch.load(graph_path)
        des_dict = ann['des_dict']
        
        if 'subject' in graph_dict.x_dict.keys(): 
            edge_type_feas_dict = self.edge_type_feas_dict_acm
            node_type_feas_dict = self.node_type_feas_dict_acm
            des_dict.pop('author') if 'author' in des_dict else None
        elif 'movie' in graph_dict.x_dict.keys(): 
            edge_type_feas_dict = self.edge_type_feas_dict_imdb
            node_type_feas_dict = self.node_type_feas_dict_imdb
        elif 'paper' in graph_dict.x_dict.keys(): 
            edge_type_feas_dict = self.edge_type_feas_dict_dblp
            node_type_feas_dict = self.node_type_feas_dict_dblp
            new_conf_reps = torch.ones(graph_dict['conference'].num_nodes, 768)
            graph_dict['conference'].x = new_conf_reps
            des_dict.pop('term') if 'term' in des_dict else None
            des_dict.pop('paper') if 'paper' in des_dict else None
        else: 
            raise NotImplementedError

        graph_id = ann['graph_id']

        # print(dsname)
        # print(ann["description"])
        
        # des_dict.pop('paper') if 'paper' in des_dict else None

        des_order = des_dict.keys()
        
        # caption = self.text_processor(ann["description"])
        caption = []
        for k in des_order: 
            des_list = des_dict[k]
            # replaced_des_list = ['nan' if type(x) != str else x for x in des_list]
            caption_tmp = [self.text_processor(des) for des in des_list]
            
            caption.extend(caption_tmp)

        return {
            "graph": graph_dict,
            "text_input": caption,
            "des_order": des_order, 
            "graph_id": graph_id, 
            "edge_type_feas_dict": edge_type_feas_dict, 
            "node_type_feas_dict": node_type_feas_dict
            # "graph": self.img_ids[ann["image_id"]],
        }
    def collater(self, samples):
        # print(type(samples))
        graph_dict = [sample['graph'] for sample in samples]
        
        caption = []
        for sample in samples: 
            caption.extend(sample['text_input'])
        des_order = [sample['des_order'] for sample in samples]
        graph_ids = [sample['graph_id'] for sample in samples]
        node_type_feas_dict = [sample['node_type_feas_dict'] for sample in samples]
        edge_type_feas_dict = [sample['edge_type_feas_dict'] for sample in samples]
        
        return {
            "graph": graph_dict,
            "text_input": caption,
            "des_order": des_order, 
            "graph_id": graph_ids, 
            # "edge_type_feas_dict": samples[0]['edge_type_feas_dict'],
            # "node_type_feas_dict": samples[0]['node_type_feas_dict']
            "edge_type_feas_dict": edge_type_feas_dict,
            "node_type_feas_dict": node_type_feas_dict
        }



class CaptionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, graph_processor, text_processor, graph_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(graph_processor, text_processor, graph_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        graph_path = os.path.join(self.graph_root, ann["graph"])
        graph = torch.load(graph_path)
        graph = self.graph_processor(graph)

        # image = self.vis_processor(image)

        return {
            "graph": graph,
            "graph_id": ann["graph_id"],
            "instance_id": ann["instance_id"],
        }
    def collater(self, samples):
        samples['graph'] = self.graph_collator(samples['graph'])
        return default_collate(samples)
