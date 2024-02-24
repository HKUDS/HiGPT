"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
import os.path as osp


class BaseDataset(Dataset):
    def __init__(
        self, graph_processor=None, text_processor=None, datasets_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.datasets_root = datasets_root

        self.annotation = []
        for ann_path in ann_paths:
            ann_data = json.load(open(ann_path, "r"))
            self.annotation.extend(ann_data)
            # handle graph path
            for ann_item in ann_data: 
                ori_graph_path = ann_item["graph"]["graph"]
                # graph_file = ori_graph_path.split("/")[-1]
                # dsname = ann_path.split("/")[-3]
                graph_path = osp.join(self.datasets_root, ori_graph_path)
                assert osp.exists(graph_path), f"Graph file {graph_path} does not exist!"
                ann_item["graph"]["graph"] = graph_path

        self.graph_processor = graph_processor
        self.text_processor = text_processor
        print(self.annotation[0]['graph'])
        print(self.annotation[1]['graph'])
        print(self.annotation[2]['graph'])

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, graph_processor, text_processor):
        self.graph_processor = graph_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
