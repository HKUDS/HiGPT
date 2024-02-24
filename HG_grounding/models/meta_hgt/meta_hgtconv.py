import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from models.meta_hgt.meta_linear import MetaHeteroDictLinear, MetaHeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index
from models.meta_hgt.hgt_constants import NODE_TYPE_DICT, EDGE_TYPE_DICT
from dataclasses import dataclass
import torch.nn as nn
from models.clip_models.tokenizer import tokenize

@dataclass
class MetaHGTConvCfg:
    in_channels: int 
    out_channels: int
    heads: int
    dynamic: bool = True


class MetaHGTConv(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dynamic: bool = False,
        text_transformer = None, 
        text_cfg = None, 
        layernorm = None, 
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.kqv_lin = MetaHeteroDictLinear(text_cfg.width, self.in_channels,
                                        self.out_channels * 3, dynamic)

        self.out_lin = MetaHeteroDictLinear(text_cfg.width, self.out_channels, self.out_channels, dynamic)

        dim = out_channels // heads

        self.k_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)
        self.v_rel = MetaHeteroLinear(text_cfg.width, dim, dim, dynamic)

        self.skipTrans = nn.Linear(text_cfg.width, 1) # node aware, skip: 1

        self.p_relTrans = nn.Linear(text_cfg.width, heads) # edge aware, p_rel: 1, heads

        self.tokenizer = tokenize

        act_layer = nn.GELU

        self.transformer = text_transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.context_length = text_cfg.context_length
        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_cfg.width)
        )
        self.ln_final = layernorm(text_cfg.width)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj], 
        edge_type_feas_dict: Dict[EdgeType, Tensor], 
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(edge_index_dict.keys())
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}

        edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(edge_index_dict.keys())
        }
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        edge_feas_dict = {edge_types_map[k]: v for k, v in edge_type_feas_dict.items()}

        k = self.k_rel(ks, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec, edge_feas_dict).view(H, -1, D).transpose(0, 1)

        return k, v, offset
    def encode_text(self, text): 
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] # 1, width

        return x

    def _construct_p_rel(self, edge_type_feas_dict: Dict[EdgeType, Tensor]):
        p_rel = {k: self.p_relTrans(v).unsqueeze(0) for k, v in edge_type_feas_dict.items()}
        return p_rel
    def _construct_skip(self, node_type_feas_dict: Dict[EdgeType, Tensor]):
        skip = {k: self.skipTrans(v) for k, v in node_type_feas_dict.items()}
        return skip

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        F = self.out_channels
        H = self.heads
        D = F // H

        node_type_feas_dict = {k: self.encode_text(self.tokenizer(NODE_TYPE_DICT[k], self.context_length).to(self.token_embedding.weight.device)).squeeze(0) for k in x_dict.keys()}

        edge_type_feas_dict = {k: self.encode_text(self.tokenizer(EDGE_TYPE_DICT[k], self.context_length).to(self.token_embedding.weight.device)).squeeze(0) for k in edge_index_dict.keys()}

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict, node_type_feas_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict, edge_type_feas_dict)
        p_rel = self._construct_p_rel(edge_type_feas_dict)
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=p_rel)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr,
                             size=None)

        dst_node_types = set([key[-1] for key in edge_index_dict.keys()])

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        }, node_type_feas_dict)

        skip = self._construct_skip(node_type_feas_dict)
        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
