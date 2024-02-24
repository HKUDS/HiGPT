"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/mlfoundations/open_clip
"""

""" CLIP Model
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import datetime
import json
import logging
import os
import re
import time
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Dict

import numpy as np
import torch
import torch.nn.functional as F
# from lavis.common.registry import registry
# from lavis.common.utils import get_abs_path
from base_model import BaseModel
from models.clip_models.clip_outputs import ClipOutput, HeteClipOutputFeatures
# from models.clip_models.timm_model import TimmModel
from models.clip_models.transform import image_transform
from models.clip_models.utils import freeze_batch_norm_2d
# from lavis.tasks.multimodal_classification import MultimodalClassificationTask
from torch import nn
from models.meta_hgt.meta_hgtconv_bert_all import MetaHGTConv, MetaHGTConvCfg

from .pretrained import (
    download_pretrained,
    get_pretrained_url,
    list_pretrained_tag_models,
)

_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent.parent / f"configs/models/clip/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


@dataclass
class CLIPTextCfg:
    context_length: int
    vocab_size: int
    width: int
    heads: int
    layers: int


# @registry.register_model("clip")
# @registry.register_model("clip_feature_extractor")
class CLIP(BaseModel):

    def __init__(
        self,
        embed_dim: int,
        graph_cfg: MetaHGTConvCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
    ):
        from .tokenizer import tokenize

        super().__init__()

        self.tokenizer = tokenize
        self._loss = None

        if isinstance(graph_cfg, dict):
            graph_cfg = MetaHGTConvCfg(**graph_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        self.graph_encoder = MetaHGTConv(
            in_channels = graph_cfg.in_channels,
            out_channels = graph_cfg.out_channels,
            heads = graph_cfg.heads,
            dynamic = graph_cfg.dynamic,
            text_cfg = text_cfg, 
            layernorm = LayerNorm 
        )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_cfg.width)
        )
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.prompt_templates = openai_imagenet_template
        self.classifier = None

        self.init_parameters()

    @property
    def loss(self):
        if self._loss is None:
            from models.clip_models.loss import HeteClipLoss

            self._loss = HeteClipLoss()

        return self._loss

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))


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

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def encode_graph(self, graph: List[Dict[str, torch.Tensor]], des_order: List[List[str]]):
        graph_list = []
        for graph_dict in graph: 
            if 'subject' in graph_dict.keys():
                data_type = 'acm'
            elif 'movie' in graph_dict.keys():
                data_type = 'imdb'
            elif 'paper' in graph_dict.keys():
                data_type = 'dblp'
                new_conf_reps = torch.ones(graph_dict['conference'].num_nodes, 768)
                graph_dict['conference'].x = new_conf_reps
            graph_list.append(self.graph_encoder(graph_dict.x_dict, graph_dict.edge_index_dict, data_type=data_type))
        graph_embeds = []
        assert len(graph_list) == len(des_order)
        for idx, order in enumerate(des_order): 
            graph_embeds.extend([graph_list[idx][o] for o in order])
        graph_embeds = torch.cat(graph_embeds, dim = 0)
        return graph_embeds

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    # def forward(self, image, text):
    def forward(self, samples):
        r'''
        samples: 
        "graph": List[Dict],
        "text_input": List[str],
        "des_order": List[str], 
        "graph_id": graph_ids
        '''
        graph: List[Dict] = samples.get("graph")
        text: List[str] = samples.get("text_input")
        des_order: List[List[str]] = samples.get("des_order")

        if text is not None:
            text = self.tokenizer(text, self.context_length).to(self.token_embedding.weight.device)

        if graph is None:
            return self.encode_text(text) # N, dim
        elif text is None:
            return self.encode_graph(graph, des_order)
        graph_embeds = self.encode_graph(graph, des_order)
        graph_features = F.normalize(graph_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)
        assert graph_features.shape == text_features.shape

        loss = self.loss(graph_features, text_features, self.logit_scale.exp())

        # return {"loss": loss}
        return ClipOutput(
            intermediate_output=HeteClipOutputFeatures(
                graph_embeds=graph_embeds,
                graph_embeds_proj=graph_features,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
            ),
            loss=loss,
            logit_scale_exp=self.logit_scale.exp(),
        )

    def extract_features(self, samples):
        """
        Extract features from the model for samples.

        Keys allowed are "image" and "text_input" in samples.
        If either key is missing, the corresponding features are not extracted.

        Args:
            samples: dict of samples to extract features from.

        Returns:
            ClipOutputFeatures object with features for the samples.
        """
        graph: List[Dict] = samples.get("graph")
        text: List[str] = samples.get("text_input")
        des_order: List[List[str]] = samples.get("des_order")

        if text is not None:
            text = self.tokenizer(text)

        if graph is None:
            return self.encode_text(text) # N, dim
        elif text is None:
            return self.encode_graph(graph, des_order)
        graph_embeds = self.encode_graph(graph, des_order)
        graph_features = F.normalize(graph_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)
        assert graph_features.shape == text_features.shape

        return HeteClipOutputFeatures(
                graph_embeds=graph_embeds,
                graph_embeds_proj=graph_features,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
            )

    def predict(self, samples):
        image = samples["image"]
        targets = samples["label"]

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        logits = 100.0 * image_features @ self.classifier

        return {"predictions": logits, "targets": targets}

    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type == MultimodalClassificationTask:
            self.classifier = self.zero_shot_classifier(
                classnames=dataset.classnames,
                templates=self.prompt_templates,
            )

    def zero_shot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [
                    template(classname) for template in templates
                ]  # format with class
                texts = self.tokenizer(texts).to(self.device)  # tokenize

                class_embeddings = self.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    @classmethod
    def default_config_path(cls, model_type="base"):
        model_type = "ViT-B-32" if model_type == "base" else model_type

        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}. \n Available types: {}".format(
            model_type, cls.PRETRAINED_MODEL_CONFIG_DICT.keys()
        )
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    @classmethod
    def from_config(cls, cfg=None):
        model_name = cfg.model_type
        pretrained = cfg.pretrained

        precision = cfg.get("precision", "fp32")

        return create_model(
            model_name=model_name, pretrained=pretrained, precision=precision
        )

    def zero_shot_predict(self, image_path, categories):
        assert isinstance(
            categories, list
        ), f"categories must be a list, got {type(categories)}."
        assert os.path.exists(image_path), f"File {image_path} does not exist."

        from lavis.processors.clip_processors import ClipImageEvalProcessor
        from PIL import Image

        image_preprocess = ClipImageEvalProcessor()
        image = image_preprocess(Image.open(image_path)).unsqueeze(0)

        text = self.tokenizer(categories)

        with torch.no_grad():
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    def compute_sim_matrix(self, data_loader, **kwargs):
        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_features = []

        for i in range(0, num_text, text_bs):

            text = texts[i : min(num_text, i + text_bs)]
            text_input = self.tokenizer(text).to(self.device)

            text_feat = self.encode_text(text_input)
            text_feat = F.normalize(text_feat, dim=-1)

            text_features.append(text_feat)

        text_features = torch.cat(text_features, dim=0)

        image_features = []
        for samples in data_loader:
            image = samples["image"]

            image = image.to(self.device)
            image_feat = self.encode_image(image)
            image_feat = F.normalize(image_feat, dim=-1)

            image_features.append(image_feat)

        image_features = torch.cat(image_features, dim=0)

        sims_matrix_i2t = image_features @ text_features.t()
        sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_from_openai_state_dict(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=True,  # OpenAI models were trained with QuickGELU
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device("cpu")):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros(
        (batch_size, model.context_length), dtype=torch.int, device=device
    )
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,),
        ),
    )
    model.visual.image_size = image_size
    return


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def create_model(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model_name = model_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names

    if pretrained.lower() == "openai":
        logging.info(f"Loading pretrained {model_name} from OpenAI.")
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        logging.info(f"No pretrained weights loaded for {model_name} model.")
        if model_name in _MODEL_CONFIGS:
            logging.info(f"Loading {model_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(
                f"Model config for {model_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {model_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if "timm_model_name" in model_cfg.get("vision_cfg", {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg["vision_cfg"]["timm_model_pretrained"] = True
            else:
                assert (
                    False
                ), "pretrained image towers currently only supported for timm models"

        model = CLIP(**model_cfg)

        if pretrained:
            checkpoint_path = ""
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f"Loading pretrained {model_name} weights ({pretrained}).")
                model.load_state_dict(load_state_dict(checkpoint_path))
            else:
                logging.warning(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {model_name}."
                )

        model.to(device=device)
        if precision == "fp16":
            assert device.type != "cpu"
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val


def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_tag_models("openai")


def load_openai_model(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if get_pretrained_url(name, "openai"):
        model_path = download_pretrained(get_pretrained_url(name, "openai"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {list_openai_models()}"
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict()
            ).to(device)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd).to(device)

        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


openai_imagenet_template = [
    lambda c: f"a bad photo of a {c}.",
    lambda c: f"a photo of many {c}.",
    lambda c: f"a sculpture of a {c}.",
    lambda c: f"a photo of the hard to see {c}.",
    lambda c: f"a low resolution photo of the {c}.",
    lambda c: f"a rendering of a {c}.",
    lambda c: f"graffiti of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a cropped photo of the {c}.",
    lambda c: f"a tattoo of a {c}.",
    lambda c: f"the embroidered {c}.",
    lambda c: f"a photo of a hard to see {c}.",
    lambda c: f"a bright photo of a {c}.",
    lambda c: f"a photo of a clean {c}.",
    lambda c: f"a photo of a dirty {c}.",
    lambda c: f"a dark photo of the {c}.",
    lambda c: f"a drawing of a {c}.",
    lambda c: f"a photo of my {c}.",
    lambda c: f"the plastic {c}.",
    lambda c: f"a photo of the cool {c}.",
    lambda c: f"a close-up photo of a {c}.",
    lambda c: f"a black and white photo of the {c}.",
    lambda c: f"a painting of the {c}.",
    lambda c: f"a painting of a {c}.",
    lambda c: f"a pixelated photo of the {c}.",
    lambda c: f"a sculpture of the {c}.",
    lambda c: f"a bright photo of the {c}.",
    lambda c: f"a cropped photo of a {c}.",
    lambda c: f"a plastic {c}.",
    lambda c: f"a photo of the dirty {c}.",
    lambda c: f"a jpeg corrupted photo of a {c}.",
    lambda c: f"a blurry photo of the {c}.",
    lambda c: f"a photo of the {c}.",
    lambda c: f"a good photo of the {c}.",
    lambda c: f"a rendering of the {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"a photo of one {c}.",
    lambda c: f"a doodle of a {c}.",
    lambda c: f"a close-up photo of the {c}.",
    lambda c: f"a photo of a {c}.",
    lambda c: f"the origami {c}.",
    lambda c: f"the {c} in a video game.",
    lambda c: f"a sketch of a {c}.",
    lambda c: f"a doodle of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a low resolution photo of a {c}.",
    lambda c: f"the toy {c}.",
    lambda c: f"a rendition of the {c}.",
    lambda c: f"a photo of the clean {c}.",
    lambda c: f"a photo of a large {c}.",
    lambda c: f"a rendition of a {c}.",
    lambda c: f"a photo of a nice {c}.",
    lambda c: f"a photo of a weird {c}.",
    lambda c: f"a blurry photo of a {c}.",
    lambda c: f"a cartoon {c}.",
    lambda c: f"art of a {c}.",
    lambda c: f"a sketch of the {c}.",
    lambda c: f"a embroidered {c}.",
    lambda c: f"a pixelated photo of a {c}.",
    lambda c: f"itap of the {c}.",
    lambda c: f"a jpeg corrupted photo of the {c}.",
    lambda c: f"a good photo of a {c}.",
    lambda c: f"a plushie {c}.",
    lambda c: f"a photo of the nice {c}.",
    lambda c: f"a photo of the small {c}.",
    lambda c: f"a photo of the weird {c}.",
    lambda c: f"the cartoon {c}.",
    lambda c: f"art of the {c}.",
    lambda c: f"a drawing of the {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a black and white photo of a {c}.",
    lambda c: f"the plushie {c}.",
    lambda c: f"a dark photo of a {c}.",
    lambda c: f"itap of a {c}.",
    lambda c: f"graffiti of the {c}.",
    lambda c: f"a toy {c}.",
    lambda c: f"itap of my {c}.",
    lambda c: f"a photo of a cool {c}.",
    lambda c: f"a photo of a small {c}.",
    lambda c: f"a tattoo of the {c}.",
]
