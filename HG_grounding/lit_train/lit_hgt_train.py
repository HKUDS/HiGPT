import sys
from pathlib import Path
# support running without installing as a package
# wd = Path(__file__).parent.parent.resolve()
# sys.path.append(str(wd))

from lightning import Trainer, seed_everything
from lit_models.lit_hgt import HeteCLIP
from lit_models.lit_hgt_data import HeteLitDataModule
# from gpt_config import Config as GPTConfig
import lightning.pytorch.callbacks as plc
from typing import Any, Optional, Dict, List, Sequence
from dataclasses import dataclass, field
import transformers


@dataclass
class TrainingArguments:
    max_epochs: Optional[int] = field(default=100)
    learning_rate: Optional[float] = field(default=1e-5)
    devices: Optional[int] = field(default=4)

@dataclass
class ModelArguments:
    # graph
    graph_in_channels: Optional[int] = field(default=768)
    graph_out_channels: Optional[int] = field(default=768)
    graph_heads: Optional[int] = field(default=12)
    graph_dynamic: Optional[bool] = field(default=True)

    # transformer
    context_length: Optional[int] = field(default=128)
    vocab_size: Optional[int] = field(default=49408)
    text_width: Optional[int] = field(default=768)
    text_heads: Optional[int] = field(default=8)
    text_layers: Optional[int] = field(default=6)
    
    # clip
    embed_dim: Optional[int] = field(default=768)
    quick_gelu: Optional[bool] = field(default=False)

    gnn_type: Optional[str] = field(default='meta')
    
@dataclass
class DataArguments:
    batch_size: Optional[int] = field(default=1)
    micro_batch_size: Optional[int] = field(default=1)
    dataset_dir: Optional[str] = field(default='datasets')
    datalist: Optional[str] = field(default='HetOAG')

def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='loss',
        mode='min',
        patience=5,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='loss',
        filename='best-{epoch:02d}-{loss:.3f}',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def train():
    seed_everything(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.gradient_accumulation_iters = data_args.batch_size // data_args.micro_batch_size
    trainer = Trainer(
        devices=training_args.devices,
        max_epochs=training_args.max_epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        precision="32-true",
        enable_progress_bar=True,
        callbacks=load_callbacks(),
        strategy='ddp', 
        gradient_clip_val=0.5, gradient_clip_algorithm="value"
    )

    pl_module = HeteCLIP(training_args, model_args, data_args)
    data_module = HeteLitDataModule(data_args)

    # print('*'*20 ,'grad none', '*'*20)
    # for n, p in pl_module.named_parameters():
    #     if p.grad is None: 
    #         print(n)

    trainer.fit(pl_module, data_module)

if __name__ == "__main__":
    train()