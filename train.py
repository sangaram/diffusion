import torch
from transformers import Trainer, TrainingArguments
from pathlib import Path
from typing import Dict, Optional
from argparse import ArgumentParser
import os
import yaml
from model import GaussianDiffusionModel
from datasets import *

DATASET_REGISTRY = {
    'cifar10': 'CIFAR10Dataset',
    'celeba_hq_256': 'CelebAHQ256Dataset'
}

class CustomTrainer(Trainer):
    def compute_loss(self, model:GaussianDiffusionModel, inputs:Dict[str, torch.Tensor], return_outputs:Optional[bool]=False):
        batch = inputs['x']
        loss = model(batch)
        return loss

## Parsing CLI arguments
parser = ArgumentParser()
parser.add_argument("model_config", type=str, help="YAML file containing the model's parameters.")
parser.add_argument("training_config", type=str, help="YAML file container the training parameters.")
args = parser.parse_args()
model_config_path = Path(args.model_config)
if not model_config_path.exists():
    raise Exception(f"Error: Model configuration file does not exists: {args.model_config}")
model_config = yaml.safe_load(model_config_path.open())

training_config_path = Path(args.training_config)
if not training_config_path.exists():
    raise Exception(f"Error: Training configuration file does not exists: {args.training_config}")
training_config = yaml.safe_load(training_config_path.open())


## Model parameters
T = model_config['timesteps']
beta_1 = model_config['beta_1']
beta_T = model_config['beta_T']
betas = torch.linspace(beta_1, beta_T, T)
model_mean_type = model_config['model_mean_type']
model_var_type = model_config['model_var_type']
loss_type = model_config['loss_type']
channels = model_config['channels']
out_channels = model_config['out_channels']
ch_mult = model_config['ch_mult']
num_res_blocks = model_config['num_res_blocks']
att_levels = model_config['att_levels']
num_groups = channels // 4
resample_with_conv = model_config['resample_with_conv'] if 'resample_with_conv' in model_config else True
p = model_config['p'] if 'p' in model_config else .0

## Dataset parameters
data_root = "."

## Training parameters
save_dir = Path('checkpoints')
checkpoint = Path(training_config['checkpoint']) if 'checkpoint' in training_config else None
epochs = training_config['epochs']
batch_size = training_config['batch_size']
learning_rate = training_config['learning_rate']
warmup_ratio = training_config['warmup_ratio']
save_steps = training_config['save_steps']
save_total_limit = training_config['save_total_limit']

dataset_name = training_config['dataset']
if dataset_name not in DATASET_REGISTRY:
    raise NotImplementedError(f'The database {dataset_name} is not implemented. Please give an implemented one or implement yourself the desired dataset in the datasets module.')


## Loading the dataset
dataset = globals()[DATASET_REGISTRY[dataset_name]](root=data_root)


## Initializing the model
model = GaussianDiffusionModel(
    betas=betas,
    model_mean_type=model_mean_type,
    model_var_type=model_var_type,
    loss_type=loss_type,
    channels=channels,
    out_channels=out_channels,
    ch_mult=ch_mult,
    num_res_blocks=num_res_blocks,
    att_levels=att_levels,
    num_groups=num_groups,
    resample_with_conv=resample_with_conv,
    p=p
)

if not save_dir.exists():
    save_dir.mkdir(parents=True)
    
args = TrainingArguments(
    output_dir=save_dir,
    evaluation_strategy="no",
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    warmup_ratio=warmup_ratio,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    dataloader_num_workers=os.cpu_count()
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=dataset
)

os.environ["WANDB_SILENT"] = "true"

if __name__ == '__main__':
    if checkpoint is None:
        trainer.train()
    else:
        if not checkpoint.exists():
            raise Exception(f"Error: file {checkpoint.absolute()} doesn't exist.")
        
        trainer.train(resume_from_checkpoint=checkpoint)