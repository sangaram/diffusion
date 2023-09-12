import torch
import yaml
from argparse import ArgumentParser
from model import GaussianDiffusionModel
from pathlib import Path
import matplotlib.pyplot as plt

## Parsing CLI arguments
parser = ArgumentParser()
parser.add_argument("command", type=str, help="The command name. Either 'image' or 'video'")
parser.add_argument("model_config", type=str, help="YAML file containing the model's parameters.")
parser.add_argument("model_weights", type=str, help="Path to the file containing the model weights.")
parser.add_argument("--image_shape", type=int, nargs=3, help="The image shape")
args = parser.parse_args()

command = args.command

model_config_path = Path(args.model_config)
if not model_config_path.exists():
    raise Exception(f"Error: Model configuration file does not exists: {args.model_config}")

model_config = yaml.safe_load(model_config_path.open())
model_weights = Path(args.model_weights)
if not model_weights.exists():
    raise Exception(f"Error: the file ${model_weights.absolute()} doesn't exist.")
image_shape = args.image_shape
assert isinstance(image_shape, list) and len(image_shape) == 3, f"Error: The image shape must be a list of 3 integers"



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

## Instaciating the model and loading the weights
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

state_dict = torch.load(model_weights, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

if command == 'image':
    ## Generating the image
    image_shape = tuple([1] + image_shape)
    img = model.generate(shape=image_shape)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
elif command == 'video':
    ## Generating the video
    image_shape = tuple(image_shape)
    img = model.generation_evolution(shape=image_shape, filename='out.mp4')
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.imshow(img.permute(1,2,0))
    plt.show()
else:
    raise NotImplementedError(f"Error: Command {command} not implemented.")