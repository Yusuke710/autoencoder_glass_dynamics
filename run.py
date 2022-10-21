import argparse
import os
import yaml

import numpy as np
from torch.utils.data import DataLoader


from load_data import CustomImageDataset
from model import Autoencoder
from train import train, train_img
from utils import save_latent_vecs, manipulate_latent, visualise_tSNE, run_1image


# read config 
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    #metavar='FILE',
                    help =  'path to the config file',
                    default='config.yaml')

args = parser.parse_args()            
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


#image = 'image_data/test/coord_md_T0.40_Np4096_L61.30_tc100_seed9.png'
#model_load_path = None #'params/best_model.th'
# define model 
#model = Autoencoder(in_channels = config['model_params']['in_channels'],latent_dim = config['model_params']['latent_dim'], hidden_dims = config['model_params']['hidden_dims'])
#run_1image(model, model_load_path, image)

# dataloader 
train_dataset = CustomImageDataset(config['data_params']['train_dir'], transform=None, target_transform=None)
val_dataset = CustomImageDataset(config['data_params']['val_dir'], transform=None, target_transform=None)
test_dataset = CustomImageDataset(config['data_params']['test_dir'], transform=None, target_transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=config['data_params']['train_batch_size'], shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=config['data_params']['val_batch_size'], shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=config['data_params']['test_batch_size'], shuffle=True, num_workers=4)

# define model 
model = Autoencoder(in_channels = config['model_params']['in_channels'],latent_dim = config['model_params']['latent_dim'], hidden_dims = config['model_params']['hidden_dims'])


# train 
if config['train_params']['train']:
    train_img(train_dataloader, val_dataloader, model, num_epochs = config['train_params']['max_epochs'], lr = config['train_params']['lr'], weight_decay = config['train_params']['weight_decay'], seed = config['train_params']['manual_seed'], model_load_path= None, model_save_path = config['train_params']['model_save_path'])
    # train(.....) for coordinates

# save latent vectors
if config['latent_vec']['save']:
    save_latent_vecs(dataloader = train_dataloader, model = model, model_load_path = None, latent_save_path = config['latent_vec']['save_path'])


# manipulate latent vectors
print('\n Manipulating latent vectors')
manipulate_latent(dataloader = test_dataloader, model = model, model_load_path = None, latent_path = config['latent_vec']['save_path'], tc_src = config['latent_vec']['tc_src'], tc_dst = config['latent_vec']['tc_dst'])

# visualise latent space
print('\n Visualising latent space using t-SNE')
visualise_tSNE(latent_path = config['latent_vec']['save_path'])


print('program ended')