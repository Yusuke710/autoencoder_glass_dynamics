import argparse
import os
import yaml

import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
#import seaborn as sns
import pandas as pd

from load_data import CustomImageDataset
from model import Autoencoder
from train import train


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


# dataloader 
train_dataset = CustomImageDataset(config['data_params']['train_dir'], transform=None, target_transform=None)
val_dataset = CustomImageDataset(config['data_params']['val_dir'], transform=None, target_transform=None)

train_dataloader = DataLoader(train_dataset, batch_size=config['data_params']['train_batch_size'], shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=config['data_params']['val_batch_size'], shuffle=True, num_workers=4)

# define model 
model = Autoencoder(in_channels = config['model_params']['in_channels'],latent_dim = config['model_params']['latent_dim'])


# train 
if config['train_params']['train']:
    train(train_dataloader, val_dataloader, model, num_epochs = config['train_params']['max_epochs'], lr = config['train_params']['lr'], weight_decay = config['train_params']['weight_decay'], seed = config['train_params']['manual_seed'], model_load_path= None, model_save_path = config['train_params']['model_save_path'])

# run validation dataset on best model and save latent vectors
'''
# make it a function
# visualise latent space using t-SNE
#1. load npz containing latent vectors
#2. performs t-SNE
# https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html
npz = np.load(config['latent_vec']['save_path'])
labels = npz['labels']
latent_vecs = npz['latent_vecs']

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(latent_vecs) 

df = pd.DataFrame()
df["labels"] = labels
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

# does it show up??
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 100),
                data=df).set(title="latent space T-SNE projection") 

# manipulate latent vec 
# input tc=0, manipulate latent vec and output tc=1000
# load npz
# 

# npz2vtk



'''
print('program ended')