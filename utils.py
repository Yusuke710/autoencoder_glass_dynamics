import os
import shutil
import re
# to run 1 image
import argparse
import yaml
from model import Autoencoder
import cv2
#import torchvision.transforms as transforms

# to download celebA dataset
import zipfile 
import gdown

import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import splitfolders

# latent vectors and label
# load best model
# save latent space in npz

# calculate mean of the cluster

# calculate and save latent direction
# in npz, check type and tc label

# 


def save_latent_vecs(dataloader, model, model_load_path, latent_save_path):
    
    # load best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.float()
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        print(f'loaded {model_load_path}')

    latent_vecs_list = []
    labels_list = []
    types_list = []
    # run encoder on dataloader, extract latent vectors
    with torch.no_grad():
        for data in dataloader:
            # ===================forward=====================
            particles_pos, labels, types = data
            particles_pos = particles_pos.to(device) 
            particles_pos = particles_pos.float()
            latent_vecs = model(particles_pos)[1]
            latent_vecs_list.append(latent_vecs)
            labels_list.append(labels)
            types_list.append(types)

        # stack tensors
        latent_vecs_list = torch.stack(latent_vecs_list, axis=0) # torch.Size([195, 1, 128])
        latent_vecs_list = torch.reshape(latent_vecs_list, (-1, model.latent_dim)) # reshape is needed to save into npz
        #latent_vecs_list = np.squeeze(latent_vecs_list, axis=1)
        #print(latent_vecs_list.shape)
        labels_list = np.stack(labels_list, axis=0) # (195, 1)
        labels_list = np.reshape(labels_list, (-1, 1))
        #print(labels_list.shape)
        types_list = np.stack(types_list, axis=0) # (195, 1, 4096)
        types_list = np.reshape(types_list, (-1, 4096))
        #types_list = np.squeeze(types_list, axis=1)
        #print(types_list.shape)


    # save latent vectors
    np.savez(latent_save_path, types = types_list, labels = labels_list, latent_vecs = latent_vecs_list) # (batch, dim_size), (batch, 1), (batch, dim_size)
    print(f'latent vectors saved into {latent_save_path}')


# file name is not reflecting what it is doing
def manipulate_latent(dataloader, model, model_load_path, latent_path, tc_src, tc_dst):
    # load latent vectors
    npz = np.load(latent_path)
    print('latent vectors loaded from {}'.format(latent_path))
    labels = npz['labels']
    types = npz['types']
    #print(labels.shape)
    latent_vecs = npz['latent_vecs']
    #print(latent_vecs.shape)

    # explore latent space
    latent_vecs_dst = []
    latent_vecs_src = []
    # match latent vectors with its labels
    for i in range(latent_vecs.shape[0]):
        if labels[i] == tc_src:
            latent_vecs_src.append(latent_vecs[i])
        elif labels[i] == tc_dst:
            latent_vecs_dst.append(latent_vecs[i])
    
    # find direction in latent space
    # tc_dst - tc_src
    #latent_vecs_src = np.array(latent_vecs_src)
    #latent_vecs_dst = np.array(latent_vecs_dst) 
    #print(len(latent_vecs_src))
    #print(len(latent_vecs_dst))
    latent_vecs_src = np.stack(latent_vecs_src, axis=0) # (n, 128)
    latent_vecs_dst = np.stack(latent_vecs_dst, axis=0)
    #print(latent_vecs_src.shape)
    #print(latent_vecs_dst.shape)
    latent_direction = np.mean(latent_vecs_dst, axis = 0) - np.mean(latent_vecs_src, axis = 0)
    latent_direction = np.reshape(latent_direction, (1, 128))
    #print(latent_direction.shape) # (1, 128)


    # load best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.float()
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path))
        print(f'loaded {model_load_path}')

    

    loss_fn = nn.MSELoss()

    # run encoder on test dataloader, extract latent vectors and add latent direction. 
    # Decode the modified latent vector and compare with the actual postions
    src_pos_list = []
    decoded_pos_list = []
    target_pos_list = []
    with torch.no_grad():
        for data in dataloader:
            # ===================forward=====================
            particles_pos, label, particle_type = data
            #print(label[0])
            # only run the correct label data
            if label[0] == tc_src: 
                src_pos_list.append(particles_pos)

                particles_pos = particles_pos.to(device) 
                particles_pos = particles_pos.float()
                latent_vecs = model.encode(particles_pos)
                # manipulate latent vectors 
                manipulated_vecs = latent_vecs + latent_direction
                # decode manipulated vectors
                decoded_pos = model.decode(manipulated_vecs) 
                #print(decoded_pos.shape)
                decoded_pos_list.append(decoded_pos)

            elif label[0] == tc_dst:
                #print(particles_pos.shape)
                target_pos_list.append(particles_pos)

        # TODO comapre loss for each test data from src to dst
        #for 

        # loss on simulated data and decoded data
        src_pos_list = torch.stack(src_pos_list, axis=0)
        decoded_pos_list = torch.stack(decoded_pos_list, axis=0)
        target_pos_list = torch.stack(target_pos_list, axis=0)
        src_pos_list = np.squeeze(src_pos_list, axis=1) # squeeze is needed for np.mean
        decoded_pos_list = np.squeeze(decoded_pos_list, axis=1)
        target_pos_list = np.squeeze(target_pos_list, axis=1)
        

        #print(torch.mean(decoded_pos_list, dim = 0).shape)
        #print(target_pos_list.shape)
        loss_reconst = loss_fn(torch.mean(decoded_pos_list, dim = 0), torch.mean(target_pos_list, dim = 0)) 
        diff_src2dst = loss_fn(torch.mean(src_pos_list, dim = 0), torch.mean(target_pos_list, dim = 0)) 
        print('Reconstruction MSE loss between predicted and target positions at {} is {}, difference between {} and target {} is {}'.format(tc_dst, loss_reconst, tc_src, tc_dst, diff_src2dst))

        # calculate pearson coefficient


    # save both simulated data and decoded data into npz (batch_size, latent_dim)
    filepath =  'params/{}to{}'.format(tc_src, tc_dst)
    np.savez(filepath, pred_pos = decoded_pos_list, target_pos = target_pos_list, types = types)  # check
    print('latent direction from {} to {} saved into {}'.format(tc_src, tc_dst, filepath))

# for visualisation in paraview 
def npz2vtk():
    print('written into vtk')


def visualise_tSNE(latent_path):
    # load latent vectors
    npz = np.load(latent_path)
    print('latent vectors loaded from {}'.format(latent_path))
    labels = npz['labels']
    labels = np.squeeze(labels, axis=1)
    #print(labels.shape)
    latent_vecs = npz['latent_vecs']

    #return
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    #z = tsne.fit_transform(latent_vecs) 

    df = pd.DataFrame()
    df["labels"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    # does it show up??
    scatterplot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.labels.tolist(),
                    #palette=sns.color_palette("hls", 100),
                    data=df).set(title="latent space T-SNE projection") 
    
    plt.savefig('t-SNE.png')
    #fig = scatterplot.get_figure()
    #fig.savefig("t-SNE.png") 

# TODO
# cross_posistion, calculate propensity and pearson coefficient
#def 

def split_folder(input_directory, output_directory):

    #np.random.seed(seed=seed) 
    #rands = np.random.rand(len(os.listdir(md_dir)))

    files = [file for file in os.listdir(input_directory)]
    for file in files:
        
        label = re.findall('seed.*', file) # extracts seedxxx
        label = label[0][:-4] # removes last 4 character .png
        label = label[4:] # remove 'seed'
        label = int(label)


        file = os.path.join(input_directory, file)
        # allocate data into train or valid based on probability
        if label >= 200:
            dst = 'image_data/train'
            # copy file from src to dst
            shutil.copy2(file, dst)
        elif label < 200 and label >= 100:
            dst = 'image_data/valid'
            # copy file from src to dst
            shutil.copy2(file, dst)
        elif label < 100:
            dst = 'image_data/test'
            # copy file from src to dst
            shutil.copy2(file, dst)



    '''
    splitfolders.ratio(input_directory, # The location of dataset
                    output=output_directory, # The output location
                    seed=42, # The number of seed
                    ratio=(.8, .1, .1), # The ratio of splited dataset
                    group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                    move=False # If you choose to move, turn this into True
                    )
    '''

def run_1image(model, model_load_path, image_path):
    print('running model on {}'.format(image_path))
    # load best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.float()
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print(f'loaded {model_load_path}')
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(64, 64))
    print(image.shape)
    #image_width = image.shape[0]
    #image = image.reshape(-1, 3)
    #image = np.transpose(image) #(3, 512*512)
    #image = image.reshape(3, 512, 512)
    #image = np.expand_dims(image, axis = 0) #(1, 3, 512, 512)
    #print(image.shape)
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    image = torch.unsqueeze(image, axis = 0)
    #print(image.shape)
    output, _= model(image)
    print(output.shape)
    output = output.detach().numpy()
    output = output.reshape(3, 64, 64)
    output = np.transpose(output, (1, 2, 0)) # 512, 512, 3
    plt.imshow((output * 255).astype(np.uint8)) # needs to be 512, 512, 3
    plt.savefig('imout.png')

def install_celebA():
    # Root directory for the dataset
    data_root = 'celeba'
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'
    url = 'https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing'
    # Path to download the dataset to
    download_path = f'{data_root}/celeba.zip'

    # Create required directories 
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        os.makedirs(dataset_folder)

    # Download the dataset from google drive
    gdown.download(url, download_path, quiet=False)

    # Unzip the downloaded file 
    with zipfile.ZipFile(download_path, 'r') as gziphandler:
        gziphandler.extractall(dataset_folder)



if __name__ == '__main__':
    #split_folder('screenshot', 'image_data')
    install_celebA()
    '''
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

    # define model 
    model = Autoencoder(in_channels = config['model_params']['in_channels'],latent_dim = config['model_params']['latent_dim'], hidden_dims = config['model_params']['hidden_dims'])
    image = 'image_data/test/coord_md_T0.40_Np4096_L61.30_tc100_seed9.png'
    model_load_path = 'params/best_model.th'

    run_1image(model, model_load_path, image)
    '''

    # python3 utils.py  --config config.yaml 