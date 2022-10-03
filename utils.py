import torch
import numpy as np
# latent vectors and label
# load best model
# save latent space in npz

# calculate mean of the cluster

# calculate and save latent direction
# in npz, check type and tc label

# 


def save_latent_vecs(dataloader, model, model_load_path, latent_save_path):
    # load best model
    model.load_state_dict(torch.load(model_load_path))
    print(f'loaded {model_load_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # run encoder on dataloader, extract latent vectors
    for data in dataloader:
        # ===================forward=====================
        particles_pos, labels, types = data
        particles_pos = particles_pos.to(device) 
        latent_vecs = model.encode(particles_pos)
        # stack tensors

    # save latent vectors
    np.savez(latent_vecs, labels, types) # (batch, dim_size), (batch, 1), (batch, dim_size)
    print(f'latent vectors saved into {latent_save_path}')


# file name is not reflecting what it is doing
def manipulate_latent(dataloader, model, latent_path, tc_src, tc_dst):
    # load best model
    model.load_state_dict(torch.load(model_load_path))
    print(f'loaded {model_load_path}')
    
    # load latent vectors
    npz = np.load(latent_path)
    labels = npz['labels']
    latent_vecs = npz['latent_vecs']

    # find direction in latent space
    # tc_dst - tc_src
    labels_dst = 0
    labels_src = 0
    latent_vecs_dst = 0
    latent_vecs_src = 0
    latent_direction = latent_vecs_dst - latent_vecs_src


    # encode input data, only run the correct label data

    # manipulate latent vectors 
    # decode manipulated vectors


    # run MSE_loss on simulated data and decoded data -> visualise it?
    # save both simulated data and decoded data into npz (batch_size, latent_dim)
    # add it into input argument
    print('hi')

# for visualisation in paraview 
def npz2vtk():
    print('written into vtk')