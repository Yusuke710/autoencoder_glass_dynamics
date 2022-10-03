import os
import re

import numpy as np
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        #super(Dataset, self).__init__()

        # iterate through data_dir
        self.files = [file for file in os.listdir(data_dir)]
        print(len(self.files))
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz = np.load(file_path) 
        raw_pos = npz['positions'] # (1, 4096, 3)
        raw_pos = np.squeeze(raw_pos, axis=0) # (4096, 3)
        x_list = []
        y_list = []
        z_list = []
        for single_particle in raw_pos:
            #print('single particle')
            #print(single_particle)
            x = single_particle[0]
            y = single_particle[1]
            z = single_particle[2]
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        particles_pos = np.array([x_list, y_list, z_list]) # (3, 4096)
        particles_pos = particles_pos.reshape(3, 64, 64) 
        #print(particles_pos.shape)
        # extract tc and type
        labels = re.findall('t.*_', self.files[idx]) # extracts tcxxx
        labels = labels[:-1] # removes last character
        types = npz['types']

        if self.transform:
            particles_pos = self.transform(particles_pos)
        if self.target_transform:
            labels = self.target_transform(labels)
        return particles_pos, labels, types

