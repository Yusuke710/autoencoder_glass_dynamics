import os
import re

import numpy as np
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt

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

    """
    # for simulated coordinates 
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        npz = np.load(file_path) 
        raw_pos = npz['positions'] # (1, 4096, 3)
        raw_pos = np.squeeze(raw_pos, axis=0) # (4096, 3)
        '''
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
        '''
        particles_pos = np.transpose(raw_pos) # (3, 4096)
        particles_pos = particles_pos.reshape(3, 64, 64) 
        # extract tc and type
        labels = re.findall('t.*_', self.files[idx]) # extracts tcxxx
        labels = labels[0][:-1] # removes last character
        types = npz['types']

        if self.transform:
            particles_pos = self.transform(particles_pos)
        if self.target_transform:
            labels = self.target_transform(labels)
        return particles_pos, labels, types
    
    """
    # for images
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        image = cv2.imread(file_path)
        image = cv2.resize(image, dsize=(64, 64))

        # check image
        #plt.savefig('iminput.png')

        image = np.transpose(image, (2, 0, 1)) # from 512, 512, 3 to 3, 512, 512

        #print(image.shape)
        #particles_pos = particles_pos.reshape(3, 64, 64) 
        # extract tc and type
        labels = re.findall('t.*_', self.files[idx]) # extracts tcxxx
        labels = labels[0][:-1] # removes last character

        if self.transform:
            particles_pos = self.transform(particles_pos)
        if self.target_transform:
            labels = self.target_transform(labels)
        return image, labels
    

