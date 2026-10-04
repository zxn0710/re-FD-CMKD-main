import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random


class CremadDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = '/root/FD-CMKD-main/data/'
        self.visual_feature_path = '/root/FD-CMKD-main/data/CREMAD'
        self.audio_feature_path = '/root/FD-CMKD-main/data/CREMAD/Audio-299'

        # self.data_root = './data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        # self.visual_feature_path = '../CREMAD'
        # self.audio_feature_path = '../CREMAD/Audio-299'

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv
 
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = sorted(os.listdir(self.image[idx]))
        file_num = len(image_samples)
        pick_num = 1
        seg = int(file_num/pick_num)
        select_index = []

        for i in range(pick_num):
            if self.mode == 'train':
                index = random.randint(i*seg , i*seg + seg - 1)
            else:
                index = i*seg + int(seg/2)
            select_index.append(index)

        images = torch.zeros((1, 3, 224, 224))
        
        for i,n in enumerate(select_index):
            # img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = Image.open(os.path.join(self.image[idx], image_samples[n])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label