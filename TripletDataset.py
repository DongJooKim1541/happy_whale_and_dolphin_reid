import os

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pickle
# num_triplets: 10000
from torchvision.datasets.folder import pil_loader


class TripletDataset(Dataset):

    def __init__(self, root_dir, csv_name, num_triplets, transform=None, train=True):

        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.train = train
        self.training_triplets = self.generate_triplets(self.df, self.train, None)


        #print(self.root_dir)
        #print(self.df)
        """
        for i in range(0,len(self.training_triplets)):
            print(self.training_triplets[i])
        """
    def fit(self, positive_ind):
        self.training_triplets = self.generate_triplets(self.df, self.train, positive_ind)
    def generate_triplets(self, df, train, positive_ind):

        def make_dictionary_for_whale_class(df, train):
            '''
            - whale_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
            '''
            whale_classes = dict()
            for idx, label in enumerate(df['individual_id']):
                if label not in whale_classes:
                    whale_classes[label] = []
                whale_classes[label].append((df.iloc[idx]['individual_id'], df.iloc[idx]['image'], df.iloc[idx]['species']))
            # if train:
            #     with open('whale_classes_train', 'rb') as fr:
            #         whale_classes = pickle.load(fr)
            # else:
            #     with open('whale_classes_val', 'rb') as fr:
            #         whale_classes = pickle.load(fr)
            return whale_classes

        triplets = []

        # print("len(ids): ",len(ids))
        # if train:
        all_ids = df['individual_id'].unique() # 중복되지 않게 individual_id 가져오기
        whale_classes = make_dictionary_for_whale_class(df, train) # 생성한 dictionary 가져오기
        k = 0
        for ids in all_ids:
            pos_class = ids
            for i in range(whale_classes[pos_class].__len__()):
                ianchor = i
                if positive_ind == None:
                    ipositive = np.random.randint(0, whale_classes[pos_class].__len__())
                    if whale_classes[pos_class].__len__() > 1:
                        while ianchor == ipositive:
                            ipositive = np.random.randint(0, whale_classes[pos_class].__len__())
                else:
                    ipositive = positive_ind[k]

                anchor_image = whale_classes[pos_class][ianchor][1]
                anchor_sp = whale_classes[pos_class][ianchor][2]
                positive_image = whale_classes[pos_class][ipositive][1]
                positive_sp = whale_classes[pos_class][ipositive][2]

                triplets.append([pos_class, anchor_image, positive_image, anchor_sp, positive_sp])
                k += 1
        #
        # else:
        #     for i in range(df.__len__()):
        #         anchor_id = df['individual_id'][i]
        #         anchor_image = df['image'][i]
        #         anchor_sp = df['species'][i]
        #         triplets.append(
        #             [anchor_id, anchor_image, anchor_image, anchor_sp, anchor_sp])

        return triplets

    def __getitem__(self, idx):
        pos_class, anchor_image, positive_image, anchor_sp, positive_sp = self.training_triplets[idx]
        anc_img = os.path.join(self.root_dir, str(anchor_image))
        pos_img = os.path.join(self.root_dir, str(positive_image))

        anc_img = pil_loader(anc_img)
        pos_img = pil_loader(pos_img)


        sample = {'anc_img': anc_img, 'pos_img': pos_img,'pos_class': pos_class,
                  'pos_sp' : positive_sp}

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)


def get_dataloader(train_root_dir, valid_root_dir,
                   train_csv_name, valid_csv_name,
                   num_train_triplets, num_valid_triplets,
                   batch_size, num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        'gallery': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])}

    whale_dataset = {
        'valid': TripletDataset(root_dir=valid_root_dir,
                                csv_name=valid_csv_name,
                                num_triplets=num_valid_triplets,
                                transform=data_transforms['valid'],
                                train=False),
        'train': TripletDataset(root_dir=train_root_dir,
                                    csv_name=train_csv_name,
                                    num_triplets=num_train_triplets,
                                    transform=data_transforms['train'],
                                train=True),
        'gallery': TripletDataset(root_dir=train_root_dir,
                                csv_name=train_csv_name,
                                num_triplets=num_train_triplets,
                                transform=data_transforms['gallery'],
                                train=False)
    }

    return whale_dataset['train'], whale_dataset['valid'], whale_dataset['gallery']
