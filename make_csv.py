import pandas as pd
import os
from collections import OrderedDict
import numpy as np

csv_path = './data/train.csv'
path = '/home/whddltkf0889/dataset/train_images_crop_after_reshape/all/'
img_list = os.listdir(path)
all_csv = pd.read_csv(csv_path)

image = []
species = []
individual_id = []
temp = {}
for img_name in img_list:
    df = all_csv[all_csv['image'] == img_name]
    if not df['image'].to_list() == []:
        image = image + df['image'].to_list()
        species = species + df['species'].to_list()
        individual_id = individual_id + df['individual_id'].to_list()
        temp[df['species'].to_list()[0]] = 0
ind = np.array(species)
print(temp.keys().__len__())
for i, (k) in enumerate(temp.keys()):
    ind[ind == k] = i
species = ind.tolist()



dic = {
    'image':image,
    'species': species,
    'individual_id': individual_id
}

df2 = pd.DataFrame.from_dict(dic)
df2.to_csv('all_list.csv')