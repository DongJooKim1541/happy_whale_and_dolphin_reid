import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
path = '/home/whddltkf0889/dataset/train_images_crop_after_reshape/'
csv_name = './data/train.csv'
img_list = os.listdir(path+'all/')

csv_file = pd.read_csv(csv_name)
train_id_list = []
test_id_list = []
all_id_list = []
all_id_dict = {}
toggle = [0, 0]
test_3id_dict = {}
all_img_dict = {}
is_train = True
for i in range(csv_file.__len__()):
    id = csv_file['individual_id'][i]
    img = csv_file['image'][i]

    if not id in all_id_list:
        all_id_list.append(id)
        all_id_dict[id] = 1
        all_img_dict[id] = [img]
    else:
        all_id_dict[id] += 1
        all_img_dict[id].append(img)

for id in all_id_dict.keys():
    n = all_id_dict[id]
    if n == 1:
        toggle[0] += 1
        if toggle[0] // 5 == 0:
            test_id_list.append(all_img_dict[id][0])
        else:
            train_id_list.append(all_img_dict[id][0])
    elif n == 2:
        for i in range(2):
            r = random.randint(1, 5)
            if r == 1:
                test_id_list.append(all_img_dict[id][i])
            else:
                train_id_list.append(all_img_dict[id][i])
    else:
        for i in range(n):
            r = random.randint(1, 5)
            if r == 1:
                test_id_list.append(all_img_dict[id][i])
            else:
                train_id_list.append(all_img_dict[id][i])

for file_name in train_id_list:
    path_origin = path + 'all/' + file_name
    path_copy = path + 'train/' + file_name
    shutil.copy(path_origin, path_copy)
for file_name in test_id_list:
    path_origin = path + 'all/' + file_name
    path_copy = path + 'test/' + file_name
    shutil.copy(path_origin, path_copy)

value = []
for id in all_id_list:
    value.append(all_id_dict[id])

fig = plt.figure()

