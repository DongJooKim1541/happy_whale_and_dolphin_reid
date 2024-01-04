import os
import wandb
import torch
from torch import nn

from sys import stdout
from Config import *
from Network import Resnet18Triplet
from TripletDataset2 import get_dataloader
from Tripletloss import TripletLoss
import numpy as np
import time

""" Device Confirmation """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
savepath = ''

def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings, pred = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size:]


    anc_pred = pred[:batch_size]
    pos_pred = pred[batch_size: batch_size * 2]


    return anc_embeddings, pos_embeddings, model, anc_pred, pos_pred
def knn(ref, query, k=5):
    dist = torch.norm(ref.unsqueeze(1) - query.unsqueeze(0), dim=2, p=None)
    knn = dist.topk(k, largest=False, dim=0)

    return knn.values, knn.indices
def train(model, optimizer, dataloaders, e):
    loss = nn.CrossEntropyLoss()
    Triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2)
    model.train()

    num_valid_training_triplets = 0
    training_triplets_loss = 0
    acc = 0
    for batch_idx, batch_sample in enumerate(dataloaders):
        anc_sp = batch_sample['pos_sp'].cuda()
        pos_sp = batch_sample['pos_sp'].cuda()

        anc_imgs = batch_sample['anc_img'].float()
        pos_imgs = batch_sample['pos_img'].float()

        # Concatenate the input images into one tensor because doing multiple forward passes would create
        #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
        #  issues
        all_imgs = torch.cat((anc_imgs, pos_imgs))  # Must be a tuple of Torch Tensors

        anc_embeddings, pos_embeddings, model, anc_pred, pos_pred = forward_pass(
            imgs=all_imgs,
            model=model,
            batch_size=anc_imgs.shape[0]
        )
        ref = torch.cat((anc_embeddings, pos_embeddings))
        pos_class = np.array(batch_sample['pos_class'])
        ref_cls = np.concatenate((pos_class, pos_class))
        _, ind = knn(ref, anc_embeddings, k=anc_imgs.shape[0] * 2)
        ind = ind.cpu()

        neg_embeddings = torch.zeros_like(anc_embeddings)
        batch_size = anc_imgs.shape[0]
        for b in range(batch_size):
            tf = pos_class != ref_cls[ind[batch_size-b-1]]
            neg_embeddings[tf] = ref[ind[batch_size-b-1][tf]]


        triplet_loss = Triplet_loss(anc_embeddings, pos_embeddings, neg_embeddings.cuda())

        # Calculate triplet loss

        iteration_triplet_loss = triplet_loss.detach().cpu()
        if np.isnan(iteration_triplet_loss) or np.isinf(iteration_triplet_loss):
            continue

        # Calculating number of triplets that met the triplet selection method during the epoch

        training_triplets_loss += triplet_loss.item()
        cl_loss = loss(anc_pred, anc_sp) + loss(pos_pred, pos_sp)

        total_loss = 0.01*cl_loss + triplet_loss
        # total_loss = triplet_loss
        acc += ((anc_pred.max(dim=1)[1] == anc_sp)*1.).mean()
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        progress = batch_idx / dataloaders.__len__()
        stdout.write("\r ===== train: %f%% completed ===== " % progress)
        stdout.flush()

    training_triplets_loss /= dataloaders.__len__()


    # Print training statistics for epoch and add to log
    print('\nEpoch {}:\ttraining_triplets_loss in epoch: {}'.format(epoch, training_triplets_loss))
    print(' acc: {}'.format(acc / (batch_idx+1) * 100))
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict()
    }
    torch.save(state, './weight/best_' + savepath +'.pth')




# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':

    savepath = 'train'

    model=Resnet18Triplet(transfer=False).cuda()
    checkpoint = torch.load('./weight/best_res18, transfer=False, clsifier x, lr=0.001')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_map = 0
    dataset_train= get_dataloader(
        train_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/all/',
        valid_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/test/',
        train_csv_name='./all_list.csv', valid_csv_name='./val_list.csv',
        num_train_triplets=num_train_triplets, num_valid_triplets=20000,
        batch_size=batch_size, num_workers=4)

    for epoch in range(0, 2000):
        dataset_train.generate_triplets()
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
        train(model, optimizer, loader_train, epoch)




