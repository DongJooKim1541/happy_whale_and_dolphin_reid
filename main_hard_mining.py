import os
import wandb
import torch
from torch import nn

from sys import stdout
from Config import *
from Network import Resnet18Triplet
from TripletDataset import get_dataloader
from Tripletloss import TripletLoss
import numpy as np
import time

""" Device Confirmation """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
savepath = ''
def cal_map(gallery_cls, pred_cls, label_cls, pred_d, t, k, num=5):
    gallery_cls = np.array(gallery_cls)
    label_cls = np.array(label_cls)
    pred_cls = gallery_cls[pred_cls]
    # pred_cls[pred_d > 0.5] = 'new_id'
    score = np.zeros(label_cls.shape[0])

    for i in range(label_cls.shape[0]):
        if not label_cls[i] in gallery_cls:
            label_cls[i] = 'new_id'
            t += 1
        else:
            k += 1

    for i in range(num):
        cls = pred_cls[num-i-1]
        tf = label_cls == cls
        score[tf] = 1/(num-i)
        # score[tf] = 1


    return score.mean(), t, k

def knn(ref, query, k=5):
    dist = torch.norm(ref.unsqueeze(1) - query.unsqueeze(0), dim=2, p=None)
    knn = dist.topk(k, largest=False, dim=0)

    return knn.values, knn.indices


def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings, pred = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size:]


    anc_pred = pred[:batch_size]
    pos_pred = pred[batch_size: batch_size * 2]


    return anc_embeddings, pos_embeddings, model, anc_pred, pos_pred

def train(model, optimizer, dataloaders):
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

    wandb.log({
        "Number of valid training triplets": num_valid_training_triplets,
        "training_triplets_loss: ": training_triplets_loss
    })

def make_gallery(model, dataloaders):
    model.eval()
    gallery_cls = []
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloaders):
            gallery_cls = gallery_cls + batch_sample['pos_class']
            anc_imgs = batch_sample['anc_img'].cuda()
            anc_embedding, _ = model(anc_imgs)

            anc_embedding = anc_embedding.cpu()
            if batch_idx == 0:
                gallery = anc_embedding
            else:
                gallery = torch.cat((gallery, anc_embedding))

            progress = batch_idx / dataloaders.__len__()
            stdout.write("\r ===== make_gallery %f%% completed =====" % progress)
            stdout.flush()
    hard_positive_ind = []
    gallery_cls = np.array(gallery_cls)
    uniq_cls = np.unique(gallery_cls)
    for c in uniq_cls:
        ref = gallery[c == gallery_cls]
        for q in ref:
            dist = torch.norm(ref.unsqueeze(1) - q.unsqueeze(0).unsqueeze(0), dim=2, p=None)
            _, ind = dist.topk(1, largest=True, dim=0)
            hard_positive_ind.append(ind.item())

    return gallery, np.array(gallery_cls), hard_positive_ind

def test(model, dataloaders, gallery, gallery_cls, best_map):
    model.eval()
    t = 0
    k = 0
    score5 = 0.
    score10 = 0.
    score20 = 0.
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(dataloaders):
            cls = batch_sample['pos_class']
            anc_imgs = batch_sample['anc_img'].cuda()
            anc_embedding, _ = model(anc_imgs)

            anc_embedding = anc_embedding.cpu()
            pred_d, pred_ind = knn(gallery, anc_embedding, 5)
            score_t, t, k = cal_map(gallery_cls, pred_ind, cls, pred_d, t, k, 5)
            score5 += score_t
            pred_d, pred_ind = knn(gallery, anc_embedding, 10)
            score_t, t, k = cal_map(gallery_cls, pred_ind, cls, pred_d, t, k, 10)
            score10 += score_t
            pred_d, pred_ind = knn(gallery, anc_embedding, 20)
            score_t, t, k = cal_map(gallery_cls, pred_ind, cls, pred_d, t, k, 20)
            score20 += score_t
            progress = batch_idx / dataloaders.__len__()
            stdout.write("\r ===== test: %f%% completed =====" % progress)
            stdout.flush()
    print(score5 / (batch_idx+1))
    print(score10 / (batch_idx+1))
    print(score20 / (batch_idx + 1))
    print(t)
    print(k)
    wandb.log({
        "MAP5": score5 / (batch_idx + 1),
        "MAP10": score10 / (batch_idx + 1),
        "MAP20": score20 / (batch_idx + 1)


    })
    if best_map < score5 / (batch_idx+1):
        best_map = score5 / (batch_idx+1)
        state = {
            'epoch': epoch,
            'map': best_map,
            'model_state_dict': model.state_dict()
        }
        torch.save(state, './weight/best_' + savepath)
    return best_map


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    wandb.init(project='Whale_triplet')
    savepath = 'res18, transfer=False, clsifier x, lr=0.001'
    wandb.run.name = savepath
    wandb.config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    wandb.config.update()


    model=Resnet18Triplet(transfer=False).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_map = 0
    dataset_train, dataset_test, dataset_gallery = get_dataloader(
        train_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/train/',
        valid_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/test/',
        train_csv_name='./train_list.csv', valid_csv_name='./val_list.csv',
        num_train_triplets=num_train_triplets, num_valid_triplets=20000,
        batch_size=batch_size, num_workers=4)

    for epoch in range(0, epochs):
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                        num_workers=4)
        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                                      num_workers=4)
        loader_gallery = torch.utils.data.DataLoader(dataset_gallery, batch_size=batch_size,
                                                          shuffle=False, num_workers=4)

        # 학습 때마다 triplet pair을 다시 만들어 줌
        current_time=time.time()
        dataset_train.train = True
        train(model, optimizer, loader_train)
        dataset_train.train = False
        gallery, gallery_cls, hard_pos_ind = make_gallery(model, loader_gallery)
        best_map = test(model, loader_test, gallery, gallery_cls, best_map)

        #dataset_train.fit(hard_pos_ind)


        # test 구현해야 함



