import os
import wandb
import torch
from torch import nn
from sklearn.manifold import TSNE
from sys import stdout
from Config import *
from Network import Resnet18Triplet
from TripletDataset2 import get_dataloader
from Tripletloss import TripletLoss
import numpy as np
import time

""" Device Confirmation """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
savepath = ''
def cal_map(gallery_cls, pred_cls, label_cls, pred_d, t, k, num=5, margin=0.1):
    gallery_cls = np.array(gallery_cls)
    label_cls = np.array(label_cls)
    pred_cls = gallery_cls[pred_cls]
    pred_cls[pred_d > margin] = 'new_id'
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
    knn = dist.topk(100, largest=False, dim=0)

    return knn.values[1:], knn.indices[1:]

def tsne(x, y):
    x = gallery.cpu().numpy()
    y = np.array(y)
    y_uni = np.unique(y)
    oh_y = np.zeros_like(y, dtype='int32')
    for i, c in enumerate(y_uni):
        oh_y[y==c] = i
    model = TSNE(random_state=0)
    digits_tsne = model.fit_transform(x, oh_y)
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, y_uni.shape[0]))
    for i in range(oh_y.max()):
        plt.plot(digits_tsne[oh_y == i, 0], digits_tsne[oh_y == i, 1], 'o', color=colors[i], markersize=0.5)

def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings, pred = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size:]


    anc_pred = pred[:batch_size]
    pos_pred = pred[batch_size: batch_size * 2]


    return anc_embeddings, pos_embeddings, model, anc_pred, pos_pred

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


            progress = batch_idx / dataloaders.__len__()
            stdout.write("\r ===== test: %f%% completed =====" % progress)
            stdout.flush()
    print(score5 / (batch_idx+1))

    # wandb.log({
    #     "MAP5": score5 / (batch_idx + 1)
    #
    #
    #
    # })
    # if best_map < score5 / (batch_idx+1):
    #     best_map = score5 / (batch_idx+1)
    #     state = {
    #         'epoch': epoch,
    #         'map': best_map,
    #         'model_state_dict': model.state_dict()
    #     }
    #     torch.save(state, './weight/best_' + savepath)
    # return best_map


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    # wandb.init(project='Whale_triplet')
    # savepath = 'res18, transfer=False, clsifier x, lr=0.001'
    # wandb.run.name = savepath
    # wandb.config={
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "epochs": epochs
    # }
    # wandb.config.update()


    model=Resnet18Triplet(transfer=False).cuda()
    checkpoint = torch.load('./weight/best_res18, transfer=False, clsifier x, lr=0.001')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_map = 0
    dataset_train = get_dataloader(
        train_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/all/',
        valid_root_dir='/home/whddltkf0889/dataset/train_images_crop_after_reshape/test/',
        train_csv_name='./all_list.csv', valid_csv_name='./val_list.csv',
        num_train_triplets=num_train_triplets, num_valid_triplets=20000,
        batch_size=batch_size, num_workers=0)

    for epoch in range(0, epochs):
        dataset_train.generate_triplets()
        loader_gallery = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                                          shuffle=False, num_workers=0)


        # 학습 때마다 triplet pair을 다시 만들어 줌
        gallery, gallery_cls, hard_pos_ind = make_gallery(model, loader_gallery)
        best_map = test(model, loader_gallery, gallery, gallery_cls, best_map)

        #dataset_train.fit(hard_pos_ind)


        # test 구현해야 함



