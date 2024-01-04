import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *

class Resnet18Triplet(nn.Module):
    def __init__(self, embedding_dimension=512, transfer=True):
        super(Resnet18Triplet, self).__init__()

        self.model = resnet18(pretrained=transfer)

        # Output embedding
        if transfer:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        self.embeding_fc = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.final_fc = nn.Linear(embedding_dimension, 30)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        h = self.model(images)
        embedding = self.embeding_fc(h)
        z = self.final_fc(h)

        return embedding, z

class Resnet34Triplet(nn.Module):
    def __init__(self, embedding_dimension=512, transfer=True):
        super(Resnet34Triplet, self).__init__()

        # self.model = resnet18(pretrained=transfer)
        self.model = resnet34(pretrained=transfer)
        # Output embedding
        if transfer:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        self.embeding_fc = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.final_fc = nn.Linear(embedding_dimension, 30)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        h = self.model(images)
        embedding = self.embeding_fc(h)
        z = self.final_fc(h)

        return embedding, z

class Resnet50Triplet(nn.Module):
    def __init__(self, embedding_dimension=512, transfer=True):
        super(Resnet50Triplet, self).__init__()

        # self.model = resnet18(pretrained=transfer)
        self.model = resnet50(pretrained=transfer)
        # Output embedding
        if transfer:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        self.embeding_fc = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.final_fc = nn.Linear(embedding_dimension, 30)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        h = self.model(images)
        embedding = self.embeding_fc(h)
        z = self.final_fc(h)

        return embedding, z

class Resnet101Triplet(nn.Module):
    def __init__(self, embedding_dimension=512, transfer=True):
        super(Resnet101Triplet, self).__init__()

        # self.model = resnet18(pretrained=transfer)
        self.model = resnet101(pretrained=transfer)
        # Output embedding
        if transfer:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)
        self.embeding_fc = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.final_fc = nn.Linear(embedding_dimension, 30)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        h = self.model(images)
        embedding = self.embeding_fc(h)
        z = self.final_fc(h)

        return embedding, z