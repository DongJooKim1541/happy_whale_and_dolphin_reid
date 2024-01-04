import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import *
import torch


class EfficientTriplet(nn.Module):
    def __init__(self, embedding_dimension=64, transfer=True):
        super(EfficientTriplet, self).__init__()
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

        # Output embedding
        if transfer:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
        input_features_fc_layer = self.model.classifier.fc.in_features
        self.model.classifier.fc = nn.Linear(input_features_fc_layer, 256, bias=False)
        self.midle_fc = nn.Linear(256, embedding_dimension, bias=False)
        self.final_fc = nn.Linear(embedding_dimension, 30)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        embedding = self.midle_fc(embedding)
        z = F.normalize(embedding, p=2, dim=1)
        z = self.final_fc(z)

        return embedding, z
