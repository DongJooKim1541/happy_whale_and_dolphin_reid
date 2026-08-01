import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import torch


class ResNetTriplet(nn.Module):
    """ResNet-based triplet network for whale/dolphin re-identification.

    Supports ResNet18, ResNet34, ResNet50, ResNet101 backbones.
    Outputs both embedding (512-dim feature) and species classification (30-way).
    """

    def __init__(self, model_name="resnet18", embedding_dimension=512, num_classes=30, pretrained=True):
        """
        Args:
            model_name (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            embedding_dimension (int): Output embedding dimension (default: 512)
            num_classes (int): Number of species classes (default: 30)
            pretrained (bool): Load ImageNet pretrained weights (default: True)
        """
        super(ResNetTriplet, self).__init__()

        # Load ResNet backbone
        resnet_models = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
        }

        if model_name not in resnet_models:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(resnet_models.keys())}")

        self.model = resnet_models[model_name](pretrained=pretrained)

        # Freeze backbone if transfer learning
        if pretrained:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        # Replace final layer
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

        # Embedding projection layer
        self.embedding_fc = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Species classification head
        self.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output embedding and species prediction.

        Args:
            images: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            embedding: Normalized embedding of shape (batch_size, embedding_dimension)
            species_pred: Species logits of shape (batch_size, num_classes)
        """
        h = self.model(images)
        embedding = self.embedding_fc(h)
        species_pred = self.classifier(h)

        return embedding, species_pred


class EfficientNetTriplet(nn.Module):
    """EfficientNet-based triplet network for whale/dolphin re-identification."""

    def __init__(self, embedding_dimension=64, num_classes=30, pretrained=True):
        """
        Args:
            embedding_dimension (int): Output embedding dimension (default: 64)
            num_classes (int): Number of species classes (default: 30)
            pretrained (bool): Load pretrained weights (default: True)
        """
        super(EfficientNetTriplet, self).__init__()

        self.model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_efficientnet_b0',
            pretrained=pretrained
        )

        # Freeze backbone if transfer learning
        if pretrained:
            for name, child in self.model.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        # Replace final layers
        input_features_fc_layer = self.model.classifier.fc.in_features
        self.model.classifier.fc = nn.Linear(input_features_fc_layer, 256, bias=False)

        self.embedding_fc = nn.Linear(256, embedding_dimension, bias=False)
        self.classifier = nn.Linear(embedding_dimension, num_classes)

    def forward(self, images):
        """Forward pass to output embedding and species prediction."""
        h = self.model(images)
        embedding = self.embedding_fc(h)
        embedding_norm = F.normalize(embedding, p=2, dim=1)
        species_pred = self.classifier(embedding_norm)

        return embedding, species_pred
