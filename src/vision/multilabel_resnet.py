import torch
import torch.nn as nn
from torchvision.models import resnet18


class MultilabelResNet18(nn.Module):
    def __init__(self):
        """Initialize network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one
        Note: Consider which activation function to use
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Download pretrained resnet using pytorch's API (Hint: see the import statements)
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None
        self.activation = None

        ############################################################################
        # Student code begin
        ############################################################################

        resnet = resnet18(pretrained=True)
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():  
            param.requires_grad = True
        for param in resnet.layer3.parameters():  
            param.requires_grad = True

        in_feats = resnet.fc.in_features
        self.activation = nn.Sigmoid()
        self.fc_layers = nn.Linear(in_feats, 7)
        self.loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net, duplicating grayscale channel to 3-channel.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images
        Returns:
            y: tensor of shape (N,num_classes) representing the output (raw scores) of the net
                Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
        ############################################################################
        # Student code begin
        ############################################################################

        feats = self.conv_layers(x)
        feats = torch.flatten(feats, start_dim=1)
        model_output = self.fc_layers(feats)

        ############################################################################
        # Student code end
        ############################################################################
        return model_output
