import torch.nn as nn
import torchvision.models as models

# Model Definition
class SimpleBaseline(nn.Module):
    def __init__(self, num_keypoints):
        super(SimpleBaseline, self).__init__()
        # Use ResNet, but remove the average pooling and fully connected layer
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove last two layers

        # Additional layers to output heatmaps
        self.heatmap_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, num_keypoints, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.resnet(x)  # Pass through the modified ResNet
        x = self.heatmap_layers(x)  # Generate heatmaps
        return x


# Loss Function
def heatmap_loss(predicted_heatmap, ground_truth):
    return nn.MSELoss()(predicted_heatmap, ground_truth)
