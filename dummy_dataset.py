from torch import randn
from torch.utils.data import Dataset

class DummyPoseDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(256, 192), num_keypoints=17, heatmap_size=(64, 48)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a dummy image (channels, height, width)
        image = randn(3, *self.image_size)  # Removed batch size dimension
        
        # Generate dummy heatmaps (num_keypoints, height, width)
        heatmaps = randn(self.num_keypoints, *self.heatmap_size)  # Removed batch size dimension
        
        return image, heatmaps
