import torch.optim as optim
from torch.utils.data import DataLoader
from simple_baseline import SimpleBaseline, heatmap_loss


## YOUR PART ##
# Dataset and DataLoader
# from your_dataset import YourDataset
# dataset = YourDataset()
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

## EXAMPLE WITH DUMMY DATASET ##
from dummy_dataset import DummyPoseDataset
dataset = DummyPoseDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Model
num_keypoints = 17 # Adjust based on your dataset
model = SimpleBaseline(num_keypoints=num_keypoints)
model.train()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10 # Set the number of epochs
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = heatmap_loss(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

# Saving the trained model
# torch.save(model.state_dict(), 'simple_baseline_model.pth')
