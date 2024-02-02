from typing import Optional

from pydantic import BaseModel
import torch
import torch.nn as nn

class NetworkConfig(BaseModel):
    num_classes: int = 3
    input_channels: int = 1
    
class Network(nn.Module):
    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        super().__init__()
        
        if config is None:
            config = NetworkConfig()
        self.config = config
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(self.config.input_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16 * 16, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.config.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)

        return x
    
if __name__=="__main__":
    # Example usage:
    config = NetworkConfig(num_classes=3, input_channels=1)
    model = Network(config)
    input_tensor = torch.randn((1, 1, 64, 64, 64))  # Adjust input size as needed
    output = model(input_tensor)
    print("Output shape:", output.shape)