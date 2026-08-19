import torch
import torch.nn as nn

class RoadConditionCNN(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self._features = nn.Sequential(
            self._conv_block(3,16),
            self._conv_block(16,32),
            self._conv_block(32,64),
            self._conv_block(64, 128),
            self._conv_block(128, 128)
        )

        self.classifier = nn.Conv2d(128, num_classes, 1)

    
    def _conv_block(self, input_channels, output_channels) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding = 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        process = self._features(image)
        classification = self.classifier(process)
        return classification

if __name__ == "__main__":
    model = RoadConditionCNN(num_classes=4)
    dummy = torch.randn(2,3,128,128)
    output = model(dummy)

    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")   

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")