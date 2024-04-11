import torch
import torch.nn as nn
import torchvision.models as models
from pyts.image import GramianAngularField

class ECGResNetModel(nn.Module):
    def __init__(self, num_channels, sequence_length, num_classes):
        super(ECGResNetModel, self).__init__()

        # Gramian Angular Field
        self.gaf_transform = GramianAngularField(method='summation')

        # ResNet model
        resnet = models.resnet18(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer

        # Linear layer for classification
        self.fc_resnet = nn.Linear(512, num_classes)  # ResNet output size is 512

    def forward(self, x):
        # Gramian Angular Field transformation
        print(x.shape)
        reshaped_x = x.view(x.size(0), -1)

        gaf_images = self.gaf_transform.transform(reshaped_x.cpu().detach().numpy())
        print(gaf_images.shape)

        # Convert gaf_images to tensor and move to the device (GPU) if needed
        gaf_images = torch.tensor(gaf_images[:, 0, :, :]).to(x.device)

        # ResNet feature extraction
        resnet_features = self.resnet_feature_extractor(gaf_images)

        # Flatten and apply linear layer for ResNet
        resnet_output = self.fc_resnet(resnet_features.view(resnet_features.size(0), -1))

        return resnet_output

# Example usage
if __name__ == '__main__':
    model = ECGResNetModel(num_channels=12, sequence_length=4096, num_classes=6)
    input_sequence = torch.randn(2, 12, 4096)  # Batch size of 2, 12 channels, sequence length 4096

    output = model(input_sequence)
    print('Output shape:', output.shape)
