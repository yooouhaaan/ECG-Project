import torch
import torch.nn as nn
import torchvision.models as models
from SC1D_models.resnet1d import resnet1d_18


class CustomModel(nn.Module):
    def __init__(self, input_dim, num_classes=6, resnet_feature_size=256):
        super(CustomModel, self).__init__()

        # Define your ResNet model (replace with your actual ResNet model)
        # self.resnet = models.resnet18(pretrained=True)
        self.resnet = resnet1d_18(input_dim, num_classes)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Linear layers for each class feature
        self.class_features = nn.ModuleList([nn.Linear(resnet_feature_size, 1) for _ in range(num_classes)])

        self.fc_class = nn.Linear(768, 256)

        # Linear layer for the final output
        self.fc_final = nn.Linear(512, num_classes)

    def forward(self, x, class_features):
        # print('x', x.shape) #256,12,4096
        # print('class_features', class_features.shape) #6, 768
        # Pass the first input through ResNet
        resnet_output = self.resnet_feature_extractor(x)
        # print('resnet_output', resnet_output.shape) #256, 512, 256

        # Average along the channel dimension
        averaged_features = torch.mean(resnet_output, dim=1)
        # print('average_features', averaged_features.shape) # 256,256

        # Pass the second input (class features) through linear layers
        class_features = class_features.to(resnet_output.device)

        # Average the class features to get 1 channel
        averaged_features_class = torch.mean(class_features, dim=0, keepdim=True)
        # print('averaged_features_class', averaged_features_class.shape) #1, 768

        averaged_features_class = self.fc_class(averaged_features_class)
        # print('averaged_features_class2', averaged_features_class.shape) # 1, 256
        # feature_class = nn.Linear(averaged_features_class.shape[1], averaged_features.shape[1])
        #Expand class features to match the shape of averaged features

        averaged_features_class_expanded = averaged_features_class.expand(averaged_features.shape[0], -1)
        # print('averaged_features_class_expanded', averaged_features_class_expanded.shape)  # 256, 256

        # Multiply the averaged ResNet features with averaged class features
        weighted_resnet_features = averaged_features * averaged_features_class_expanded
        # print('weighted_resnet_features', weighted_resnet_features.shape) #256, 256

        # class_weights = [class_feature(class_features[:, i]) for i, class_feature in enumerate(self.class_features)]
        # class_weights = [class_feature(averaged_features) for class_feature in self.class_features]
        # # class_weights = [class_feature(class_features[i, :].unsqueeze(1)) for i, class_feature in enumerate(self.class_features)]
        # # print('class weights', class_weights)
        # class_weights = torch.cat(class_weights, dim=1)
        # print('class weights', class_weights.shape) #256, 6
        # class_features = class_features.to(resnet_output.device)
        # class_weights = [class_feature(class_features[i, :]) for i, class_feature in
        #                  enumerate(self.class_features)]
        # class_weights = [class_feature(averaged_features[:, i]) for i,class_feature in self.class_features]
        # class_weights = [class_feature(class_features[i, :]) for i, class_feature in enumerate(self.class_features)]
        # # class_weights = torch.cat(class_weights, dim=1)
        # class_weights = torch.cat(class_weights)
        # print('class weights', class_weights.shape)  # 1536, 1

        # Concatenate the averaged ResNet features and class features
        concatenated_features = torch.cat([averaged_features, weighted_resnet_features], dim=1)
        # print('concatenated_features', concatenated_features.shape) #256, 512

        # Pass through the final linear layer
        final_output = self.fc_final(concatenated_features)

        # Pass through the final linear layer
        # final_output = self.fc_final(weighted_resnet_features)

        return final_output


# Example usage
if __name__ == '__main__':
    # Define the model
    model = CustomModel(num_classes=6)

    # Create dummy input
    input_data = torch.randn(2, 12, 256, 256)  # Batch size of 2, 12 channels, spatial dimensions 256x256
    class_features = torch.load('path/to/your/class_features.pt')  # Load your class features from a .pt file

    # Forward pass
    output = model(input_data, class_features)
    print('Output shape:', output.shape)
