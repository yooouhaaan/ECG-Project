import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),  # Adjust activation function as needed
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class SharedDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(SharedDecoder, self).__init__()

        self.fc_mean = nn.Linear(input_size, output_size)
        self.fc_var = nn.Linear(input_size, output_size)

    def forward(self, mean, var):
        x = torch.cat([mean, var], dim=1)
        # output_mean = self.fc_mean(x)
        # output_var = self.fc_var(x)
        return x

class MulticlassClassification(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassClassification, self).__init__()

        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        output = self.fc(x)
        return output

class AutoencoderMulticlassModel(nn.Module):
    def __init__(self, num_leads, input_size, hidden_size, num_classes=6):
        super(AutoencoderMulticlassModel, self).__init__()

        self.autoencoders = nn.ModuleList([Autoencoder(input_size, hidden_size) for _ in range(num_leads)])
        self.shared_decoder = SharedDecoder(num_leads * hidden_size, 2 * hidden_size)
        self.classification_model = MulticlassClassification(2 * hidden_size, num_classes)

    def forward(self, x):
        lead_encodings = []
        for i, autoencoder in enumerate(self.autoencoders):
            encoded, _ = autoencoder(x[:, :, i])
            lead_encodings.append(encoded)

        mean_concatenated = torch.cat([torch.mean(enc, dim=1, keepdim=True) for enc in lead_encodings], dim=1)
        var_concatenated = torch.cat([torch.var(enc, dim=1, keepdim=True) for enc in lead_encodings], dim=1)

        print('mean shape:', mean_concatenated.shape)
        print('variance shape:', var_concatenated.shape)

        dec_out = self.shared_decoder(mean_concatenated, var_concatenated)
        print('shared decoder output shape:', dec_out.shape)

        classification_output = self.classification_model(dec_out)
        print('classification output shape:', classification_output.shape)

        return classification_output

# Example usage
if __name__ == '__main__':
    num_leads = 12
    input_size = 4096  # Adjust based on your input size
    hidden_size = 128  # Adjust based on your desired hidden size
    decoder_input_size = 2 * hidden_size  # Concatenated mean and variance
    num_classes = 10  # Adjust based on your number of classes

    model = AutoencoderMulticlassModel(num_leads, input_size, hidden_size, decoder_input_size, num_classes)
    input_data = torch.randn(2, num_leads, input_size)  # Batch size of 2, 12 leads, input size 4096

    output = model(input_data)
    print('Output shape:', output.shape)
