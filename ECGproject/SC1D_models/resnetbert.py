import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
import torchvision.models as models

class BertModelForPrompt(nn.Module):
    def __init__(self, num_channels, sequence_length, num_classes, bert_model_name='bert-base-uncased'):
        super(BertModelForPrompt, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)

        self.fc = nn.Linear(self.bert_model.config.hidden_size, num_classes)

        resnet = models.resnet18(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc_resnet = nn.Linear(512,num_classes)

        self.fc_combined = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x, input_text):
        print('x', x.shape)
        # print('input_text', input_text[0].shape)
        # Tokenize input text
        max_length = 37  # You can set this to the desired length
        input_ids_list = [
            self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)['input_ids']
            for text in input_text]

        print(input_ids_list[0])

        input_ids_batch = torch.cat(input_ids_list, dim=0)

        # Forward pass through BERT
        _, pooled_output = self.bert_model(input_ids_batch)

        # Linear layer for classification
        bert_output = self.fc(pooled_output)

        resnet_features = self.resnet_feature_extractor(x)
        resnet_output = self.fc_resnet(resnet_features.view(resnet_features.size(0), -1))

        combined_output = torch.cat([resnet_output, bert_output], dim=1)

        final_output = self.fc_combined(combined_output)

        return final_output




# Load CSV file
# csv_file_path = 'your_file_path.csv'
# df = pd.read_csv(csv_file_path)
#
# # Extract relevant information from the CSV
# prompt_data = df[['exam_id', 'age', 'is_male', 'nn_predicted_age', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'patient_id', 'death', 'timey', 'normal_ecg', 'trace_file']]
# prompt_text = prompt_data.apply(lambda row: ' '.join(map(str, row)), axis=1).tolist()
#
# # Instantiate and use the model
# bert_model = BertModelForPrompt()
# output_bert = bert_model(prompt_text)
# print('Output shape:', output_bert.shape)
