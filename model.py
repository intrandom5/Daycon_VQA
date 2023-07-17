import torch
import torch.nn as nn
import torchvision.models as models
from transformers import lmModel, BartModel


class VQAModel(nn.Module):
    def __init__(self, vocab_size, contain_resnet: bool = True, model_type: str = "gpt2"):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size
        self.contain_resnet = contain_resnet
        self.model_type = model_type
        assert model_type in ["gpt2", "bart", "bert"]

        if contain_resnet:
            self.resnet = models.resnet50(pretrained=True)

        if model_type == "gpt2":
            self.lm = lmModel.from_pretrained('lm')
            self.lm.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영
        elif model_type == "bart":
            self.lm = BartModel.from_pretrained("facebook/bart-base")

        combined_features_size = 1000 + self.lm.config.hidden_size
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question, answer, attention_mask):
        if self.contain_resnet:
            images = self.resnet(images)
        image_features = images.view(images.size(0),-1)

        if self.model_type == "gpt2":
            outputs = self.lm(question)
        elif self.model_type == "bart":
            outputs = self.lm(question, attention_mask=attention_mask, decoder_input_ids=answer)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    