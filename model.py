import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import GPT2Model


class scaled_dot_product_attn(nn.Module):
    def __init__(self, d_k): # d_k : dimension of key.
        super(scaled_dot_product_attn, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value):
        print(query.shape, key.shape, value.shape)
        x = torch.matmul(query, key.transpose(2, 3))
        x /= np.sqrt(self.d_k)
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, value)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, self.d_head * num_heads)
        self.key_proj = nn.Linear(dim, self.d_head * num_heads)
        self.value_proj = nn.Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = scaled_dot_product_attn(dim)

    def forward(self, query, key, value):
        batch_size = value.size(0)
        query = self.query_proj(query)
        query = query.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key)
        key = key.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value)
        value = value.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        context = self.scaled_dot_attn(query, key, value)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context


class VQAModel(nn.Module):
    def __init__(self, vocab_size, contain_resnet: bool = True):
        super(VQAModel, self).__init__()
        self.vocab_size = vocab_size
        self.contain_resnet = contain_resnet

        if contain_resnet == True:
            resnet = models.resnet50(pretrained=True)
            self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        self.res_lin = nn.Linear(2048, 768)

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영

        self.attention = MultiHeadAttention(dim=768, num_heads=8)

        # combined_features_size = 1000 + self.gpt2.config.hidden_size # resnet 출력 차원 + gpt2 출력 차원
        # self.classifier = nn.Linear(combined_features_size, vocab_size)
        self.classifier = nn.Linear(768, vocab_size)

    def forward(self, images, question):
        if self.contain_resnet:
            images = self.resnet(images).squeeze()

        image_features = self.res_lin(images)

        outputs = self.gpt2(question)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        # combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        combined = self.attention(output_features, image_features, image_features)
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    