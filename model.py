import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2Model, BartModel, T5ForConditionalGeneration 


class BaseVQAModel(nn.Module):
    def __init__(self, vocab_size, contain_resnet: bool = True, model_type: str = "gpt2"):
        super(BaseVQAModel, self).__init__()
        self.vocab_size = vocab_size
        self.contain_resnet = contain_resnet
        self.model_type = model_type
        assert model_type in ["gpt2", "bart", "bert"]

        if contain_resnet:
            self.resnet = models.resnet50(pretrained=True)

        if model_type == "gpt2":
            self.lm = GPT2Model.from_pretrained('gpt2')
            self.lm.resize_token_embeddings(vocab_size) # 추가한 [PAD] 토큰 반영
        elif model_type == "bart":
            self.lm = BartModel.from_pretrained("facebook/bart-base")

        combined_features_size = 1024 + self.lm.config.hidden_size
        self.classifier = nn.Linear(combined_features_size, vocab_size)

    def forward(self, images, question, attention_mask, answer=None):
        if self.contain_resnet:
            images = self.resnet(images)
        image_features = images.view(images.size(0),-1)

        if self.model_type == "gpt2":
            outputs = self.lm(question)
        elif self.model_type == "bart":
            outputs = self.lm(input_ids=question, attention_mask=attention_mask, decoder_input_ids=answer)
        output_features = outputs.last_hidden_state # [batch, sequence, hidden]

        image_features = image_features.unsqueeze(1).expand(-1, output_features.size(1),-1) # [batch, sequence, 1000]

        combined = torch.cat([image_features, output_features], dim=-1) # [batch, sequence, 1000+hidden]
        output = self.classifier(combined) # [batch, vocab_size]
        return output
    

class VLT5(nn.Module):
    def __init__(self, N):
        super(VLT5, self).__init__()
        self.N = N
        self.T5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        self.feat_embedding = nn.Linear(2048, self.T5.config.hidden_size)
        self.pos_embedding = nn.Linear(4, self.T5.config.hidden_size)
        self.obj_embedding = nn.Linear(self.N, self.T5.config.hidden_size)
        self.device = "cpu"
    
    def set_device(self, device):
        self.to(device)
        self.device = device
        
    def forward(self, question, answer, vis_feats, pos, obj=None, vis_mask=None):
        text_emb = self.T5.encoder.embed_tokens(question["input_ids"]) # [B, n, n_dim]
        
        vis_emb = self.feat_embedding(vis_feats) # [B, N, n_dim]
        pos_emb = self.pos_embedding(pos) # [B, N, n_dim]
        if obj == None:
            obj = torch.arange(self.N, device=self.device)
            obj = obj.unsqueeze(0)
        obj_emb = self.obj_embedding(obj.float())
        img_emb = vis_emb + pos_emb + obj_emb
        
        x = torch.cat([text_emb, img_emb], dim=1) # [B, n+M, n_dim]
        for block in self.T5.encoder.block:
            x, _ = block(x)
            
        if vis_mask == None:
            vis_mask = torch.ones(size=(vis_feats.shape[0], self.N), device=self.device)
        encoder_mask = torch.cat([question["attention_mask"], vis_mask], dim=1)
        
        # decoding
        output = self.T5.decoder(
            input_ids=answer["input_ids"],
            attention_mask=answer["attention_mask"],
            encoder_hidden_states=x,
            encoder_attention_mask=encoder_mask,
        )
        
        return output.last_hidden_state
    