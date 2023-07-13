from torch.utils.data import Dataset
from PIL import Image
import torch
import os


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, representation, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.representation = torch.cat(representation, dim=0)
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"].split("_")[1]
        img_rep = self.representation[int(img_id)]

        question = row['question'] # 질문
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if not self.is_test:
            answer = row['answer'] # 답변
            answer = self.tokenizer.encode_plus(
                answer,
                max_length=32,
                padding='max_length',
                truncation=True,
                return_tensors='pt')
            return {
                'image': img_rep,
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'image': img_rep,
                'question': question['input_ids'].squeeze(),
            }
        