from torch.utils.data import Dataset
from PIL import Image
import pickle
import torch
import glob
import os


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, transforms, img_path, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.img_path = img_path
        if img_path.endswith("pkl"):
            with open(img_path, "rb") as f:
                reps = pickle.load(f)
            self.reps = torch.cat(reps, dim=0)
            
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.img_path.endswith("pkl"):
            img_id = row["image_id"].split("_")[1]
            image = self.reps[int(img_id)]
        else:
            img_name = os.path.join(self.img_path, row['image_id'] + '.jpg') # 이미지
            image = Image.open(img_name).convert('RGB')
            image = self.transform(image)

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
                'image': image,
                'question': question['input_ids'].squeeze(),
                'answer': answer['input_ids'].squeeze()
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
            }
        