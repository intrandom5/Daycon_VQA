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
                self.reps = pickle.load(f)
            
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
            image = self.transforms(image)

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
                'question_attention_mask': question['attention_mask'].squeeze(),
                'answer': answer['input_ids'].squeeze(),
                'answer_attention_mask': answer['attention_mask'].squeeze()
            }
        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
                'attention_mask': question['attention_mask'].squeeze(),
            }
        
        
class VLT5_Dataset(Dataset):
    def __init__(self, df, tokenizer, transforms, img_feat_path, bbox_path, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.is_test = is_test

        self.img_feats = []
        self.bboxes = []

        print("load FRCNN features...")
        img_pkl_files = sorted(glob.glob(img_feat_path+"/*.pkl"))
        for pkl in img_pkl_files:
            with open(pkl, "rb") as f:
                self.img_feats += pickle.load(f)
        bbox_pkl_files = sorted(glob.glob(bbox_path+"/*.pkl"))
        for pkl in bbox_pkl_files:
            with open(pkl, "rb") as f:
                self.bboxes += pickle.load(f)
        print("Done!")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = self.img_feats[idx]
        bbox = self.bboxes[idx]

        question = self.tokenizer(
            row['question'],
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        if not self.is_test:
            answer = self.tokenizer(
                row['answer'],
                max_length=32,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                'image': img_feat,
                'pos': bbox,
                'question': question,
                'answer': answer
            }
        else:
            return {
                'image': img_feat,
                'pos': bbox,
                'question': question
            }
        