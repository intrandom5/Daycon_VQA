from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

        
class VLT5_Dataset(Dataset):
    def __init__(self, df, tokenizer, img_feats, bboxes, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.img_feats = img_feats
        self.bboxes = bboxes
        self.is_test = is_test

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image_id"].split("_")[1]
        img_feat = self.img_feats[int(img_id)]
        bbox = self.bboxes[int(img_id)]

        question = "Q: " + row['question']

        question = self.tokenizer.encode(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            return_tensors="pt"
        )
        if not self.is_test:
            answer = "A: " + row['answer']
        else:
            answer = "A: "
        answer = self.tokenizer.encode(
            answer,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'image': img_feat.squeeze(),
            'pos': bbox.squeeze(),
            'question': question.squeeze(),
            'answer': answer.squeeze(),
        }
        
class Git_Dataset(Dataset):
    def __init__(self, df_path, processor, is_test=True, caption=False):
        super(Git_Dataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.processor = processor
        self.is_test = is_test
        self.caption = caption
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_test:
            img_path = os.path.join("../image/test", row["image_id"] + ".jpg") # train_000000
        else:
            img_path = os.path.join("../image/train", row["image_id"] + ".jpg") # train_000000
        img = Image.open(img_path).convert("RGB")
        return_dict = self.processor(images=img, return_tensors="pt")
        return_dict["pixel_values"] = return_dict["pixel_values"].squeeze()
        
        question = "question : " + row['question']
        if self.caption:
            question = f"info : {row['caption']}, {row['ocr']}. {question}"
            question = question.replace(", nan", "")
                
        if not self.is_test:
            answer = "answer : " + row['answer']
            question = question + " " + answer

        question = self.processor.tokenizer(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=40,
            padding="max_length",
            return_tensors="pt"
        )
            
        return_dict["input_ids"] = question['input_ids'].squeeze()
        return_dict["attention_mask"] = question["attention_mask"].squeeze()
        
        if not self.is_test:
            return_dict["labels"] = return_dict["input_ids"]

        return return_dict
