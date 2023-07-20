from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, img_feats, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.img_feats = img_feats
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["image_id"].split("_")[1]
        image = self.reps[int(img_id)]

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

        question = "question : " + row['question']

        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        if not self.is_test:
            answer = "answer : " + row['answer']
        else:
            answer = "answer : "
        answer = self.tokenizer.encode_plus(
            answer,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            'image': img_feat.squeeze(),
            'pos': bbox.squeeze(),
            'question': {k: v.squeeze() for k, v in question.items()},
            'answer': {k: v.squeeze() for k, v in answer.items()},
        }
        