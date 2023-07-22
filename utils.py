import glob
import torch
import pickle
import pandas as pd
from tqdm.auto import tqdm
from dataset import VQADataset, VLT5_Dataset
from torch.utils.data import DataLoader


def load_pickles(files):
    results = []
    for file in files:
        with open(file, "rb") as f:
            results += pickle.load(f)

    return results


def prepare_data(df_path, tokenizer, test_mode, shuffle, img_feats, bboxes=None):
    df = pd.read_csv(df_path)

    if bboxes is None:
        dataset = VQADataset(df, tokenizer, img_feats, is_test=test_mode)
    else:
        dataset = VLT5_Dataset(df, tokenizer, img_feats, bboxes, test_mode)
    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return loader

def train(model, train_loader, valid_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        answer = data['answer'].to(device)
        attention_mask = data['attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(images, question, answer, attention_mask)

        # output: [batch, sequence, vocab], answer : [batch, sequence]
        loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(train_loader)

    model.eval()
    total_loss = 0
    for data in tqdm(valid_loader, total=len(valid_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        answer = data['answer'].to(device)
        attention_mask = data['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(images, question, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), answer.view(-1))
        total_loss += loss.item()
    
    valid_loss = total_loss / len(valid_loader)

    return train_loss, valid_loss, model.state_dict()

def inference(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            images = data['image'].to(device)
            attention_mask = data['attention_mask'].to(device)
            question = data['question'].to(device)

            outputs = model(images, question, attention_mask)

            _, pred = torch.max(outputs, dim=2) # values, indices = _, pred
            preds.extend(pred.cpu().numpy())

    return preds

def train_vlt5(model, train_loader, valid_loader, optimizer, pad_token_id, device):
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        data = {k: v.to(device) for k, v in data.items()}

        optimizer.zero_grad()

        output = model(
            input_ids=data["question"],
            vis_inputs=(data["image"], data["pos"]),
            labels=data["answer"],
            return_dict=True
        )
        lm_labels = data["answer"]
        lm_mask = (lm_labels != pad_token_id).float()
        B, L = lm_labels.size()
        loss = output["loss"]
        loss = loss.view(B, L)*lm_mask
        loss = loss.sum(dim=1)/lm_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()
        total_loss += loss

        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(train_loader)

    model.eval()
    total_loss = 0
    for data in tqdm(valid_loader, total=len(valid_loader)):
        data = {k: v.to(device) for k, v in data.items()}
        with torch.no_grad():
            output = model(
                input_ids=data["question"],
                vis_inputs=(data["image"], data["pos"]),
                labels=data["answer"],
                return_dict=True
            )
        lm_labels = data["answer"]
        lm_mask = (lm_labels != pad_token_id).float()
        B, L = lm_labels.size()
        loss = output["loss"]
        loss = loss.view(B, L)*lm_mask
        loss = loss.sum(dim=1)/lm_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()
        total_loss += loss
    
    valid_loss = total_loss / len(valid_loader)

    return train_loss, valid_loss, model.state_dict()

def inference_vlt5(model, loader, device):
    model.eval()
    preds = []
    for data in tqdm(loader):
        data = {k: v.to(device) for k, v in data.items()}
        output = model.generate(
            data["question"],
            vis_inputs=(data["image"], data["pos"])
        )
        preds.append(output)
    preds = torch.cat(preds, dim=0)

    return preds