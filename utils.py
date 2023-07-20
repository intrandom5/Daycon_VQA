import glob
import torch
import pickle
import pandas as pd
from tqdm.auto import tqdm
from dataset import VQADataset, VLT5_Dataset
from torch.utils.data import DataLoader


def prepare_data(df_path, tokenizer, test_mode, shuffle, img_path, bbox_path=None):
    df = pd.read_csv(df_path)

    if bbox_path != None:
        img_feats = []
        bboxes = []
        print("load FRCNN features...")
        img_pkl_files = sorted(glob.glob(img_path+"/*.pkl"))
        for pkl in img_pkl_files:
            with open(pkl, "rb") as f:
                img_feats += pickle.load(f)
        bbox_pkl_files = sorted(glob.glob(bbox_path+"/*.pkl"))
        for pkl in bbox_pkl_files:
            with open(pkl, "rb") as f:
                bboxes += pickle.load(f)
        print("Done!")
    else:
        with open(img_path, "rb") as f:
            img_feats = pickle.load(f)

    if bbox_path is None:
        dataset = VQADataset(df, tokenizer, img_feats, is_test=test_mode)
    else:
        dataset = VLT5_Dataset(df, tokenizer, img_feats, bboxes, test_mode)
    loader = DataLoader(dataset, bathc_size=64, shuffle=shuffle)

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
