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

def train_vlt5(model, train_loader, valid_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        data['image'] = data['image'].to('cuda')
        data['pos'] = data['pos'].to('cuda')
        data['question'] = {k: v.to('cuda') for k, v in data['question'].items()}
        data['answer'] = {k: v.to('cuda') for k, v in data['answer'].items()}
        label = data['answer']['input_ids']
        data['answer']['input_ids'] = model.T5._shift_right(data['answer']['input_ids'])

        optimizer.zero_grad()

        outputs = model(data['question'], data['answer'], data['image'], data['pos'])

        # output: [batch, sequence, vocab], answer : [batch, sequence]
        loss = criterion(outputs.view(-1, outputs.size(-1)), label.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(train_loader)

    model.eval()
    total_loss = 0
    for data in tqdm(valid_loader, total=len(valid_loader)):
        data['image'] = data['image'].to('cuda')
        data['pos'] = data['pos'].to('cuda')
        data['question'] = {k: v.to('cuda') for k, v in data['question'].items()}
        data['answer'] = {k: v.to('cuda') for k, v in data['answer'].items()}
        label = data['answer']['input_ids']
        data['answer']['input_ids'] = model.T5._shift_right(data['answer']['input_ids'])
        with torch.no_grad():
            outputs = model(data['question'], data['answer'], data['image'], data['pos'])

        # output: [batch, sequence, vocab], answer : [batch, sequence]
        loss = criterion(outputs.view(-1, outputs.size(-1)), label.view(-1))
        total_loss += loss.item()
    
    valid_loss = total_loss / len(valid_loader)

    return train_loss, valid_loss, model.state_dict()

def inference_vlt5(model, loader, device, max_length=32):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            data['image'] = data['image'].to(device)
            data['pos'] = data['pos'].to('cuda')
            data['question'] = {k: v.to('cuda') for k, v in data['question'].items()}
            data['answer'] = {k: v.to('cuda') for k, v in data['answer'].items()}
            data['answer']['input_ids'] = model.T5._shift_right(data['answer']['input_ids'])

            batch_size = data['image'].shape[0]

            for _ in range(max_length):
                with torch.no_grad():
                    outputs = model(data['question'], data['answer'], data['image'], data['pos'])


                next_token_id = torch.argmax(outputs, dim=-1)[:, -1]
                next_token_id = next_token_id.view(-1, 1)
                data['answer']['input_ids'] = torch.cat(
                    [data['answer']['input_ids'], next_token_id], dim=-1
                )
                data['answer']['attention_mask'] = torch.cat(
                    [data['answer']['attention_mask'], torch.ones(batch_size, 1).to('cuda')], dim=-1
                )
            
            preds.extend(data['answer']['input_ids'].cpu().numpy())

    return preds