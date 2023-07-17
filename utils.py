import torch
import pandas as pd
from tqdm.auto import tqdm
from dataset import VQADataset
from torch.utils.data import DataLoader
from torchvision import transforms


def prepare_data(df_path, img_path, tokenizer, test_mode):
    df = pd.read_csv(df_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = VQADataset(df, tokenizer, transform, img_path, is_test=test_mode)
    shuffle = not test_mode
    loader = DataLoader(dataset, batch_size=64, shuffle=shuffle)

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
            outputs = model(images, question, answer, attention_mask)
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
