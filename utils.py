import torch
import pandas as pd
from tqdm.auto import tqdm
from dataset import VQADataset
from transformers import GPT2Tokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms


def prepare_training_data(train_df_path, valid_df_path, train_img_path, model_type):
    train_df = pd.read_csv(train_df_path)
    valid_df = pd.read_csv(valid_df_path)

    if model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif model_type == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    vocab_size = len(tokenizer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = VQADataset(train_df, tokenizer, transform, train_img_path, is_test=False)
    valid_dataset = VQADataset(valid_df, tokenizer, transform, train_img_path, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, vocab_size

def prepare_test_data(test_df_path, test_img_path):
    test_df = pd.read_csv(test_df_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = VQADataset(test_df, tokenizer, transform, test_img_path, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, tokenizer

def train(model, train_loader, valid_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        answer = data['answer'].to(device)

        optimizer.zero_grad()

        outputs = model(images, question)

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

        with torch.no_grad():
            outputs = model(images, question)
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
            question = data['question'].to(device)

            outputs = model(images, question) # [batch, sequence, vocab]

            _, pred = torch.max(outputs, dim=2) # values, indices = _, pred
            preds.extend(pred.cpu().numpy())

    return preds
