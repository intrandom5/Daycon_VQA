import os
import glob
import wandb
import torch
import pickle
import pandas as pd
from tqdm.auto import tqdm

import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from dataset import VLT5_Dataset, Git_Dataset
from model import get_vlt5


def load_pickles(files):
    results = []
    for file in files:
        with open(file, "rb") as f:
            results += pickle.load(f)

    return results

def prepare_vlt5_data(df_path, tokenizer, test_mode, shuffle, img_feats, bboxes=None):
    df = pd.read_csv(df_path)

    dataset = VLT5_Dataset(df, tokenizer, img_feats, bboxes, test_mode)
    loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

    return loader

def train_vlt5(
        model, train_loader, valid_loader, optimizer, pad_token_id, device, epochs, model_path
        ):
    step = 0
    for epoch in range(epochs):
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
            wandb.log({"Trainging step": step, "Train Loss": loss})

            loss.backward()
            optimizer.step()
            step += 1

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
        wandb.log({"Valid Loss": valid_loss})
        torch.save(model.state_dict(), os.path.join(model_path, f"epoch{epoch+1}.pt"))

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

    return preds

def vlt5_process(args):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    # open pickle files
    print("load FRCNN features...")
    img_feat_pkls = sorted(glob.glob(args.train_img_path+"/*.pkl"))
    train_img_feats = load_pickles(img_feat_pkls)

    bboxes = sorted(glob.glob(args.train_bbox_path+"/*.pkl"))
    train_bboxes = load_pickles(bboxes)
    print("Done!")
    train_df = pd.read_csv(args.train_df)
    valid_df = pd.read_csv(args.valid_df)
    train_dataset = VLT5_Dataset(train_df, tokenizer, train_img_feats, train_bboxes)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataset = VLT5_Dataset(valid_df, tokenizer, train_img_feats, train_bboxes)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    model = get_vlt5("google/flan-t5-base")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    model.to(args.device)
    if args.logger == "wandb":
        run = wandb.init(
            project="DayCon_VQA", 
            entity="intrandom5", 
            name=args.log_name, 
            notes=args.log_note
        )

    train_vlt5(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        tokenizer.pad_token_id, 
        args.device,
        args.epochs,
        args.model_path
    )