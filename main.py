import os
import yaml
import glob
import wandb
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import BaseVQAModel, get_vlt5
from transformers import GPT2Tokenizer, AutoTokenizer, T5Tokenizer
from utils import load_pickles, prepare_data, train, inference, train_vlt5, inference_vlt5


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert args.model_type in ["gpt2", "bart", "vlt5"]

    if args.model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif args.model_type == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    elif args.model_type == "vlt5":
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    vocab_size = len(tokenizer)

    # open pickle files
    print("load FRCNN features...")
    img_feat_pkls = sorted(glob.glob(args.train_img_path+"/*.pkl"))
    train_img_feats = load_pickles(img_feat_pkls)

    bboxes = sorted(glob.glob(args.train_bbox_path+"/*.pkl"))
    train_bboxes = load_pickles(bboxes)
    print("Done!")

    # define Dataloader
    if args.model_type == "vlt5":
        train_loader = prepare_data(
            args.train_df,
            tokenizer,
            test_mode=False,
            shuffle=True,
            img_feats=train_img_feats,
            bboxes=train_bboxes
        )
        valid_loader = prepare_data(
            args.valid_df,
            tokenizer,
            test_mode=False,
            shuffle=False,
            img_feats=train_img_feats,
            bboxes=train_bboxes
        )
    else:
        train_loader = prepare_data(
            args.train_df,
            tokenizer,
            test_mode=False,
            shuffle=True,
            img_path=args.train_img_path
        )
        valid_loader = prepare_data(
            args.valid_df,
            tokenizer,
            test_mode=False,
            shuffle=False,
            img_feats=train_img_feats
        )

    # Define Model
    if args.model_type == "gpt2":
        if args.train_img_path.endswith("pkl"):
            model = BaseVQAModel(vocab_size, False, args.model_type)
        else:
            model = BaseVQAModel(vocab_size, True, args.model_type)
    else:
        model = get_vlt5("google/flan-t5-base")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    model.to(device)
    if args.logger == "wandb":
        run = wandb.init(
            project="DayCon_VQA", 
            entity="intrandom5", 
            name=args.log_name, 
            notes=args.log_note
        )

    # Training Start!
    for epoch in range(args.epochs):
        if args.model_type == "vlt5":
            train_loss, valid_loss, model_state = train_vlt5(
                model, train_loader, valid_loader, optimizer, tokenizer.pad_token_id, device
            )
        else:
            train_loss, valid_loss, model_state = train(
                model, train_loader, valid_loader, optimizer, criterion, device
            )
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        wandb.log({"epoch": epoch+1, "Train Loss": train_loss, "Valid Loss": valid_loss})
        torch.save(model_state, os.path.join(args.model_path, f"epoch{epoch+1}.pt"))

    print("load test FRCNN features...")
    img_feat_pkls = sorted(glob.glob(args.test_img_path+"/*.pkl"))
    test_img_feats = load_pickles(img_feat_pkls)
    if args.test_bbox_path != None:
        bboxes = sorted(glob.glob(args.test_bbox_path+"/*.pkl"))
        test_bboxes = load_pickles(bboxes)
        test_loader = prepare_data(
            args.test_df, 
            tokenizer=tokenizer,
            test_mode=True,
            shuffle=False,
            img_feats=test_img_feats,
            bboxes=test_bboxes
        )
        print("Done!")
        preds = inference_vlt5(model, test_loader, device)
    else:
        test_loader = prepare_data(
            args.test_df, 
            tokenizer=tokenizer,
            test_mode=True,
            shuffle=False,
            img_feats=test_img_feats
        )
        preds = inference(model, test_loader, device)

    no_pad_output = tokenizer.batch_decode(preds, skip_special_tokens=True)

    sample_submission = pd.read_csv('../sample_submission.csv')
    sample_submission["answer"] = no_pad_output
    sample_submission.to_csv(args.submission_name, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_df", type=str, help="path of train csv file.")
    parser.add_argument("--valid_df", type=str, help="path of valid csv file.")
    parser.add_argument("--test_df", type=str, help="path of test csv file.")
    parser.add_argument("--train_img_path", type=str, help="path of train image features in '.pkl' format or folder contains image.")
    parser.add_argument("--test_img_path", type=str, help="path of test image features in '.pkl' format or folder contains image.")
    parser.add_argument("--train_bbox_path", type=str, default="none", help="path of train bbox features.")
    parser.add_argument("--test_bbox_path", type=str, default="none", help="path of test bbox features.")
    parser.add_argument("--model_path", type=str, help="path of model to save.")
    parser.add_argument("--model_type", type=str, help="type of pretrained language model to use. ['gpt2', 'bart', 'vlt5']")
    parser.add_argument("--epochs", type=int, help="epochs of training.")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--submission_name", type=str, help="name of submission file.")
    parser.add_argument("--logger", type=str, help="select logger. ['wandb', 'none']")
    parser.add_argument("--log_name", type=str, help="name of experiment. only used when using wandb.")
    parser.add_argument("--log_note", type=str, help="note of experiment. only used when using wandb.")
    args = parser.parse_args()

    main(args)
