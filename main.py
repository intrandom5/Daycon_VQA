import os
import yaml
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import VQAModel
from transformers import GPT2Tokenizer, AutoTokenizer
from utils import prepare_data, train, inference


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert args.model_type in ["gpt2", "bart"]

    if args.model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif args.model_type == "bart":
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    vocab_size = len(tokenizer)

    train_loader = prepare_data(
        args.train_df,
        args.train_img_path,
        tokenizer=tokenizer,
        test_mode=False
    )
    valid_loader = prepare_data(
        args.valid_df,
        args.train_img_path,
        tokenizer=tokenizer,
        test_mode=True
    )

    if args.train_img_path.endswith("pkl"):
        model = VQAModel(vocab_size, False, args.model_type)
    else:
        model = VQAModel(vocab_size, True, args.model_type)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    model.to(device)

    for epoch in range(args.epochs):
        train_loss, valid_loss, model_state = train(
            model, train_loader, valid_loader, optimizer, criterion, device
        )
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        torch.save(model_state, os.path.join(args.model_path, f"epoch{epoch+1}.pt"))

    test_loader = prepare_data(
        args.test_df, 
        args.test_img_path, 
        tokenizer=tokenizer,
        test_mode=True
    )
    preds = inference(model, test_loader, device)

    no_pad_output = []
    for pred in preds:
        output = pred[pred != tokenizer.pad_token_id]
        no_pad_output.append(tokenizer.decode(output).strip())

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
    parser.add_argument("--model_path", type=str, help="path of model to save.")
    parser.add_argument("--model_type", type=str, help="type of pretrained language model to use. ['gpt2', 'bart']")
    parser.add_argument("--epochs", type=int, help="epochs of training.")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--submission_name", type=str, help="name of submission file.")
    args = parser.parse_args()

    main(args)
