import os
import yaml
import wandb
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import VQAModel
from utils import prepare_training_data, prepare_test_data, train, inference


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, valid_loader, vocab_size = prepare_training_data(
        args.train_df,
        args.valid_df,
        args.train_img_path
    )

    if args.train_img_path.endswith("pkl"):
        model = VQAModel(vocab_size, contain_resnet=False)
    else:
        model = VQAModel(vocab_size, contain_resnet=True)
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

    for epoch in range(args.epochs):
        train_loss, valid_loss, model_state = train(model, train_loader, valid_loader, optimizer, criterion, device)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        wandb.log({"epoch": epoch+1, "Train Loss": train_loss, "Valid Loss": valid_loss})
        torch.save(model_state, os.path.join(args.model_path, f"epoch{epoch+1}.pt"))

    test_loader, tokenizer = prepare_test_data(args.test_df, args.test_img_path)
    preds = inference(model, test_loader)

    no_pad_output = []
    for pred in preds:
        output = pred[pred != 50257]
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
    parser.add_argument("--epochs", type=int, help="epochs of training.")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--submission_name", type=str, help="name of submission file.")
    parser.add_argument("--logger", type=str, help="select logger. ['wandb', 'none']")
    parser.add_argument("--log_name", type=str, help="name of experiment. only used when using wandb.")
    parser.add_argument("--log_note", type=str, help="note of experiment. only used when using wandb.")
    args = parser.parse_args()

    main(args)
