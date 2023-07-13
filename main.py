import yaml
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from model import VQAModel
from utils import prepare_training_data, prepare_test_data, train, inference


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, valid_loader, vocab_size = prepare_training_data(
        config["train_df"],
        config["valid_df"],
        config["train_img_path"]
    )

    model = VQAModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        train_loss, valid_loss = train(model, train_loader, valid_loader, optimizer, criterion, device)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    test_loader, tokenizer = prepare_test_data(config["test_df"], config["test_img_path"])
    preds = inference(model, test_loader)

    no_pad_output = []
    for pred in preds:
        output = pred[pred != 50257]
        no_pad_output.append(tokenizer.decode(output).strip())

    sample_submission = pd.read_csv('sample_submission.csv')
    sample_submission["answer"] = no_pad_output
    sample_submission.to_csv(config["submission_name"], index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config)