from transformers import TrainingArguments, Trainer, GitProcessor
import pandas as pd

from dataset import Git_Dataset
from model import get_git


def git_process(args):
    train_df = pd.read_csv(args.train_df)
    valid_df = pd.read_csv(args.valid_df)
    
    processor = GitProcessor.from_pretrained("microsoft/git-base-coco")
    train_dataset = Git_Dataset(train_df, processor, is_test=False, caption=args.caption)
    valid_dataset = Git_Dataset(valid_df, processor, is_test=False, caption=args.caption)

    model = get_git()

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        save_steps=args.eval_step,
        output_dir=args.model_path,
        learning_rate=args.learning_rate,
        fp16=True,
        report_to=args.logger,
        run_name=args.log_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )