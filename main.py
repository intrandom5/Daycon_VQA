import torch
import argparse
from vlt5_utils import vlt5_process


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    assert args.model_type in ["vlt5"]

    if args.model_type == "vlt5":
        vlt5_process(args)

    # print("load test FRCNN features...")
    # img_feat_pkls = sorted(glob.glob(args.test_img_path+"/*.pkl"))
    # test_img_feats = load_pickles(img_feat_pkls)
    # if args.model_type == "vlt5":
    #     bboxes = sorted(glob.glob(args.test_bbox_path+"/*.pkl"))
    #     test_bboxes = load_pickles(bboxes)
    #     test_loader = prepare_data(
    #         args.test_df, 
    #         tokenizer=tokenizer,
    #         test_mode=True,
    #         shuffle=False,
    #         img_feats=test_img_feats,
    #         bboxes=test_bboxes
    #     )
    #     print("Done!")
    #     preds = inference_vlt5(model, test_loader, device)


    # no_pad_output = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # sample_submission = pd.read_csv('../sample_submission.csv')
    # sample_submission["answer"] = no_pad_output
    # sample_submission.to_csv(args.submission_name, index=False)

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
    parser.add_argument("--model_type", type=str, help="type of pretrained language model to use. ['vlt5']")
    parser.add_argument("--epochs", type=int, help="epochs of training.")
    parser.add_argument("--learning_rate", type=float, help="learning rate")
    parser.add_argument("--submission_name", type=str, help="name of submission file.")
    parser.add_argument("--logger", type=str, help="select logger. ['wandb', 'none']")
    parser.add_argument("--log_name", type=str, help="name of experiment. only used when using wandb.")
    parser.add_argument("--log_note", type=str, help="note of experiment. only used when using wandb.")
    args = parser.parse_args()

    main(args)
