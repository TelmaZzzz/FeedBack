import argparse


def BaseConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--model_save", type=str)
    parser.add_argument("--model_load", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed", type=int, default=959)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--valid_batch_size", type=int, default=1024)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    # Trainer Config
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--opt_step", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=500)
    parser.add_argument("--Tmax", type=int, default=500)
    parser.add_argument("--max_norm", type=int, default=1)
    parser.add_argument("--metrics", type=str, default="pearson")
    parser.add_argument("--eval_continue", type=int, default=0)
    # Model Config
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--num_labels", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Other Config
    parser.add_argument("--fix_length", type=int, default=1536)
    parser.add_argument("--model_name", type=str, default="base")
    parser.add_argument("--record", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fgm", action="store_true")
    parser.add_argument("--radam", action="store_true")
    parser.add_argument("--da", action="store_true")
    parser.add_argument("--deberta", action="store_true")
    parser.add_argument("--freeze_step", type=int, default=-1)
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()
    return args