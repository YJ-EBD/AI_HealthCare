import numpy as np
import torch
from torchvision import models

import sys
import os
import copy
from data_loader import CustomDataset
from model import resume_checkpoint
from test_model import Model_test
import torch.nn as nn
import argparse
from logger import setup_logger
from torch.utils import data

torch.manual_seed(523)
torch.cuda.manual_seed_all(523)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="released/1_2_3",
        type=str,
    )

    parser.add_argument(
        "--img_path",
        default="dataset/img",
        type=str,
    )
    
    parser.add_argument(
        "--data",
        default="all",
        choices=["all", "train", "val", "test"],
        type=str,
    )
    
    parser.add_argument(
        "--output_dir",
        default="checkpoint",
        type=str,
    )
    
    parser.add_argument("--train", action="store_true")
    
    parser.add_argument("--log", action="store_true")

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--json_path",
        default="dataset/label",
        type=str,
    )

    parser.add_argument(
        "--res",
        default=128,
        type=int,
    )
    
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--limit",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    return args


def snapshot_environment():
    return "\n".join(
        [
            f"cwd={os.getcwd()}",
            f"argv={' '.join(sys.argv)}",
            "entries=" + ", ".join(sorted(os.listdir("."))),
        ]
    )


def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    logger = setup_logger(args.name, args.mode + f"_{args.data}1")
    logger.info(snapshot_environment())
    model = models.resnet50(weights=None)
    model_list = [copy.deepcopy(model) for _ in range(9)]
    # Define 8 resnet models for each region

    ## Class Definitionq
    model_num_class = (
        [np.nan, 15, 7, 7, 0, 12, 0, 5, 7]
        if args.mode == "class"
        else [1, 2, np.nan, 1, 0, 3, 0, np.nan, 2]
    )

    resume_list = list()
    for idx, item in enumerate(model_num_class):
        if not np.isnan(item):
            model_list[idx].fc = nn.Linear(
                model_list[idx].fc.in_features, model_num_class[idx]
            )
            resume_list.append(idx)

    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "1", "state_dict.bin")

    if os.path.isfile(model_dict_path):
        logger.info(f"\033[92mResuming......{check_path}\033[0m")

        for idx in resume_list:
            if idx in [4, 6]:
                continue
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )
    else:
        assert 0, "Check the check-point path, there's not any file in that"

    dataset = CustomDataset(args)
    
    dataset.load_dataset(args, args.data)
    dataset_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    ) 
    # Data Loader

    resnet_model = Model_test(args, model_list, dataset_loader, logger)
    # If the model's acc is higher than best acc, it saves this model
    logger.info("Inferece ...")
    resnet_model.test(model_num_class, dataset_loader)
    logger.info("Finish!")

    return logger


if __name__ == "__main__":
    args = parse_args()
    logger = main(args)
    logger.info(snapshot_environment())
