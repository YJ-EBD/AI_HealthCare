import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils import data
from torchvision import models

from data_loader import CustomDataset, class_num_list


CHECKPOINT_OUTPUTS = {
    1: 15,
    2: 7,
    3: 7,
    5: 12,
    7: 5,
    8: 7,
}

MIRRORED_FACEPARTS = {
    4: 3,
    6: 5,
}

LABEL_LAYOUT = {
    1: [("forehead_wrinkle", 9), ("forehead_pigmentation", 6)],
    2: [("glabellus_wrinkle", 7)],
    3: [("l_perocular_wrinkle", 7)],
    4: [("r_perocular_wrinkle", 7)],
    5: [("l_cheek_pigmentation", 6), ("l_cheek_pore", 6)],
    6: [("r_cheek_pigmentation", 6), ("r_cheek_pore", 6)],
    7: [("lip_dryness", 5)],
    8: [("chin_sagging", 7)],
}


def build_default_paths():
    face_ai_root = Path(__file__).resolve().parents[2]
    return {
        "face_ai_root": face_ai_root,
        "img_path": face_ai_root / "data" / "validation" / "images",
        "json_path": face_ai_root / "data" / "validation" / "labels",
        "checkpoint_root": face_ai_root / "model" / "checkpoint" / "class" / "released" / "1_2_3",
        "output_json": face_ai_root / "executable" / "output" / "classification_summary.json",
    }


def parse_args():
    defaults = build_default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default=str(defaults["img_path"]))
    parser.add_argument("--json_path", default=str(defaults["json_path"]))
    parser.add_argument("--checkpoint_root", default=str(defaults["checkpoint_root"]))
    parser.add_argument("--output_json", default=str(defaults["output_json"]))
    parser.add_argument("--data", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--res", default=128, type=int)
    parser.add_argument("--mode", default="class", choices=["class"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--verbose_log", default="")
    return parser.parse_args()


def resolve_runtime_args(args):
    args.img_path = str(Path(args.img_path).resolve())
    args.json_path = str(Path(args.json_path).resolve())
    args.checkpoint_root = str(Path(args.checkpoint_root).resolve())
    args.output_json = str(Path(args.output_json).resolve())
    return args


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_outputs, checkpoint_path, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    model.to(device)
    model.eval()
    return model


def predict_logits(model, image_tensor):
    if image_tensor.shape[-1] > 128:
        img_l = image_tensor[:, :, :, :128]
        img_r = torch.flip(image_tensor[:, :, :, 128:], dims=[3])
        return model(img_l) + model(img_r)

    if image_tensor.shape[-2] > 128:
        img_t = image_tensor[:, :, :128, :]
        img_b = torch.flip(image_tensor[:, :, 128:, :], dims=[2])
        return model(img_t) + model(img_b)

    return model(image_tensor)


def metric_name(label_name):
    return label_name.split("_")[-1]


def main():
    args = resolve_runtime_args(parse_args())
    device = pick_device()
    checkpoint_root = Path(args.checkpoint_root)

    missing = [
        region_id
        for region_id in CHECKPOINT_OUTPUTS
        if not (checkpoint_root / str(region_id) / "state_dict.bin").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for regions: {missing} in {checkpoint_root}"
        )

    models_by_region = {
        region_id: build_model(
            num_outputs,
            checkpoint_root / str(region_id) / "state_dict.bin",
            device,
        )
        for region_id, num_outputs in CHECKPOINT_OUTPUTS.items()
    }

    dataset = CustomDataset(args)
    dataset.load_dataset(args, args.data)
    loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    task_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    metric_scores = defaultdict(lambda: {"correct": 0, "total": 0})
    verbose_lines = []

    with torch.no_grad():
        for patch_list in loader:
            for facepart, label_spec in LABEL_LAYOUT.items():
                key = str(facepart)
                if key not in patch_list:
                    continue

                area_batch = patch_list[key]
                labels = area_batch[1]
                if labels == {}:
                    continue

                image_tensor = area_batch[0].to(device)
                if facepart in MIRRORED_FACEPARTS:
                    image_tensor = torch.flip(image_tensor, dims=[3])

                model_region = MIRRORED_FACEPARTS.get(facepart, facepart)
                logits = predict_logits(models_by_region[model_region], image_tensor)

                cursor = 0
                desc = area_batch[2][0] if isinstance(area_batch[2], list) else area_batch[2]

                for label_name, num_classes in label_spec:
                    gt = labels[label_name].to(device)
                    pred = torch.argmax(logits[:, cursor : cursor + num_classes], dim=1)
                    cursor += num_classes

                    is_correct = int(abs((pred - gt).item()) < 2)
                    task_scores[label_name]["correct"] += is_correct
                    task_scores[label_name]["total"] += pred.shape[0]

                    group_name = metric_name(label_name)
                    metric_scores[group_name]["correct"] += is_correct
                    metric_scores[group_name]["total"] += pred.shape[0]

                    if args.verbose:
                        verbose_lines.append(
                            f"{desc}::{label_name} pred={pred.item()} gt={gt.item()} correct={bool(is_correct)}"
                        )

    task_summary = {}
    for name, stats in sorted(task_scores.items()):
        total = stats["total"]
        task_summary[name] = {
            "correct": stats["correct"],
            "total": total,
            "accuracy": round(stats["correct"] / total, 6) if total else None,
        }

    metric_summary = {}
    for name, stats in sorted(metric_scores.items()):
        total = stats["total"]
        metric_summary[name] = {
            "correct": stats["correct"],
            "total": total,
            "accuracy": round(stats["correct"] / total, 6) if total else None,
        }

    valid_metric_acc = [
        item["accuracy"] for item in metric_summary.values() if item["accuracy"] is not None
    ]

    summary = {
        "device": str(device),
        "img_path": args.img_path,
        "json_path": args.json_path,
        "checkpoint_root": args.checkpoint_root,
        "dataset_mode": args.data,
        "dataset_items": len(dataset),
        "limit": args.limit,
        "task_accuracy": task_summary,
        "metric_accuracy": metric_summary,
        "mean_metric_accuracy": round(sum(valid_metric_acc) / len(valid_metric_acc), 6)
        if valid_metric_acc
        else None,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.verbose_log:
        verbose_path = Path(args.verbose_log).resolve()
        verbose_path.parent.mkdir(parents=True, exist_ok=True)
        verbose_path.write_text("\n".join(verbose_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
