#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import torch

import nibabel as nib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models.video import (
    r3d_18,
    r2plus1d_18,
    mc3_18,
    R3D_18_Weights,
    R2Plus1D_18_Weights,
    MC3_18_Weights,
)

try:
    from monai.networks.nets import SwinUNETR
    MONAI_AVAILABLE = True
except Exception:
    MONAI_AVAILABLE = False

try:
    from timesformer.models.vit import TimeSformer
    TIMESFORMER_AVAILABLE = True
except Exception:
    TIMESFORMER_AVAILABLE = False


# ============================================================
# DEFAULTS
# ============================================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 15

PHASE_ORDER: Tuple[str, ...] = (
    "C-pre",
    "C+A",
    "C+V",
    "C+Delay",
    "T2WI",
    "DWI",
    "In_Phase",
    "Out_Phase",
)
THREE_PHASES: Tuple[str, ...] = ("C-pre", "C+A", "C+V")

FEATURES_PER_PHASE_TOPO = 150
TOTAL_PHASES = 8
TOTAL_TOPO_FEATURES = FEATURES_PER_PHASE_TOPO * TOTAL_PHASES  # 1200

MLP_HIDDEN_DIMS = [256, 128]
MLP_DROPOUT = 0.3


# ============================================================
# ARGUMENTS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Switchable 3D backbone + TopoGate topology-only gated fusion"
    )

    parser.add_argument("--setting", type=str, choices=["binary", "7class"], required=True)
    parser.add_argument("--phase_mode", type=str, choices=["3phase", "allphase"], required=True)

    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18_3d", "r2plus1d_18", "mc3_18", "x3d", "timesformer", "swinunetr"],
        default="resnet18_3d",
    )
    parser.add_argument("--x3d_variant", type=str, choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"], default="x3d_m")
    parser.add_argument("--timesformer_ckpt", type=str, default="")
    parser.add_argument("--timesformer_num_frames", type=int, default=8)
    parser.add_argument("--timesformer_img_size", type=int, default=224)
    parser.add_argument("--swinunetr_ckpt", type=str, default="")
    parser.add_argument("--swinunetr_feature_size", type=int, default=24)

    # image manifests
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--test_manifest", type=str, required=True)

    # sources on/off
    parser.add_argument("--use_w20", type=str, choices=["yes", "no"], default="yes")
    parser.add_argument("--use_w40", type=str, choices=["yes", "no"], default="yes")

    # optional preprocessing weights
    parser.add_argument("--w20_weight", type=float, default=1.0)
    parser.add_argument("--w40_weight", type=float, default=1.0)

    # topological feature file paths

    parser.add_argument("--w20_train_csv", type=str, default="")
    parser.add_argument("--w20_val_csv", type=str, default="")
    parser.add_argument("--w20_test_csv", type=str, default="")

    parser.add_argument("--w40_train_csv", type=str, default="")
    parser.add_argument("--w40_val_csv", type=str, default="")
    parser.add_argument("--w40_test_csv", type=str, default="")


    parser.add_argument("--out_root", type=str, required=True)

    # image settings
    parser.add_argument("--target_shape", type=int, nargs=3, default=[14, 128, 128])
    parser.add_argument("--crop_shape", type=int, nargs=3, default=[14, 112, 112])

    # training
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional list of random seeds to run sequentially. If omitted, --seed is used.")
    parser.add_argument("--combined_results_csv", type=str, default="", help="Optional path for one combined CSV across all seeds.")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--classifier_hidden_dims", type=int, nargs="+", default=MLP_HIDDEN_DIMS)
    parser.add_argument("--classifier_dropout", type=float, default=MLP_DROPOUT)

    parser.add_argument("--freeze_stem", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--freeze_layer1", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--freeze_layer2", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--freeze_layer3", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--freeze_layer4", type=str, choices=["yes", "no"], default="no")

    # embedding sizes
    parser.add_argument("--fusion_embed_dim", type=int, default=512)
    parser.add_argument("--phase_attn_hidden_dim", type=int, default=256)
    parser.add_argument("--modality_attn_hidden_dim", type=int, default=256)
    parser.add_argument("--topology_hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--topology_dropout", type=float, default=0.3)

    parser.add_argument("--exclude_case_ids_txt", type=str, default="")
    parser.add_argument("--threshold_metric", type=str, choices=["youden", "f1", "kappa", "balanced_accuracy"], default="youden", help="Validation metric used to select binary decision threshold.")
    return parser.parse_args()


# ============================================================
# UTILS
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def yesno_to_bool(x: str) -> bool:
    return str(x).lower() == "yes"


def find_best_binary_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "youden") -> Tuple[float, Dict]:
    """Choose a binary decision threshold using validation predictions only."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).reshape(-1)
    candidates = np.unique(np.concatenate([
        np.linspace(0.05, 0.95, 181),
        probs
    ]))
    best_t = 0.5
    best_score = -1e18
    best_metrics = None
    for t in candidates:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if metric == "youden":
            score = sens + spec - 1.0
        elif metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "kappa":
            score = cohen_kappa_score(y_true, preds)
        elif metric == "balanced_accuracy":
            score = 0.5 * (sens + spec)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        # tie-breaker: prefer threshold closest to 0.5 for stability
        score_for_compare = (float(score), -abs(float(t) - 0.5))
        best_for_compare = (float(best_score), -abs(float(best_t) - 0.5))
        if score_for_compare > best_for_compare:
            best_score = float(score)
            best_t = float(t)
            best_metrics = {"threshold_metric": metric, "threshold_score": float(score), "sensitivity": sens, "specificity": spec}
    return best_t, best_metrics or {}



def r4(x):
    """Round numerical values to 4 decimals for CSV/JSON reporting."""
    if x is None:
        return None
    try:
        if isinstance(x, (float, np.floating)) and np.isnan(float(x)):
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return round(float(x), 4)
    except Exception:
        return x
    return x


def r4_metrics(metrics: Dict) -> Dict:
    """Round scalar metric values to 4 decimals while preserving confusion matrices."""
    out = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            out[k] = v
        else:
            out[k] = r4(v)
    return out


def get_active_phases(phase_mode: str) -> Tuple[str, ...]:
    if phase_mode == "3phase":
        return THREE_PHASES
    if phase_mode == "allphase":
        return PHASE_ORDER
    raise ValueError(f"Unsupported phase_mode: {phase_mode}")


def apply_label_mapping(series: pd.Series, setting: str) -> pd.Series:
    s = series.astype(int).copy()
    if setting == "binary":
        return s.apply(lambda x: 0 if x in [0, 2, 4, 5] else 1).astype(int)
    elif setting == "7class":
        return s.astype(int)
    else:
        raise ValueError(f"Unsupported setting: {setting}")


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)


def read_exclude_case_ids(path_str: str) -> set:
    if not path_str:
        return set()
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Missing exclude file: {p}")
    vals = set()
    with open(p, "r") as f:
        for line in f:
            x = line.strip()
            if x:
                vals.add(x)
    return vals


def maybe_strip_prefix_from_state_dict(state_dict, prefixes=("module.", "backbone.", "model.")):
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        cleaned[nk] = v
    return cleaned


def load_checkpoint_safely(ckpt_path: str):
    if not ckpt_path:
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    return maybe_strip_prefix_from_state_dict(ckpt)


# ============================================================
# IMAGE HELPERS
# ============================================================
def percentile_clip_and_normalize(vol: np.ndarray, pmin=1.0, pmax=99.0) -> np.ndarray:
    lo = np.percentile(vol, pmin)
    hi = np.percentile(vol, pmax)
    vol = np.clip(vol, lo, hi)
    vol = (vol - vol.mean()) / (vol.std() + 1e-6)
    return vol.astype(np.float32)


def resize_volume_torch(vol: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=out_shape, mode="trilinear", align_corners=False)
    return x[0, 0].cpu().numpy().astype(np.float32)


def center_crop_3d(vol: np.ndarray, crop_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = vol.shape
    cd, ch, cw = crop_shape
    sd = max((d - cd) // 2, 0)
    sh = max((h - ch) // 2, 0)
    sw = max((w - cw) // 2, 0)
    return vol[sd:sd + cd, sh:sh + ch, sw:sw + cw]


def random_crop_3d(vol: np.ndarray, crop_shape: Tuple[int, int, int]) -> np.ndarray:
    d, h, w = vol.shape
    cd, ch, cw = crop_shape
    sd = 0 if d <= cd else random.randint(0, d - cd)
    sh = 0 if h <= ch else random.randint(0, h - ch)
    sw = 0 if w <= cw else random.randint(0, w - cw)
    return vol[sd:sd + cd, sh:sh + ch, sw:sw + cw]


def random_flip_3d(vol: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        vol = vol[::-1, :, :].copy()
    if random.random() < 0.5:
        vol = vol[:, ::-1, :].copy()
    if random.random() < 0.5:
        vol = vol[:, :, ::-1].copy()
    return vol


def resize_3d_for_backbone(x, out_t=None, out_h=None, out_w=None):
    size = (
        out_t if out_t is not None else x.shape[2],
        out_h if out_h is not None else x.shape[3],
        out_w if out_w is not None else x.shape[4],
    )
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


def find_phase_file(folder: str, phase_name: str) -> str:
    candidates = [
        os.path.join(folder, phase_name),
        os.path.join(folder, f"{phase_name}.nii.gz"),
        os.path.join(folder, f"{phase_name}.nii"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find phase '{phase_name}' inside {folder}")


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================
TOPO_NON_FEATURE = {
    "case_id", "image_path", "split", "Label", "label"
}


def ordered_f_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("f_")]
    return sorted(cols, key=lambda x: int(x.split("_")[1]))


def get_topo_feature_columns(df: pd.DataFrame, phase_mode: str) -> List[str]:
    fcols = ordered_f_columns(df)

    if len(fcols) < TOTAL_TOPO_FEATURES:
        raise ValueError(
            f"Expected at least {TOTAL_TOPO_FEATURES} topo features, found {len(fcols)}"
        )

    expected = [f"f_{i}" for i in range(TOTAL_TOPO_FEATURES)]
    if fcols[:TOTAL_TOPO_FEATURES] != expected:
        raise ValueError("Topological flat features are not in expected f_0 ... f_1199 order.")

    active_phases = get_active_phases(phase_mode)
    phase_to_idx = {ph: i for i, ph in enumerate(PHASE_ORDER)}
    selected = []

    for ph in active_phases:
        pidx = phase_to_idx[ph]
        start = pidx * FEATURES_PER_PHASE_TOPO
        end = start + FEATURES_PER_PHASE_TOPO
        selected.extend([f"f_{i}" for i in range(start, end)])

    return selected


# ============================================================
# BLOCK PREPARATION
# ============================================================
def prepare_block_from_df(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    case_id_col: str,
    setting: str,
    block_name: str,
):
    feature_cols = [c for c in feature_cols if c in val_df.columns and c in test_df.columns]
    if len(feature_cols) == 0:
        raise ValueError(f"No usable feature columns found for block: {block_name}")

    for df in (train_df, val_df, test_df):
        if case_id_col not in df.columns:
            raise ValueError(f"{block_name}: missing case_id column '{case_id_col}'")
        if label_col not in df.columns:
            raise ValueError(f"{block_name}: missing label column '{label_col}'")

    out = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tmp = df[[case_id_col, label_col] + feature_cols].copy()
        tmp[case_id_col] = tmp[case_id_col].astype(str)
        tmp[label_col] = apply_label_mapping(tmp[label_col], setting)

        X = tmp[feature_cols].apply(pd.to_numeric, errors="coerce")
        case_ids = tmp[case_id_col].tolist()
        y = tmp[label_col].astype(int).values

        out[split_name] = {
            "case_ids": case_ids,
            "y": y,
            "X_df": X,
        }

    return out, feature_cols


def fit_transform_block(train_X_df, val_X_df, test_X_df, weight: float):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_X_df)
    X_val = imputer.transform(val_X_df)
    X_test = imputer.transform(test_X_df)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = X_train * float(weight)
    X_val = X_val * float(weight)
    X_test = X_test * float(weight)

    return (
        X_train.astype(np.float32),
        X_val.astype(np.float32),
        X_test.astype(np.float32),
    )


# ============================================================
# MERGING / INTERSECTION
# ============================================================
def get_common_case_ids(blocks: Dict[str, Dict], split: str, exclude_case_ids: set) -> List[str]:
    sets = []
    for _block_name, block in blocks.items():
        sets.append(set(block[split]["case_ids"]))
    common = set.intersection(*sets) if sets else set()
    common = common - exclude_case_ids
    return sorted(common)


def build_caseid_to_rowidx(case_ids: List[str]) -> Dict[str, int]:
    return {cid: i for i, cid in enumerate(case_ids)}


def align_block_to_common_ids(block_split_dict: Dict, common_ids: List[str]):
    idx_map = build_caseid_to_rowidx(block_split_dict["case_ids"])
    rows = [idx_map[cid] for cid in common_ids]
    X_df = block_split_dict["X_df"].iloc[rows].reset_index(drop=True)
    y = block_split_dict["y"][rows]
    return X_df, y


def check_same_labels(labels_list: List[np.ndarray], split_name: str):
    ref = labels_list[0]
    for i, arr in enumerate(labels_list[1:], start=1):
        if not np.array_equal(ref, arr):
            raise ValueError(f"Label mismatch detected across blocks for split={split_name}, block index={i}")


# ============================================================
# IMAGE DATASET
# ============================================================
class MultiPhaseImageDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        case_ids: List[str],
        phase_names: Tuple[str, ...],
        labels: np.ndarray,
        target_shape=(16, 128, 128),
        crop_shape=(14, 112, 112),
        is_train=True,
    ):
        self.phase_names = list(phase_names)
        self.target_shape = tuple(target_shape)
        self.crop_shape = tuple(crop_shape)
        self.is_train = is_train

        manifest_df = manifest_df.copy()
        manifest_df["case_id"] = manifest_df["case_id"].astype(str)
        self.manifest = manifest_df.set_index("case_id")

        self.case_ids = list(case_ids)
        self.labels = np.asarray(labels).astype(int)

    def __len__(self):
        return len(self.case_ids)

    def _load_single_phase(self, folder: str, phase_name: str) -> np.ndarray:
        path = find_phase_file(folder, phase_name)
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = percentile_clip_and_normalize(vol)
        vol = resize_volume_torch(vol, self.target_shape)

        if self.is_train:
            vol = random_flip_3d(vol)
            vol = random_crop_3d(vol, self.crop_shape)
        else:
            vol = center_crop_3d(vol, self.crop_shape)

        # Keep current behavior from your working script: 3-channel repeated input
        vol = np.repeat(vol[None, ...], 3, axis=0).astype(np.float32)
        return vol

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        label = int(self.labels[idx])

        if case_id not in self.manifest.index:
            raise KeyError(f"case_id {case_id} not found in manifest")

        row = self.manifest.loc[case_id]
        folder = row["prepared_dir"]

        phase_vols = []
        for phase_name in self.phase_names:
            phase_vol = self._load_single_phase(folder, phase_name)
            phase_vols.append(phase_vol)

        image = np.stack(phase_vols, axis=0).astype(np.float32)

        return {
            "case_id": case_id,
            "image": torch.from_numpy(image),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# MULTIMODAL DATALOADER WRAPPER
# ============================================================
class JointCaseDataset(Dataset):
    def __init__(self, image_dataset: MultiPhaseImageDataset, topology_blocks: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]):
        self.image_dataset = image_dataset
        self.case_ids = image_dataset.case_ids
        self.labels = image_dataset.labels

        self.block_tensors = {}
        for block_name, (X, y, case_ids) in topology_blocks.items():
            self.block_tensors[block_name] = {
                "X": torch.tensor(X, dtype=torch.float32),
                "y": np.asarray(y),
                "case_ids": list(case_ids),
            }

            if self.case_ids != list(case_ids):
                raise ValueError(f"Case ID ordering mismatch in block {block_name}")
            if not np.array_equal(self.labels, y):
                raise ValueError(f"Label mismatch in block {block_name}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        img_item = self.image_dataset[idx]
        out = {
            "case_id": img_item["case_id"],
            "image": img_item["image"],
            "label": img_item["label"],
            "topology": {}
        }
        for block_name, block in self.block_tensors.items():
            out["topology"][block_name] = block["X"][idx]
        return out


# ============================================================
# BACKBONES
# ============================================================
class R3D18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

    def forward(self, x):
        return self.backbone(x)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        if freeze_stem:
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
        if freeze_layer1:
            for p in self.backbone.layer1.parameters():
                p.requires_grad = False
        if freeze_layer2:
            for p in self.backbone.layer2.parameters():
                p.requires_grad = False
        if freeze_layer3:
            for p in self.backbone.layer3.parameters():
                p.requires_grad = False
        if freeze_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = False


class R2Plus1DBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

    def forward(self, x):
        return self.backbone(x)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        if freeze_stem:
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
        if freeze_layer1:
            for p in self.backbone.layer1.parameters():
                p.requires_grad = False
        if freeze_layer2:
            for p in self.backbone.layer2.parameters():
                p.requires_grad = False
        if freeze_layer3:
            for p in self.backbone.layer3.parameters():
                p.requires_grad = False
        if freeze_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = False


class MC3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = mc3_18(weights=MC3_18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

    def forward(self, x):
        return self.backbone(x)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        if freeze_stem:
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
        if freeze_layer1:
            for p in self.backbone.layer1.parameters():
                p.requires_grad = False
        if freeze_layer2:
            for p in self.backbone.layer2.parameters():
                p.requires_grad = False
        if freeze_layer3:
            for p in self.backbone.layer3.parameters():
                p.requires_grad = False
        if freeze_layer4:
            for p in self.backbone.layer4.parameters():
                p.requires_grad = False



class X3DBackbone(nn.Module):
    def __init__(self, model_name="x3d_m", pretrained=True):
        super().__init__()
        model = torch.hub.load(
            "facebookresearch/pytorchvideo:main",
            model=model_name,
            pretrained=pretrained,
        )

        # Remove the original head that uses fixed pooling/kernel assumptions
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            self.feature_blocks = nn.ModuleList(list(model.blocks[:-1]))
        else:
            raise ValueError("Unexpected X3D model structure: missing blocks")

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = None
        self.out_dim = 512

    def forward(self, x):
        # x: [B, C, D, H, W]
        feat = x
        for block in self.feature_blocks:
            feat = block(feat)

        feat = self.pool(feat).flatten(1)

        if self.proj is None:
            self.proj = nn.Linear(feat.shape[1], self.out_dim).to(feat.device)

        return self.proj(feat)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        # optional: keep unfrozen for now
        pass







class TimeSformerBackbone(nn.Module):
    def __init__(self, ckpt_path: str, num_frames=8, img_size=224):
        super().__init__()
        if not TIMESFORMER_AVAILABLE:
            raise ImportError("TimeSformer package is not installed.")
        if not ckpt_path:
            raise ValueError("TimeSformer requires --timesformer_ckpt to a pretrained .pyth checkpoint.")
        self.num_frames = num_frames
        self.img_size = img_size
        self.backbone = TimeSformer(
            img_size=img_size,
            num_classes=400,
            num_frames=num_frames,
            attention_type="divided_space_time",
            pretrained_model=ckpt_path,
        )
        self.proj = nn.Linear(400, 512)
        self.out_dim = 512

    def forward(self, x):
        x = resize_3d_for_backbone(x, out_t=self.num_frames, out_h=self.img_size, out_w=self.img_size)
        logits = self.backbone(x)
        return self.proj(logits)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        pass


class SwinUNETRBackbone(nn.Module):
    def __init__(self, img_size=(14, 128, 128), feature_size=24, ckpt_path=""):
        super().__init__()
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required for SwinUNETR.")
        try:
            self.backbone = SwinUNETR(
                img_size=img_size,
                in_channels=3,
                out_channels=2,
                feature_size=feature_size,
                use_checkpoint=False,
            )
        except TypeError:
            self.backbone = SwinUNETR(
                in_channels=3,
                out_channels=2,
                feature_size=feature_size,
                use_checkpoint=False,
            )

        if ckpt_path:
            state = load_checkpoint_safely(ckpt_path)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"[SwinUNETR] Loaded checkpoint: {ckpt_path}")
            print(f"[SwinUNETR] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = None
        self.out_dim = 512

    def _forward_encoder(self, x):
        hidden = self.backbone.swinViT(x, self.backbone.normalize)
        return hidden[-1]

    def forward(self, x):
        feat = self._forward_encoder(x)
        feat = self.pool(feat).flatten(1)
        if self.proj is None:
            self.proj = nn.Linear(feat.shape[1], self.out_dim).to(feat.device)
        return self.proj(feat)

    def freeze_parts(self, freeze_stem=False, freeze_layer1=False, freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
        params = list(self.backbone.parameters())
        n = len(params)
        flags = [freeze_stem, freeze_layer1, freeze_layer2, freeze_layer3, freeze_layer4]
        cuts = [int(n * 0.15), int(n * 0.35), int(n * 0.55), int(n * 0.75), int(n * 0.9)]
        start = 0
        for flag, end in zip(flags, cuts):
            if flag:
                for p in params[start:end]:
                    p.requires_grad = False
            start = end


def build_backbone(backbone_name: str, img_size, args):
    if backbone_name == "resnet18_3d":
        return R3D18Backbone(pretrained=True)
    if backbone_name == "r2plus1d_18":
        return R2Plus1DBackbone(pretrained=True)
    if backbone_name == "mc3_18":
        return MC3Backbone(pretrained=True)
    if backbone_name == "x3d":
        return X3DBackbone(model_name=args.x3d_variant, pretrained=True)
    if backbone_name == "timesformer":
        return TimeSformerBackbone(
            ckpt_path=args.timesformer_ckpt,
            num_frames=args.timesformer_num_frames,
            img_size=args.timesformer_img_size,
        )
    if backbone_name == "swinunetr":
        return SwinUNETRBackbone(
            img_size=img_size,
            feature_size=args.swinunetr_feature_size,
            ckpt_path=args.swinunetr_ckpt,
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")


# ============================================================
# MODEL COMPONENTS
# ============================================================
class PhaseAttention(nn.Module):
    def __init__(self, feat_dim=512, hidden_dim=256):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, phase_feats):
        scores = self.score_net(phase_feats).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = torch.sum(phase_feats * weights.unsqueeze(-1), dim=1)
        return fused, weights


class TopologyEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 256), out_dim=512, dropout=0.3):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout)
                ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TopoGateFusion(nn.Module):
    """Image-conditioned gates for topological feature sources.

    The projected image embedding acts as context. For each topological block,
    a sigmoid gate decides which feature dimensions to preserve or suppress.
    A lightweight attention layer then combines image + gated topology blocks.
    """
    def __init__(self, embed_dim=512, hidden_dim=256, block_names=None):
        super().__init__()
        self.block_names = list(block_names or [])
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, embed_dim),
                nn.Sigmoid(),
            ) for name in self.block_names
        })
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_embed, block_embeds: Dict[str, torch.Tensor]):
        modality_feats = [image_embed]
        modality_names = ["image_context"]
        gate_scalars = [torch.ones(image_embed.size(0), 1, device=image_embed.device)]

        for name in self.block_names:
            emb = block_embeds[name]
            gate = self.gates[name](image_embed)
            gated_emb = emb * gate
            modality_feats.append(gated_emb)
            modality_names.append(f"gated_{name}")
            gate_scalars.append(gate.mean(dim=1, keepdim=True))

        x = torch.stack(modality_feats, dim=1)
        scores = self.score_net(x).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        fused = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        fused = self.norm(fused + image_embed)

        # Diagnostic weights: attention multiplied by average gate strength for topological blocks.
        gate_strength = torch.cat(gate_scalars, dim=1)
        diagnostic_weights = attn_weights * gate_strength
        diagnostic_weights = diagnostic_weights / (diagnostic_weights.sum(dim=1, keepdim=True) + 1e-8)
        return fused, diagnostic_weights, modality_names


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dims=None, dropout=0.3, n_classes=7):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1 if n_classes == 2 else n_classes)

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


class MultiBackboneTopoGate(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        img_size: Tuple[int, int, int],
        phase_names: Tuple[str, ...],
        topology_input_dims: Dict[str, int],
        n_classes: int,
        fusion_embed_dim=512,
        classifier_hidden_dims=None,
        classifier_dropout=0.3,
        topology_hidden_dims=(256, 256),
        topology_dropout=0.3,
        phase_attn_hidden_dim=256,
        modality_attn_hidden_dim=256,
        freeze_stem=False,
        freeze_layer1=False,
        freeze_layer2=False,
        freeze_layer3=False,
        freeze_layer4=False,
        args=None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.phase_names = list(phase_names)
        self.topology_block_names = list(topology_input_dims.keys())
        self.n_classes = n_classes
        self.fusion_embed_dim = fusion_embed_dim

        self.phase_encoder = build_backbone(backbone_name, img_size=img_size, args=args)
        self.phase_encoder.freeze_parts(
            freeze_stem=freeze_stem,
            freeze_layer1=freeze_layer1,
            freeze_layer2=freeze_layer2,
            freeze_layer3=freeze_layer3,
            freeze_layer4=freeze_layer4,
        )

        self.phase_attention = PhaseAttention(
            feat_dim=self.phase_encoder.out_dim,
            hidden_dim=phase_attn_hidden_dim
        )
        self.image_projector = (
            nn.Identity()
            if fusion_embed_dim == self.phase_encoder.out_dim
            else nn.Linear(self.phase_encoder.out_dim, fusion_embed_dim)
        )

        self.topology_encoders = nn.ModuleDict()
        for block_name, in_dim in topology_input_dims.items():
            self.topology_encoders[block_name] = TopologyEncoder(
                in_dim=in_dim,
                hidden_dims=tuple(topology_hidden_dims),
                out_dim=fusion_embed_dim,
                dropout=topology_dropout,
            )

        self.topogate_fusion = TopoGateFusion(
            embed_dim=fusion_embed_dim,
            hidden_dim=modality_attn_hidden_dim,
            block_names=self.topology_block_names,
        )

        self.classifier = ClassifierHead(
            in_dim=fusion_embed_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=classifier_dropout,
            n_classes=n_classes,
        )

    def forward(self, image, topology_blocks: Dict[str, torch.Tensor]):
        B, P, C, D, H, W = image.shape
        assert P == len(self.phase_names), f"Expected {len(self.phase_names)} phases, got {P}"

        per_phase_feats = []
        for p in range(P):
            x_p = image[:, p]
            feat_p = self.phase_encoder(x_p)
            per_phase_feats.append(feat_p)

        phase_feats = torch.stack(per_phase_feats, dim=1)
        deep_feat, phase_weights = self.phase_attention(phase_feats)
        image_embed = self.image_projector(deep_feat)

        modality_feats = [image_embed]
        modality_names = [f"image_{self.backbone_name}"]
        block_embeds = {}

        for block_name in self.topology_block_names:
            encoder = self.topology_encoders[block_name]
            emb = encoder(topology_blocks[block_name])
            block_embeds[block_name] = emb
            modality_feats.append(emb)
            modality_names.append(block_name)

        fused_feat, modality_weights, modality_names = self.topogate_fusion(image_embed, block_embeds)
        logits = self.classifier(fused_feat)

        return {
            "logits": logits,
            "phase_feats": phase_feats,
            "phase_weights": phase_weights,
            "deep_feat": deep_feat,
            "image_embed": image_embed,
            "block_embeds": block_embeds,
            "modality_weights": modality_weights,
            "modality_names": modality_names,
            "fused_feat": fused_feat,
        }


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true)

    if n_classes == 2:
        probs = sigmoid_np(logits.reshape(-1))
        preds = (probs >= threshold).astype(int)

        acc = accuracy_score(y_true, preds)
        kappa = cohen_kappa_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float("nan")

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return r4_metrics({
            "auc": float(auc),
            "accuracy": float(acc),
            "cohen_kappa": float(kappa),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "confusion_matrix": cm.tolist(),
        })

    probs = softmax_np(logits, axis=1)
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_true, preds)
    kappa = cohen_kappa_score(y_true, preds)
    prec = precision_score(y_true, preds, average="macro", zero_division=0)
    rec = recall_score(y_true, preds, average="macro", zero_division=0)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, preds)

    sensitivities = []
    specificities = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivities.append(sens)
        specificities.append(spec)

    return r4_metrics({
        "auc": float(auc),
        "accuracy": float(acc),
        "cohen_kappa": float(kappa),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "sensitivity": float(np.mean(sensitivities)),
        "specificity": float(np.mean(specificities)),
        "confusion_matrix": cm.tolist(),
    })


def get_loss_function(n_classes: int):
    return nn.BCEWithLogitsLoss() if n_classes == 2 else nn.CrossEntropyLoss()


def save_predictions(out_path: Path, case_ids: List[str], y_true: np.ndarray, logits: np.ndarray, n_classes: int, threshold: float = 0.5):
    rows = []

    if n_classes == 2:
        probs = sigmoid_np(logits.reshape(-1))
        preds = (probs >= threshold).astype(int)
        for cid, yt, yp, pr, lg in zip(case_ids, y_true, preds, probs, logits.reshape(-1)):
            rows.append({
                "case_id": cid,
                "y_true": int(yt),
                "y_pred": int(yp),
                "prob_class_1": r4(pr),
                "threshold": r4(threshold),
                "logit": r4(lg),
            })
    else:
        probs = softmax_np(logits, axis=1)
        preds = np.argmax(probs, axis=1)
        for i, (cid, yt, yp) in enumerate(zip(case_ids, y_true, preds)):
            row = {
                "case_id": cid,
                "y_true": int(yt),
                "y_pred": int(yp),
            }
            for c in range(probs.shape[1]):
                row[f"prob_class_{c}"] = r4(probs[i, c])
                row[f"logit_class_{c}"] = r4(logits[i, c])
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_phase_attention_csv(out_path: Path, case_ids: List[str], phase_weights: np.ndarray, phase_names: Tuple[str, ...]):
    rows = []
    for i, cid in enumerate(case_ids):
        row = {"case_id": cid}
        for j, ph in enumerate(phase_names):
            row[f"phase_weight__{ph}"] = r4(phase_weights[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def save_modality_attention_csv(out_path: Path, case_ids: List[str], modality_weights: np.ndarray, modality_names: List[str]):
    rows = []
    for i, cid in enumerate(case_ids):
        row = {"case_id": cid}
        for j, name in enumerate(modality_names):
            row[f"modality_weight__{name}"] = r4(modality_weights[i, j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def summarize_modality_weights(modality_weights: np.ndarray, modality_names: List[str], y_true: np.ndarray):
    summary = {
        "overall_mean": {},
        "overall_std": {},
        "per_class_mean": {},
        "per_class_count": {},
    }

    for j, name in enumerate(modality_names):
        summary["overall_mean"][name] = r4(np.mean(modality_weights[:, j]))
        summary["overall_std"][name] = r4(np.std(modality_weights[:, j]))

    classes = sorted(np.unique(y_true).tolist())
    for cls in classes:
        idx = np.where(y_true == cls)[0]
        summary["per_class_count"][str(cls)] = int(len(idx))
        summary["per_class_mean"][str(cls)] = {
            name: r4(np.mean(modality_weights[idx, j])) for j, name in enumerate(modality_names)
        }

    return summary


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, n_classes):
    model.train()
    running_loss = 0.0

    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        topology = {k: v.to(DEVICE, non_blocking=True) for k, v in batch["topology"].items()}

        optimizer.zero_grad()
        outputs = model(image, topology)
        logits = outputs["logits"]

        if n_classes == 2:
            loss = criterion(logits.view(-1), y.float())
        else:
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, n_classes):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_y = []
    all_case_ids = []
    all_phase_weights = []
    all_modality_weights = []
    modality_names = None

    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        topology = {k: v.to(DEVICE, non_blocking=True) for k, v in batch["topology"].items()}

        outputs = model(image, topology)
        logits = outputs["logits"]
        phase_weights = outputs["phase_weights"]
        modality_weights = outputs["modality_weights"]
        modality_names = outputs["modality_names"]

        if n_classes == 2:
            loss = criterion(logits.view(-1), y.float())
            logits_np = logits.view(-1).detach().cpu().numpy()
        else:
            loss = criterion(logits, y)
            logits_np = logits.detach().cpu().numpy()

        running_loss += loss.item() * image.size(0)
        all_logits.append(logits_np)
        all_y.append(y.detach().cpu().numpy())
        all_case_ids.extend(list(batch["case_id"]))
        all_phase_weights.append(phase_weights.detach().cpu().numpy())
        all_modality_weights.append(modality_weights.detach().cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    all_y = np.concatenate(all_y)

    if n_classes == 2:
        all_logits = np.concatenate(all_logits).reshape(-1)
    else:
        all_logits = np.concatenate(all_logits, axis=0)

    all_phase_weights = np.concatenate(all_phase_weights, axis=0)
    all_modality_weights = np.concatenate(all_modality_weights, axis=0)

    metrics = compute_metrics(all_y, all_logits, n_classes)
    return avg_loss, metrics, all_logits, all_y, all_case_ids, all_phase_weights, all_modality_weights, modality_names


# ============================================================
# MAIN
# ============================================================
def run_one_seed(args, seed: int):
    args.seed = int(seed)
    set_seed(args.seed)

    if args.backbone == "swinunetr" and not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for the swinunetr backbone.")
    if args.backbone == "timesformer" and not TIMESFORMER_AVAILABLE:
        raise ImportError("TimeSformer package is required for the timesformer backbone.")

    active_phases = get_active_phases(args.phase_mode)
    exclude_case_ids = read_exclude_case_ids(args.exclude_case_ids_txt)

    use_w20 = yesno_to_bool(args.use_w20)
    use_w40 = yesno_to_bool(args.use_w40)
    use_any_topology = any([use_w20, use_w40])

    # --------------------------------------------------------
    # load manifests
    # --------------------------------------------------------
    train_manifest = load_csv(args.train_manifest)
    val_manifest = load_csv(args.val_manifest)
    test_manifest = load_csv(args.test_manifest)

    required_manifest_cols = {"case_id", "label", "prepared_dir"}
    for name, df in [("train", train_manifest), ("val", val_manifest), ("test", test_manifest)]:
        missing = required_manifest_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} manifest missing columns: {missing}")

    train_manifest = train_manifest.copy()
    val_manifest = val_manifest.copy()
    test_manifest = test_manifest.copy()
    train_manifest["case_id"] = train_manifest["case_id"].astype(str)
    val_manifest["case_id"] = val_manifest["case_id"].astype(str)
    test_manifest["case_id"] = test_manifest["case_id"].astype(str)

    train_manifest["label"] = apply_label_mapping(train_manifest["label"], args.setting)
    val_manifest["label"] = apply_label_mapping(val_manifest["label"], args.setting)
    test_manifest["label"] = apply_label_mapping(test_manifest["label"], args.setting)

    # --------------------------------------------------------
    # prepare selected topological blocks
    # --------------------------------------------------------
    blocks = {}
    block_meta = {}

    if use_w20:
        tr = load_csv(args.w20_train_csv)
        va = load_csv(args.w20_val_csv)
        te = load_csv(args.w20_test_csv)

        feat_cols = get_topo_feature_columns(tr, args.phase_mode)
        block, used_cols = prepare_block_from_df(
            tr, va, te,
            feature_cols=feat_cols,
            label_col="Label" if "Label" in tr.columns else "label",
            case_id_col="case_id",
            setting=args.setting,
            block_name="w20",
        )
        blocks["w20"] = block
        block_meta["w20"] = {
            "num_raw_features": len(used_cols),
            "weight": args.w20_weight,
            "feature_columns": used_cols,
        }

    if use_w40:
        tr = load_csv(args.w40_train_csv)
        va = load_csv(args.w40_val_csv)
        te = load_csv(args.w40_test_csv)

        feat_cols = get_topo_feature_columns(tr, args.phase_mode)
        block, used_cols = prepare_block_from_df(
            tr, va, te,
            feature_cols=feat_cols,
            label_col="Label" if "Label" in tr.columns else "label",
            case_id_col="case_id",
            setting=args.setting,
            block_name="w40",
        )
        blocks["w40"] = block
        block_meta["w40"] = {
            "num_raw_features": len(used_cols),
            "weight": args.w40_weight,
            "feature_columns": used_cols,
        }

    # --------------------------------------------------------
    # common ids across selected blocks + manifests
    # --------------------------------------------------------
    manifest_train_ids = set(train_manifest["case_id"].tolist()) - exclude_case_ids
    manifest_val_ids = set(val_manifest["case_id"].tolist()) - exclude_case_ids
    manifest_test_ids = set(test_manifest["case_id"].tolist()) - exclude_case_ids

    if use_any_topology:
        common_train_ids = get_common_case_ids(blocks, "train", exclude_case_ids)
        common_val_ids = get_common_case_ids(blocks, "val", exclude_case_ids)
        common_test_ids = get_common_case_ids(blocks, "test", exclude_case_ids)

        common_train_ids = sorted(set(common_train_ids).intersection(manifest_train_ids))
        common_val_ids = sorted(set(common_val_ids).intersection(manifest_val_ids))
        common_test_ids = sorted(set(common_test_ids).intersection(manifest_test_ids))

        if len(common_train_ids) == 0:
            raise RuntimeError("No common training cases across manifests and selected feature sources.")
        if len(common_val_ids) == 0:
            raise RuntimeError("No common validation cases across manifests and selected feature sources.")
        if len(common_test_ids) == 0:
            raise RuntimeError("No common test cases across manifests and selected feature sources.")
    else:
        common_train_ids = sorted(manifest_train_ids)
        common_val_ids = sorted(manifest_val_ids)
        common_test_ids = sorted(manifest_test_ids)

    # --------------------------------------------------------
    # build topological maps if any topological blocks are enabled
    # --------------------------------------------------------
    train_labels_all, val_labels_all, test_labels_all = [], [], []

    block_dims = {}
    topology_input_dims = {}
    topology_train_map = {}
    topology_val_map = {}
    topology_test_map = {}

    if use_any_topology:
        for block_name, block in blocks.items():
            Xtr_df, ytr = align_block_to_common_ids(block["train"], common_train_ids)
            Xva_df, yva = align_block_to_common_ids(block["val"], common_val_ids)
            Xte_df, yte = align_block_to_common_ids(block["test"], common_test_ids)

            train_labels_all.append(ytr)
            val_labels_all.append(yva)
            test_labels_all.append(yte)

            weight = block_meta[block_name]["weight"]
            Xtr, Xva, Xte = fit_transform_block(Xtr_df, Xva_df, Xte_df, weight=weight)

            topology_train_map[block_name] = (Xtr, ytr, common_train_ids)
            topology_val_map[block_name] = (Xva, yva, common_val_ids)
            topology_test_map[block_name] = (Xte, yte, common_test_ids)

            topology_input_dims[block_name] = Xtr.shape[1]

            block_dims[block_name] = {
                "train_dim": int(Xtr.shape[1]),
                "val_dim": int(Xva.shape[1]),
                "test_dim": int(Xte.shape[1]),
                "weight": float(weight),
            }

        check_same_labels(train_labels_all, "train")
        check_same_labels(val_labels_all, "val")
        check_same_labels(test_labels_all, "test")

        y_train = train_labels_all[0]
        y_val = val_labels_all[0]
        y_test = test_labels_all[0]
    else:
        train_manifest_idx = train_manifest.set_index("case_id")
        val_manifest_idx = val_manifest.set_index("case_id")
        test_manifest_idx = test_manifest.set_index("case_id")

        y_train = train_manifest_idx.loc[common_train_ids]["label"].to_numpy(dtype=int)
        y_val = val_manifest_idx.loc[common_val_ids]["label"].to_numpy(dtype=int)
        y_test = test_manifest_idx.loc[common_test_ids]["label"].to_numpy(dtype=int)

    n_classes = len(np.unique(y_train))

    # --------------------------------------------------------
    # output dir
    # --------------------------------------------------------
    run_name_parts = [f"{args.backbone}_topogate"]
    if use_w20:
        run_name_parts.append("w20")
    if use_w40:
        run_name_parts.append("w40")
    if not use_any_topology:
        run_name_parts.append("image_only")

    run_name = f"{args.setting}_{args.phase_mode}_{'_'.join(run_name_parts)}"
    run_out = Path(args.out_root) / run_name / f"seed_{args.seed}"
    ensure_dir(run_out)

    print("=" * 90)
    print("[INFO] Switchable 3D backbone TopoGate gated fusion")
    print(f"[INFO] setting            : {args.setting}")
    print(f"[INFO] phase_mode         : {args.phase_mode}")
    print(f"[INFO] backbone           : {args.backbone}")
    print(f"[INFO] seed               : {args.seed}")
    print(f"[INFO] active_phases      : {active_phases}")
    print(f"[INFO] enabled blocks     : {list(blocks.keys()) if use_any_topology else ['image_only']}")
    print(f"[INFO] common train       : {len(common_train_ids)}")
    print(f"[INFO] common val         : {len(common_val_ids)}")
    print(f"[INFO] common test        : {len(common_test_ids)}")
    print(f"[INFO] n_classes          : {n_classes}")
    print("=" * 90)

    # --------------------------------------------------------
    # datasets / loaders
    # --------------------------------------------------------
    train_img_ds = MultiPhaseImageDataset(
        manifest_df=train_manifest,
        case_ids=common_train_ids,
        phase_names=active_phases,
        labels=y_train,
        target_shape=tuple(args.target_shape),
        crop_shape=tuple(args.crop_shape),
        is_train=True,
    )
    val_img_ds = MultiPhaseImageDataset(
        manifest_df=val_manifest,
        case_ids=common_val_ids,
        phase_names=active_phases,
        labels=y_val,
        target_shape=tuple(args.target_shape),
        crop_shape=tuple(args.crop_shape),
        is_train=False,
    )
    test_img_ds = MultiPhaseImageDataset(
        manifest_df=test_manifest,
        case_ids=common_test_ids,
        phase_names=active_phases,
        labels=y_test,
        target_shape=tuple(args.target_shape),
        crop_shape=tuple(args.crop_shape),
        is_train=False,
    )

    train_ds = JointCaseDataset(train_img_ds, topology_train_map)
    val_ds = JointCaseDataset(val_img_ds, topology_val_map)
    test_ds = JointCaseDataset(test_img_ds, topology_test_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------
    model = MultiBackboneTopoGate(
        backbone_name=args.backbone,
        img_size=tuple(args.target_shape),
        phase_names=active_phases,
        topology_input_dims=topology_input_dims,
        n_classes=n_classes,
        fusion_embed_dim=args.fusion_embed_dim,
        classifier_hidden_dims=args.classifier_hidden_dims,
        classifier_dropout=args.classifier_dropout,
        topology_hidden_dims=tuple(args.topology_hidden_dims),
        topology_dropout=args.topology_dropout,
        phase_attn_hidden_dim=args.phase_attn_hidden_dim,
        modality_attn_hidden_dim=args.modality_attn_hidden_dim,
        freeze_stem=yesno_to_bool(args.freeze_stem),
        freeze_layer1=yesno_to_bool(args.freeze_layer1),
        freeze_layer2=yesno_to_bool(args.freeze_layer2),
        freeze_layer3=yesno_to_bool(args.freeze_layer3),
        freeze_layer4=yesno_to_bool(args.freeze_layer4),
        args=args,
    ).to(DEVICE)

    criterion = get_loss_function(n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -float("inf")
    best_epoch = -1
    best_state = None
    patience_counter = 0
    history = []

    # --------------------------------------------------------
    # training
    # --------------------------------------------------------
   
   


    for epoch in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, n_classes)
        val_loss, val_metrics, _, _, _, _, _, _ = evaluate(model, val_loader, criterion, n_classes)

        val_auc = val_metrics["auc"]
        improved = False
        if np.isnan(val_auc):
            if best_epoch == -1:
                improved = True
        elif val_auc > best_val_auc:
            improved = True

        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch,
            "train_loss": r4(train_loss),
            "val_loss": r4(val_loss),
            "val_auc": r4(val_metrics["auc"]),
            "val_accuracy": r4(val_metrics["accuracy"]),
            "val_cohen_kappa": r4(val_metrics["cohen_kappa"]),
            "val_precision": r4(val_metrics["precision"]),
            "val_recall": r4(val_metrics["recall"]),
            "val_f1": r4(val_metrics["f1"]),
            "val_sensitivity": r4(val_metrics["sensitivity"]),
            "val_specificity": r4(val_metrics["specificity"]),
        })

        print(
            f"[{args.backbone}-ModAttn] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_kappa={val_metrics['cohen_kappa']:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"[{args.backbone}-ModAttn] Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best state saved.")

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # save model / history
    # --------------------------------------------------------
    model_path = run_out / f"{args.backbone}_TopoGate_best.pt"
    torch.save(model.state_dict(), model_path)

    history_csv = run_out / f"{args.backbone}_TopoGate_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)

    # --------------------------------------------------------
    # final val / test
    # --------------------------------------------------------
    val_loss, val_metrics_fixed, val_logits, val_y, val_case_ids, val_phase_weights, val_modality_weights, modality_names = evaluate(model, val_loader, criterion, n_classes)
    test_loss, test_metrics_fixed, test_logits, test_y, test_case_ids, test_phase_weights, test_modality_weights, _ = evaluate(model, test_loader, criterion, n_classes)

    selected_threshold = 0.5
    threshold_info = {"threshold_metric": "fixed_0.5", "threshold_score": None}
    if n_classes == 2:
        val_probs = sigmoid_np(val_logits.reshape(-1))
        selected_threshold, threshold_info = find_best_binary_threshold(val_y, val_probs, metric=args.threshold_metric)
        val_metrics = compute_metrics(val_y, val_logits, n_classes, threshold=selected_threshold)
        test_metrics = compute_metrics(test_y, test_logits, n_classes, threshold=selected_threshold)
        print(f"[THRESHOLD] Selected threshold={selected_threshold:.4f} using validation {args.threshold_metric}")
    else:
        val_metrics = val_metrics_fixed
        test_metrics = test_metrics_fixed

    save_predictions(run_out / f"{args.backbone}_val_predictions.csv", val_case_ids, val_y, val_logits, n_classes, threshold=selected_threshold)
    save_predictions(run_out / f"{args.backbone}_test_predictions.csv", test_case_ids, test_y, test_logits, n_classes, threshold=selected_threshold)

    save_phase_attention_csv(run_out / f"{args.backbone}_val_phase_attention.csv", val_case_ids, val_phase_weights, active_phases)
    save_phase_attention_csv(run_out / f"{args.backbone}_test_phase_attention.csv", test_case_ids, test_phase_weights, active_phases)

    save_modality_attention_csv(run_out / f"{args.backbone}_val_modality_attention.csv", val_case_ids, val_modality_weights, modality_names)
    save_modality_attention_csv(run_out / f"{args.backbone}_test_modality_attention.csv", test_case_ids, test_modality_weights, modality_names)

    val_modality_summary = summarize_modality_weights(val_modality_weights, modality_names, val_y)
    test_modality_summary = summarize_modality_weights(test_modality_weights, modality_names, test_y)
    save_json(val_modality_summary, run_out / f"{args.backbone}_val_modality_attention_summary.json")
    save_json(test_modality_summary, run_out / f"{args.backbone}_test_modality_attention_summary.json")

    results = {
        "model": f"{args.backbone}_TopoGate",
        "seed": int(args.seed),
        "setting": args.setting,
        "phase_mode": args.phase_mode,
        "backbone": args.backbone,
        "blocks_used": ",".join(blocks.keys()) if use_any_topology else "image_only",
        "modality_names": ",".join(modality_names),
        "best_epoch": int(best_epoch),
        "selected_threshold": r4(selected_threshold),
        "threshold_metric": threshold_info.get("threshold_metric"),
        "threshold_score": r4(threshold_info.get("threshold_score")),
        "best_val_auc": r4(val_metrics["auc"]),
        "val_loss": r4(val_loss),
        "val_auc": r4(val_metrics["auc"]),
        "val_accuracy": r4(val_metrics["accuracy"]),
        "val_cohen_kappa": r4(val_metrics["cohen_kappa"]),
        "val_precision": r4(val_metrics["precision"]),
        "val_recall": r4(val_metrics["recall"]),
        "val_f1": r4(val_metrics["f1"]),
        "val_sensitivity": r4(val_metrics["sensitivity"]),
        "val_specificity": r4(val_metrics["specificity"]),
        "val_confusion_matrix": json.dumps(val_metrics["confusion_matrix"]),
        "test_loss": r4(test_loss),
        "test_auc": r4(test_metrics["auc"]),
        "test_accuracy": r4(test_metrics["accuracy"]),
        "test_cohen_kappa": r4(test_metrics["cohen_kappa"]),
        "test_precision": r4(test_metrics["precision"]),
        "test_recall": r4(test_metrics["recall"]),
        "test_f1": r4(test_metrics["f1"]),
        "test_sensitivity": r4(test_metrics["sensitivity"]),
        "test_specificity": r4(test_metrics["specificity"]),
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        "num_train": len(common_train_ids),
        "num_val": len(common_val_ids),
        "num_test": len(common_test_ids),
        "model_path": str(model_path),
        "history_csv": str(history_csv),
    }
    result_df = pd.DataFrame([results])
    result_df.to_csv(run_out / "fusion_results.csv", index=False)

    combined_csv = (
        Path(args.combined_results_csv)
        if args.combined_results_csv
        else Path(args.out_root) / run_name / "all_seed_fusion_results.csv"
    )
    ensure_dir(combined_csv.parent)
    result_df.to_csv(
        combined_csv,
        mode="a",
        header=not combined_csv.exists(),
        index=False,
    )
    print(f"[INFO] Appended this seed results to: {combined_csv}")

    with open(run_out / "common_train_case_ids.txt", "w") as f:
        for cid in common_train_ids:
            f.write(f"{cid}\n")
    with open(run_out / "common_val_case_ids.txt", "w") as f:
        for cid in common_val_ids:
            f.write(f"{cid}\n")
    with open(run_out / "common_test_case_ids.txt", "w") as f:
        for cid in common_test_ids:
            f.write(f"{cid}\n")

    config = {
        "setting": args.setting,
        "phase_mode": args.phase_mode,
        "backbone": args.backbone,
        "active_phases": list(active_phases),
        "use_w20": use_w20,
        "use_w40": use_w40,
        "w20_weight": float(args.w20_weight),
        "w40_weight": float(args.w40_weight),
        "common_train": len(common_train_ids),
        "common_val": len(common_val_ids),
        "common_test": len(common_test_ids),
        "n_classes": int(n_classes),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "seed": int(args.seed),
        "device": DEVICE,
        "selected_threshold": r4(selected_threshold),
        "threshold_metric": threshold_info.get("threshold_metric"),
        "threshold_score": r4(threshold_info.get("threshold_score")),
        "block_dims": block_dims,
        "exclude_case_ids_count": len(exclude_case_ids),
        "freeze_stem": args.freeze_stem,
        "freeze_layer1": args.freeze_layer1,
        "freeze_layer2": args.freeze_layer2,
        "freeze_layer3": args.freeze_layer3,
        "freeze_layer4": args.freeze_layer4,
        "target_shape": args.target_shape,
        "crop_shape": args.crop_shape,
        "fusion_embed_dim": args.fusion_embed_dim,
        "phase_attn_hidden_dim": args.phase_attn_hidden_dim,
        "modality_attn_hidden_dim": args.modality_attn_hidden_dim,
        "modality_names": modality_names,
    }
    save_json(config, run_out / "run_config.json")

    feature_manifest = {}
    for k, meta in block_meta.items():
        feature_manifest[k] = {
            "num_raw_features": meta["num_raw_features"],
            "weight": meta["weight"],
            "feature_columns": meta["feature_columns"],
        }
    save_json(feature_manifest, run_out / "feature_manifest.json")

    print(f"[DONE] Results saved to: {run_out}")
    print("[INFO] Finished seed run.")
    return results


def main():
    args = parse_args()
    seed_list = args.seeds if args.seeds is not None else [args.seed]

    print("=" * 90)
    print(f"[INFO] Running {len(seed_list)} seed(s): {seed_list}")
    print("=" * 90)

    for seed in seed_list:
        print("\n" + "#" * 90)
        print(f"[INFO] Starting seed {seed}")
        print("#" * 90)
        run_one_seed(args, seed)

    print("=" * 90)
    print(f"[DONE] Completed {len(seed_list)} seed(s).")
    print("=" * 90)


if __name__ == "__main__":
    main()
