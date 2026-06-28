# TopoGate: Topology-Only Gated Fusion for Multi-Phase Liver MRI

This repository contains a PyTorch implementation of **TopoGate**, a topology-guided gated fusion framework for binary or seven-class liver lesion classification from multi-phase MRI.

This version intentionally uses only:

- deep image features from multi-phase MRI volumes,
- sliding-band topological features with bandwidth `w20`,
- sliding-band topological features with bandwidth `w40`.

Radiomics and signed-distance features have been removed from this cleaned version.

## Model overview

The model follows this pipeline:

1. Each MRI phase is processed by a shared 3D backbone.
2. Phase attention aggregates phase-specific embeddings into one image representation.
3. The image representation is projected into a shared latent space.
4. The `w20` and `w40` topological feature vectors are separately encoded using MLP encoders.
5. The projected image embedding gates each topological embedding through a feature-wise sigmoid gate.
6. A modality-attention layer learns the contribution of the image embedding, gated `w20`, and gated `w40`.
7. The fused representation is combined with the image embedding using a residual connection and LayerNorm.
8. A lightweight MLP classifier outputs either binary or seven-class predictions.

## Main script

```bash
topogate_topology_only.py
```

## Required inputs

### Image manifests

The script requires train, validation, and test manifests:

```bash
--train_manifest path/to/train_manifest.csv
--val_manifest path/to/val_manifest.csv
--test_manifest path/to/test_manifest.csv
```

Each manifest should contain at least:

```text
case_id,label,prepared_dir
```

The `prepared_dir` column should point to a folder containing the phase volumes for each patient.

### Topological feature CSVs

For `w20`:

```bash
--w20_train_csv path/to/w20_train.csv
--w20_val_csv path/to/w20_val.csv
--w20_test_csv path/to/w20_test.csv
```

For `w40`:

```bash
--w40_train_csv path/to/w40_train.csv
--w40_val_csv path/to/w40_val.csv
--w40_test_csv path/to/w40_test.csv
```

Each topology CSV should contain:

```text
case_id,label or Label,f_0,f_1,...,f_1199
```

The code assumes 150 topology features per phase and 8 phases, giving 1200 topology features in total.

## Basic usage

### Binary classification with all 8 MRI phases

```bash
python topogate_topology_only.py \
  --setting binary \
  --phase_mode allphase \
  --backbone r2plus1d_18 \
  --train_manifest data/train_manifest.csv \
  --val_manifest data/val_manifest.csv \
  --test_manifest data/test_manifest.csv \
  --use_w20 yes \
  --use_w40 yes \
  --w20_train_csv features/w20_train.csv \
  --w20_val_csv features/w20_val.csv \
  --w20_test_csv features/w20_test.csv \
  --w40_train_csv features/w40_train.csv \
  --w40_val_csv features/w40_val.csv \
  --w40_test_csv features/w40_test.csv \
  --out_root outputs/topogate
```

### Seven-class classification

```bash
python topogate_topology_only.py \
  --setting 7class \
  --phase_mode allphase \
  --backbone r2plus1d_18 \
  --train_manifest data/train_manifest.csv \
  --val_manifest data/val_manifest.csv \
  --test_manifest data/test_manifest.csv \
  --use_w20 yes \
  --use_w40 yes \
  --w20_train_csv features/w20_train.csv \
  --w20_val_csv features/w20_val.csv \
  --w20_test_csv features/w20_test.csv \
  --w40_train_csv features/w40_train.csv \
  --w40_val_csv features/w40_val.csv \
  --w40_test_csv features/w40_test.csv \
  --out_root outputs/topogate
```

### Multi-seed run

```bash
python topogate_topology_only.py \
  --setting binary \
  --phase_mode allphase \
  --backbone r2plus1d_18 \
  --seeds 42 43 44 45 46 \
  --train_manifest data/train_manifest.csv \
  --val_manifest data/val_manifest.csv \
  --test_manifest data/test_manifest.csv \
  --use_w20 yes \
  --use_w40 yes \
  --w20_train_csv features/w20_train.csv \
  --w20_val_csv features/w20_val.csv \
  --w20_test_csv features/w20_test.csv \
  --w40_train_csv features/w40_train.csv \
  --w40_val_csv features/w40_val.csv \
  --w40_test_csv features/w40_test.csv \
  --out_root outputs/topogate \
  --combined_results_csv outputs/topogate/all_seed_results.csv
```

## Important arguments

| Argument | Description |
|---|---|
| `--setting` | Either `binary` or `7class`. |
| `--phase_mode` | Either `3phase` or `allphase`. |
| `--backbone` | One of `resnet18_3d`, `r2plus1d_18`, `mc3_18`, `x3d`, `timesformer`, or `swinunetr`. |
| `--use_w20` | Include topology features from bandwidth `w20`. |
| `--use_w40` | Include topology features from bandwidth `w40`. |
| `--fusion_embed_dim` | Shared latent dimension for image and topology embeddings. Default: `512`. |
| `--topology_hidden_dims` | Hidden dimensions for topology encoders. Default: `256 256`. |
| `--topology_dropout` | Dropout used in topology encoders. Default: `0.3`. |
| `--modality_attn_hidden_dim` | Hidden size of the modality-attention network. Default: `256`. |
| `--threshold_metric` | Validation metric used to select binary threshold: `youden`, `f1`, `kappa`, or `balanced_accuracy`. |

## Outputs

For each run, the script saves:

- best model checkpoint,
- training history CSV,
- validation and test prediction CSVs,
- phase-attention weights,
- modality-attention weights,
- modality-attention summaries,
- run configuration JSON,
- feature manifest JSON,
- final results CSV.

## Notes

- The cleaned script removes all radiomics and signed-distance options and feature-loading blocks.
- The only optional auxiliary sources are `w20` and `w40` topology features.
- The image embedding is included in the modality-attention fusion together with gated topology embeddings.
- The final fused representation is computed using residual normalization:

```text
LayerNorm(fused_representation + image_embedding)
```
