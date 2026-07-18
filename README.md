# HTFD

Official implementation for **HTFD**: Heterogeneous Time-Frequency Diffusion for Financial Index Generation.

HTFD is a dual-branch Mallat DWT VP-DDPM with time–scale conditioning, DyWPE, and cross-frequency losses for financial index time-series generation.

## Setup

```bash
pip install -r requirements.txt
export MPLBACKEND=Agg
```

Python ≥ 3.10 recommended.

## Experiments

```bash
# SP500, T=128
bash script/spx_t128.sh
# Windows:
#   .\script\spx_t128.ps1
```

Or:

```bash
python -u run.py \
  --data spx --seq_len 128 --train_epochs 200 \
  --batch_size 2000 --d_model 64 --norm_mode revin --export 1
```

Runtime exports are written under `outputs/` (gitignored). Example figures live in `figs/`; example metric txts live in `results/`.

## Repository layout

```
HTFD/
├── run.py                 # CLI entry
├── training.py            # training helper
├── Model.py               # public model facade
├── configs/               # dataset / run configs
├── data_preprocessing/    # CSV loaders
├── dataset/               # SPX / CSI / CSI300 / CSI500
├── models/                # HTFD + BranchDenoiser
├── layers/                # RevIN, DWT, DyWPE, attention, samplers
├── exp/                   # generation pipeline
├── eval_utils/            # DS/PS, Context-FID, DTW-JS, financial metrics helpers
├── utils/                 # training / export helpers
├── figs/                  # example figures
├── results/               # example metric txts
└── script/                # launch scripts
```

## Data

Bundled under `dataset/`: **SPX**, **CSI**, **CSI300**, **CSI500** (DateTime + close). Set `--data` / `HTFD_DATASET` to `spx` | `csi` | `csi300` | `csi500`.

## Citation

```bibtex
@inproceedings{htfd2026,
  title={HTFD: Heterogeneous Time-Frequency Diffusion for Financial Index Generation},
  author={Anonymous},
  booktitle={AAAI},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
