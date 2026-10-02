# MSTA-Mamba

Official implementation of **“Multi-Scale Spatio-Temporal Alignment-Enhanced
Mamba for End-diastolic and End-systolic Frame Detection in Echocardiography”**
(accepted by IEEE BIBM).

MSTA-Mamba localizes end-diastolic (ED) and end-systolic (ES) frames by combining:

- a pretrained MambaVision-S backbone for hierarchical visual representations;
- a Temporal Context Mixer (TCM) for short-range inter-frame motion;
- Spatio-Temporal Alignment (STA) across four spatial scales; and
- Key Cycle Constraint Loss (KCCL) for phase anchoring, local rhythm, and
  cross-cycle consistency.

The code is a clean research release reconstructed against the equations in the
accepted manuscript. It contains no patient data, institutional paths, experiment
logs, or model weights.

## Results reported in the paper

| Dataset | Task | ED MAE (frames) | ES MAE (frames) |
|---|---|---:|---:|
| Private POCUS | PLAX ED localization | 1.267 | — |
| EchoNet-Dynamic | A4C ED/ES localization | 1.868 | 2.036 |

The private POCUS dataset contains 2,331 videos from 816 adults and cannot be
redistributed from this repository. EchoNet-Dynamic is publicly available from
its maintainers and remains subject to their data-use terms.

## Installation

The paper experiments used one NVIDIA A100 with CUDA 11.8. Create a fresh Python
3.10 environment and install a PyTorch build appropriate for your CUDA runtime,
then install this project:

```bash
git clone https://github.com/OWNER/MSTA-Mamba.git
cd MSTA-Mamba
pip install -r requirements.txt
pip install -e .
```

The model loads `nvidia/MambaVision-S-1K` through Hugging Face Transformers.
The first run therefore requires network access or a populated local model cache.
Please review the upstream MambaVision license before redistributing its weights.

## Data preparation

Each input contains 32 frames. The portable manifest format is documented in
[`data/README.md`](data/README.md). Medical data and real patient identifiers must
not be committed.

Validate split isolation before training:

```bash
python tools/validate_manifest.py data/echonet/manifest.csv
```

### POCUS

The paper uses non-overlapping 32-frame clips. Short videos are zero padded and a
valid-frame mask excludes padding from both the loss and evaluation. Splits are
made at patient level: 1,854 training videos from 660 patients and 477 test videos
from 156 patients. Only ED is annotated, so labels are background=0 and ED=1.

### EchoNet-Dynamic

Use its official train/validation/test split (7,465/1,288/1,277 videos). Construct
a 32-frame clip around the annotated ED-to-ES interval. When the interval is
shorter than 32 frames, include neighboring frames while keeping ED and ES inside
the clip. The paper excludes 74 videos whose ED-to-ES interval is longer than 32
frames. Labels are background=0, ED=1, and ES=2. For exact submitted-result
reproduction, preprocessing defines interval length as `ES-ED` and labels its
final sampled frame as ES. This explicit submitted-code convention excludes 74
videos; inclusive endpoint indexing would exclude 87.

> The current release intentionally expects prepared clips and manifests. Before
> publication, we plan to add a converter tied to the exact downloaded
> EchoNet-Dynamic metadata layout, so its behavior can be regression-tested.

## Training

POCUS settings (224×224, 50 epochs, batch size 4):

```bash
python tools/train.py --config configs/pocus.yaml
```

EchoNet-Dynamic settings (112×112, 40 epochs, batch size 16):

```bash
python tools/train.py --config configs/echonet.yaml
```

Both configurations use Adam, learning rate 1e-4, weight decay 5e-4, five warm-up
epochs, cosine decay, and the paper's class weights. KCCL uses λr=0.5, α=0.4,
and β=0.2. EchoNet-Dynamic has one annotated ED/ES pair per video, so its
cross-cycle consistency term is naturally zero.

## Evaluation and inference

```bash
python tools/evaluate.py \
  --config configs/echonet.yaml \
  --checkpoint outputs/echonet/last.pth \
  --split test

python tools/infer.py \
  --config configs/echonet.yaml \
  --checkpoint outputs/echonet/last.pth \
  --input path/to/video.avi
```

Evaluation selects the frame with maximum phase probability. POCUS ED error is
the distance to the nearest annotated ED event. EchoNet ED/ES errors are measured
against their respective annotated frames.

## Tests

Tests cover TCM channel behavior, within-video STA alignment, event grouping in
KCCL, loss gradients, and an end-to-end model forward pass with a synthetic
backbone:

```bash
pytest -q
ruff check .
```

They do not download pretrained weights and do not require medical data.

## Repository layout

```text
configs/               paper hyperparameters for both datasets
data/                  manifest specification (no medical data)
msta_mamba/
  data.py               portable clip dataset
  modules.py            TCM and STA
  model.py              complete MSTA-Mamba
  losses.py             KCCL
  metrics.py            localization metrics
tools/                  train, evaluate, and infer entry points
tests/                  CPU-only unit and smoke tests
```

## Reproducibility notes

- Random seeds are set for Python, NumPy, and PyTorch.
- Patient-level separation must be verified before creating a POCUS manifest.
- The code reports population SD (`ddof=0`), matching per-video localization
  errors; confirm this convention against the final camera-ready table.
- Exact paper checkpoints will be linked here after their redistribution and
  privacy review.

## Medical and privacy disclaimer

This software is for research use only. It is not a medical device and must not
be used for diagnosis or treatment. Users are responsible for ethics approval,
consent, de-identification, security, and compliance with applicable law and
dataset agreements.

## Citation

The final DOI and proceedings pages are not yet available in this local draft.
Update the placeholder repository URL and publication metadata in `CITATION.cff`
before release. The remaining author checks are tracked in
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md). A temporary BibTeX entry is:

```bibtex
@inproceedings{cao2026mstamamba,
  title={Multi-Scale Spatio-Temporal Alignment-Enhanced Mamba for
         End-diastolic and End-systolic Frame Detection in Echocardiography},
  author={Cao, Kangyang and Li, Rebecca Sin Tung and Zhou, Kang and Huang, Yuhao
          and Li, Xue and Wong, Randolph Hung Leung and Lee, Alex Pui Wai and
          Hung, Kevin Kei Ching and Leung, Ling Yan and Graham, Colin A and
          Liu, Hongbin and Meng, Gaofeng},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine},
  year={2026}
}
```

## Acknowledgements and license

This work builds on MambaVision and research infrastructure derived from
`lxztju/pytorch_classification`. See [`NOTICE`](NOTICE) for attribution. Code in
this release is provided under the MIT License; third-party code, pretrained
weights, and datasets retain their own licenses.
