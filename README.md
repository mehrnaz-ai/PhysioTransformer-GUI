# LT-Predict GUI

**Interactive Lactate Threshold Predictor — PhysioTransformer Demo**

An interactive desktop application for non-invasive lactate threshold (LT)
estimation using the PhysioTransformer deep learning model.
Enter heart rate and power data from an incremental exercise test and
get an instant LT prediction with attention visualization.

> **This GUI is a demonstration tool.**
> For scientific evaluation and paper results, see the main repository:
> [github.com/mehrnaz-ai/PhysioTransformer](https://github.com/mehrnaz-ai/PhysioTransformer)

-----

## Screenshot

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Lactate Threshold Predictor                                 │
├──────────────────┬──────────────────────────────────────────────┤
│  Sport: Running  │                                              │
│  Age: 34         │         Predicted Lactate Curve              │
│  Height: 1.79 m  │                                              │
│  Weight: 69 kg   │    Lactate ↑                    ● LT         │
│  HR Max: 189 bpm │          ╱─────────────────────╱            │
│  Gender: M       │         ╱                      │             │
│                  │    ────╱                   ◉◉◉◉│ attention   │
│  HR Sequence:    │                                              │
│  63,139,151,161  │         HR (bpm) →                           │
│  167,173,179,183 │                                              │
│                  │    LT HR = 168.2 bpm                        │
│  Power Sequence: │    (attention weights shown as dots)         │
│  0,12,14,16,17   │                                              │
│                  │                                              │
│ [Predict Lactate]│                                              │
└──────────────────┴──────────────────────────────────────────────┘
```

-----

## Features

- **4 sports supported**: Running, Cycling, Rowing, Kayak
- **Real-time prediction**: Enter HR and power sequence → instant LT estimate
- **Attention visualization**: See which test stages the model relies on most
- **Dark theme**: Clean, professional interface
- **Safe loading**: Clear warning if checkpoint is missing — never serves random-weight predictions
- **Pre-filled example**: Loads with a valid running test so you can try immediately

-----

## Installation

### Requirements

- Python 3.11
- See `requirements.txt`

### Setup

```bash
# 1. Clone this repository
git clone https://github.com/mehrnaz-ai/PhysioTransformer-GUI.git
cd PhysioTransformer-GUI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the pretrained checkpoint
#    → See "Checkpoint" section below

# 4. Run the GUI
python gui/final_gui.py
```

-----

## Checkpoint

The pretrained model checkpoint is required to run predictions.

**Download:**
[checkpoint_final_model.pt](https://github.com/mehrnaz-ai/PhysioTransformer-GUI/releases/latest)
(trained on all 825 samples)

**Place it in the repo root:**

```
PhysioTransformer-GUI/
├── checkpoint_final_model.pt   ← here
├── gui/
│   └── final_gui.py
└── ...
```

> **Note on checkpoint versions:**
> The checkpoint shipped here (`checkpoint_final_model.pt`) was trained on
> all 825 samples and is intended for deployment.
> The 5-fold CV checkpoints in the main research repository are for
> evaluation only — do not use them here.

-----

## How to Use

### Step 1 — Select Sport

Choose from: Running / Cycling / Rowing / Kayak

### Step 2 — Enter Athlete Info

|Field |Example|Unit  |
|------|-------|------|
|Age   |34     |years |
|Height|1.79   |meters|
|Weight|69     |kg    |
|HR Max|189    |bpm   |
|Gender|M or F |—     |

### Step 3 — Enter Test Sequences

Paste your heart rate values and power values from each test stage,
separated by commas:

```
HR Sequence:    63, 139, 151, 161, 167, 173, 179, 183, 188
Power Sequence:  0,  12,  14,  16,  17,  18,  19,  20,  21
```

Both sequences must have the **same number of values** (one per test stage, minimum 5).

### Step 4 — Click Predict

- **Yellow dot**: Predicted LT location on the lactate curve
- **Yellow dashed line**: Predicted LT heart rate
- **Magma-colored dots**: Attention weights — brighter = model relies more on this stage

-----

## Model Details

PhysioTransformer architecture:

```
Input (T stages × 14 features)
    ├── Sport Embedding (16-dim)
    ├── Transformer Encoder (4 layers, 8 heads, d=128)
    ├── Attention Pooling → stage importance weights
    └── LT Head → ordinal bin regression over [0.40, 1.00] × HRmax
```

**14 input features per stage:**
HR reserve, normalised power, HR slope, HR acceleration,
HR moving average, HR rolling SD, VO₂ proxy, cumulative fatigue,
normalised stage time + height, weight, BMI, sex, age

-----

## Limitations

- **Within-sport model**: Validated on running, cycling, and rowing.
  Kayak predictions are less reliable (see paper for details).
- **Field estimate only**: Not a clinical diagnostic tool.
  Always confirm with direct blood lactate testing for clinical decisions.
- **Minimum 5 stages**: Shorter sequences will return an error.
- **No uncertainty intervals**: The model’s internal uncertainty estimates
  are not calibrated and are therefore not shown. (See Section 4.4 of the paper.)

-----

## Paper

If you use this software in academic work, please cite:

```bibtex
@article{eskandari2025physiotransformer,
  title   = {PhysioTransformer: Sport-Aware Transformer Modeling for
             Non-Invasive Lactate Threshold Estimation Across Multiple
             Endurance Disciplines},
  author  = {Eskandari Sani, Mehrnaz and Daryanoosh, Farhad},
  journal = {[Journal Name]},
  year    = {2025},
  doi     = {[to be added after acceptance]}
}
```

Also cite the dataset:

```bibtex
@dataset{mooney2022physio,
  author    = {Mooney, Ronan and Quinlan, Leo R. and others},
  title     = {Physiological graded incremental exercise testing database},
  year      = {2022},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7693648}
}
```

-----

## Related Repository

Main research code (training, evaluation, figures):
[github.com/mehrnaz-ai/PhysioTransformer](https://github.com/mehrnaz-ai/PhysioTransformer)

-----

## Contact

**Corresponding author:**
Prof. Farhad Daryanoosh — [daryanooshf@shirazu.ac.ir](mailto:daryanoosh@shirazu.ac.ir)
Department of Sport Science, Shiraz University, Iran

-----

## License

MIT License — see <LICENSE>

The pretrained model weights are released for research and educational use only,
consistent with the terms of the original dataset (Zenodo DOI: 10.5281/zenodo.7693648).
