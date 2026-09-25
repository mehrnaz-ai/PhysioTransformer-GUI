LT-Predict GUI
Interactive Lactate Threshold Predictor — PhysioTransformer Demo
===========

An interactive desktop application for non-invasive lactate threshold (LT)
estimation using the PhysioTransformer deep learning model.
Enter heart rate and performance data from an incremental exercise test and
get an instant LT prediction with attention visualization.

--------------------------------------------------------------------------------
This GUI is a demonstration tool.
For scientific evaluation and paper results, see the main research repository:
https://github.com/mehrnaz-ai/PhysioTransformer
--------------------------------------------------------------------------------


PAPER & PREPRINT
--------------------------------------------------------------------------------
Authors: Eskandarisani M, Daryanoosh F.
Title:   Deep learning accurately estimates the 2 mmol·L⁻¹ lactate threshold 
         from heart rate and power data across endurance sports: 
         A cross-validation and transferability study.
Status:  Under Review at PLOS ONE
Preprint DOI: 10.21203/rs.3.rs-10811779/v1
              (https://doi.org/10.21203/rs.3.rs-10811779/v1)


FEATURES
--------------------------------------------------------------------------------
- 4 sports supported: Running, Cycling, Rowing, Kayak
- Real-time prediction: Enter HR and power sequence -> instant LT estimate
- Attention visualization: See which test stages the model relies on most
- Dark theme: Clean, professional interface
- Safe loading: Clear warning if checkpoint is missing (never serves random weights)
- Pre-filled example: Loads with a valid running test so you can try immediately


INSTALLATION & SETUP
--------------------------------------------------------------------------------
Requirements:
- Python 3.11
- See requirements.txt

Commands:
1. Clone this repository:
   git clone https://github.com/mehrnaz-ai/PhysioTransformer-GUI.git
   cd PhysioTransformer-GUI

2. Install dependencies:
   pip install -r requirements.txt

3. Download the pretrained checkpoint (see "CHECKPOINT" section below)

4. Run the GUI:
   python gui/final_gui.py


CHECKPOINT
--------------------------------------------------------------------------------
The pretrained model checkpoint is required to run predictions.

Download:
checkpoint_final_model.pt
(Available at: https://github.com/mehrnaz-ai/PhysioTransformer-GUI/releases/latest)

Place it in the repo root directory:

PhysioTransformer-GUI/
├── checkpoint_final_model.pt   <-- place file here
├── gui/
│   └── final_gui.py
└── ...

Note on checkpoint versions:
The checkpoint shipped here (checkpoint_final_model.pt) was trained on
all 823 quality-filtered samples and is intended for prospective deployment.
The 5-fold CV checkpoints in the main research repository are for
evaluation only — do not use them here.


HOW TO USE
--------------------------------------------------------------------------------
Step 1 — Select Sport
Choose from: Running / Cycling / Rowing / Kayak

Step 2 — Enter Athlete Info
- Age: years (e.g., 34)
- Height: meters (e.g., 1.79)
- Weight: kg (e.g., 69)
- HR Max: bpm (e.g., 189)
- Gender: M or F

Step 3 — Enter Test Sequences
Paste your heart rate values and power values from each test stage,
separated by commas:

HR Sequence:    63, 139, 151, 161, 167, 173, 179, 183, 188
Power Sequence:  0,  12,  14,  16,  17,  18,  19,  20,  21

Both sequences must have the SAME number of values (minimum 5 stages).

Step 4 — Click Predict
- Yellow dot: Predicted LT location on the lactate curve
- Yellow dashed line: Predicted LT heart rate
- Magma-colored dots: Attention weights (brighter = model relies more on stage)


MODEL DETAILS
--------------------------------------------------------------------------------
PhysioTransformer architecture:
Input (T stages × 14 features)
    ├── Sport Embedding (16-dim)
    ├── Transformer Encoder (4 layers, 8 heads, d=128)
    ├── Attention Pooling -> stage importance weights
    └── LT Head -> ordinal bin regression over [0.40, 1.00] × HRmax (soft-argmax)

14 input features per stage:
HR reserve, normalized power, HR slope, HR acceleration, HR moving average, 
HR rolling SD, VO2 proxy, cumulative fatigue, normalized stage time + 
height, weight, BMI, sex, age


LIMITATIONS
--------------------------------------------------------------------------------
- Within-sport model: Validated on running, cycling, and rowing. Kayak predictions 
  are less reliable due to smaller sample representation and upper-body 
  physiological differences (see paper for details).
- Field/Screening estimate only: Not a clinical diagnostic tool. Always confirm 
  with direct blood lactate testing for precise individual training zone 
  prescription (+/- 18.80 bpm limits of agreement).
- Minimum 5 stages: Shorter sequences will return an error.
- No uncertainty intervals: Single-network uncertainty heads are miscalibrated 
  and excluded from prediction displays (see Section 4 of paper).


CITATION
--------------------------------------------------------------------------------
If you use this software in academic work, please cite the preprint:

@article{eskandarisani2026physiotransformer,
  title   = {Deep learning accurately estimates the 2 mmol·L⁻¹ lactate threshold 
             from heart rate and power data across endurance sports: 
             A cross-validation and transferability study},
  author  = {Eskandarisani, Mehrnaz and Daryanoosh, Farhad},
  journal = {Research Square (Preprint)},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-10811779/v1}
}

Also cite the dataset:

@dataset{mooney2024physio,
  author    = {Mooney, Ronan and Quinlan, Leo R. and Corrión, Gonzalo 
               and Clarke, Gerard and Knapp, Thomas and O'Laighin, Gearóid},
  title     = {Physiological graded incremental exercise testing database (v2)},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10841412}
}


RELATED REPOSITORY
--------------------------------------------------------------------------------
Main research code (training, evaluation, figures, ablations):
https://github.com/mehrnaz-ai/PhysioTransformer


CONTACT
--------------------------------------------------------------------------------
Corresponding author:
Mehrnaz Eskandarisani
Email: mehrnazeskandarisani1@gmail.com


LICENSE
--------------------------------------------------------------------------------
This project is licensed under the MIT License — see LICENSE file.

The pretrained model weights are released for research and educational use only,
consistent with the terms of the original dataset (Zenodo DOI: 10.5281/zenodo.10841412).
