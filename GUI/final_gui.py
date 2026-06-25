

import os
import sys
import traceback

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from model_full import (          
    PhysioTransformerFull,
    DEVICE,
    SPORT_TO_IDX,
    SPORT_POWER_MAX,
)


class _GuiInferenceModel(PhysioTransformerFull):


    def forward(self, x, L, sport):
        B, T, _ = x.shape
        emb = self.sport_emb(sport).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x, emb], dim=-1)

        z = self.pos(self.inp(x))
        mask = torch.arange(T, device=x.device)[None, :] >= L[:, None]
        h = self.encoder(z, src_key_padding_mask=mask)
        h = self.repr(h)

        score = self.attn_pool(h).masked_fill(mask.unsqueeze(-1), -1e9)
        w = torch.softmax(score, dim=1)
        h_global = (w * h).sum(dim=1)

        curve = self.curve_head(h)

        lt_logits = self.lt_bins(h_global)
        lt_prob = torch.softmax(lt_logits, dim=-1)
        lt_pred = (lt_prob * self.lt_bin_centers).sum(dim=-1)


        return curve, lt_pred, w


# DARK THEME

BG = "#111827"
CARD = "#1f2937"
FG = "#f9fafb"
ACCENT = "#60a5fa"
WARNING_COLOR = "#f87171"

SPORT_COLORS = {
    "running": "#ef4444",
    "cycling": "#3b82f6",
    "rowing": "#10b981",
    "kayak": "#f59e0b",
}

sport_to_idx = SPORT_TO_IDX


# CHECKPOINT RESOLUTION


CHECKPOINT_NAME = "checkpoint_final_model.pt"
CHECKPOINT_CANDIDATES = [
    os.path.join(REPO_ROOT, CHECKPOINT_NAME),
    os.path.join(SCRIPT_DIR, CHECKPOINT_NAME),
    CHECKPOINT_NAME,  # fall back to whatever the current working dir is
]


def find_checkpoint():
    for path in CHECKPOINT_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


model = _GuiInferenceModel(input_dim=14).to(DEVICE)
MODEL_READY = False
MODEL_STATUS_MSG = ""

ckpt_path = find_checkpoint()
if ckpt_path is None:
    MODEL_STATUS_MSG = (
        f"No checkpoint found ({CHECKPOINT_NAME}). Looked in:\n"
        + "\n".join(f"  - {p}" for p in CHECKPOINT_CANDIDATES)
        + "\nPredictions are disabled until a checkpoint is available."
    )
    print(f"WARNING: {MODEL_STATUS_MSG}")
else:
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        MODEL_READY = True
        MODEL_STATUS_MSG = f"Model loaded: {os.path.relpath(ckpt_path, REPO_ROOT)}"
        print(MODEL_STATUS_MSG)
    except Exception as e:
        MODEL_STATUS_MSG = f"Failed to load checkpoint at {ckpt_path}: {e}"
        print(f"WARNING: {MODEL_STATUS_MSG}")

model.eval()


# HELPERS & PREDICTION LOGIC

def clean(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def parse_sequence(text):
    text = text.replace("\n", ",").replace(";", ",").replace("،", ",")
    vals = [float(x.strip()) for x in text.split(",") if x.strip() != ""]
    return vals


def safe_num(x):
    try:
        x = str(x).strip()
        return float(x) if x != "" else 0.0
    except Exception:
        return 0.0


def safe_array(seq):
    cleaned = [float(str(x).strip()) for x in seq if str(x).strip() != ""]
    return np.array(cleaned, dtype=np.float32)


def real_model_predict(hr_seq, power_seq, age, height, weight, hrmax, sport, gender):
   
    if not MODEL_READY:
        raise RuntimeError(MODEL_STATUS_MSG or "Model is not loaded.")

    hr = safe_array(hr_seq)
    power = safe_array(power_seq)

    if len(hr) < 5:
        raise ValueError("HR sequence too short")
    if len(power) != len(hr):
        raise ValueError("HR and Power length mismatch")

    if hrmax <= 0:
        hrmax = np.max(hr)

    # Calculated identically to models/final_model.py's feature pipeline
    hr_rest = float(np.min(hr))
    hr_norm = hr / (hrmax + 1e-6)
    hr_reserve = np.clip((hr - hr_rest) / (hrmax - hr_rest + 1e-6), 0.0, 1.0)

    power = clean(power) / SPORT_POWER_MAX.get(sport, 1.0)

    t = np.linspace(0.0, 1.0, len(hr))
    hr_slope = np.gradient(hr_norm)
    hr_accel = np.gradient(hr_slope)
    hr_ma = pd.Series(hr_norm).rolling(3, min_periods=1).mean().values
    hr_std = pd.Series(hr_norm).rolling(3, min_periods=1).std().fillna(0.0).values
    vo2_proxy = hr_reserve * power
    fatigue = np.cumsum(hr_reserve) / len(hr)

    # Static features — GUI height is in meters; model expects cm / 200.0
    height_cm = height * 100.0
    h = height_cm / 200.0
    w = weight / 120.0
    bmi = (weight / (height ** 2)) / 40.0  # approximation 


    age = np.clip(age, 10, 80) / 80.0

    static = np.tile([h, w, bmi, gender, age], (len(hr), 1))

    # X shape: (T, 14)
    X = np.column_stack([
        hr_reserve, power, hr_slope, hr_accel,
        hr_ma, hr_std, vo2_proxy, fatigue, t, static
    ])
    X = clean(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    L = torch.tensor([len(hr)]).to(DEVICE)

    sport = str(sport).strip().lower()
    sport_tensor = torch.tensor([sport_to_idx.get(sport, 4)], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        curve, lt_pred, attn = model(X_tensor, L, sport_tensor)

    # Processing Outputs
    pred_curve = torch.expm1(curve[0, :, 0].clamp(-3.0, 4.0)).cpu().numpy()

    # Smooth curve for visual clarity
    pred_curve = np.maximum.accumulate(pred_curve)
    pred_curve = pd.Series(pred_curve).rolling(3, center=True, min_periods=1).mean().values


    lt_hr = lt_pred.item() * hrmax
    lt_hr = np.clip(lt_hr, 0.70 * hrmax, 0.95 * hrmax)
    lt_idx = np.argmin(np.abs(hr - lt_hr))

    attention = attn[0, :len(hr), 0].cpu().numpy()

    return pred_curve,  lt_hr, lt_idx, attention



# GUI SETUP 

root = tk.Tk()
root.title("AI Lactate Threshold Predictor")
root.geometry("1400x950")
root.configure(bg=BG)

style = ttk.Style()
style.theme_use("clam")

left = tk.Frame(root, bg=CARD, width=350)
left.pack(side="left", fill="y")

title = tk.Label(left, text="AI Lactate Threshold", bg=CARD, fg=FG, font=("Segoe UI", 20, "bold"))
title.pack(pady=20)


status_label = tk.Label(
    left,
    text=(MODEL_STATUS_MSG if MODEL_READY else f"⚠ {MODEL_STATUS_MSG}"),
    bg=CARD,
    fg=(FG if MODEL_READY else WARNING_COLOR),
    font=("Segoe UI", 9),
    wraplength=310,
    justify="left",
)
status_label.pack(fill="x", padx=20, pady=(0, 10))

sport_var = tk.StringVar(value="running")
tk.Label(left, text="Sport", bg=CARD, fg=FG, font=("Segoe UI", 11)).pack(anchor="w", padx=20)
sport_box = ttk.Combobox(left, textvariable=sport_var, values=["running", "cycling", "rowing", "kayak"], state="readonly")
sport_box.pack(fill="x", padx=20, pady=8)

entries = {}
for field in ["Age", "Height (m)", "Weight (kg)", "HR Max(bpm)", "Gender (M/F)"]:
    tk.Label(left, text=field, bg=CARD, fg=FG, font=("Segoe UI", 11)).pack(anchor="w", padx=20)
    e = tk.Entry(left, bg="#374151", fg=FG, insertbackground=FG, relief="flat", font=("Segoe UI", 11))
    e.pack(fill="x", padx=20, pady=8)
    entries[field] = e

tk.Label(left, text="Heart Rate Sequence", bg=CARD, fg=FG, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(20, 5))
entries["HR Sequence"] = tk.Text(left, height=5, bg="#374151", fg=FG, insertbackground=FG, relief="flat", font=("Consolas", 10))
entries["HR Sequence"].pack(fill="x", padx=20, pady=8)

tk.Label(left, text="Actual Power Sequence", bg=CARD, fg=FG, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=20, pady=(20, 5))
entries["Power Sequence"] = tk.Text(left, height=5, bg="#374151", fg=FG, insertbackground=FG, relief="flat", font=("Consolas", 10))
entries["Power Sequence"].pack(fill="x", padx=20, pady=8)

entries["Age"].insert(0, "34")
entries["Height (m)"].insert(0, "1.79")
entries["Weight (kg)"].insert(0, "69")
entries["HR Max(bpm)"].insert(0, "189")
entries["Gender (M/F)"].insert(0, "M")
entries["HR Sequence"].insert("1.0", "63,139,151,161,167,173,179,183,188")
entries["Power Sequence"].insert("1.0", "0,12,14,16,17,18,19,20,21")

right = tk.Frame(root, bg=BG)
right.pack(side="right", fill="both", expand=True)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().pack(fill="both", expand=True)



# PREDICT CALLBACK

def predict():
    if not MODEL_READY:
        messagebox.showerror(
            "Model Not Loaded",
            "Prediction is disabled because no trained checkpoint could be loaded.\n\n"
            + MODEL_STATUS_MSG,
        )
        return

    try:
        hr_seq = parse_sequence(entries["HR Sequence"].get("1.0", "end"))
        power_seq = parse_sequence(entries["Power Sequence"].get("1.0", "end"))

        if len(hr_seq) != len(power_seq):
            messagebox.showerror("Error", "HR and Power sequences must have same length.")
            return

        sport = sport_var.get()
        age = safe_num(entries["Age"].get())
        height = safe_num(entries["Height (m)"].get())
        hrmax = safe_num(entries["HR Max(bpm)"].get())
        weight = safe_num(entries["Weight (kg)"].get())
        gender_text = entries["Gender (M/F)"].get().strip().lower()

        if gender_text == "m":
            gender = 1.0
        elif gender_text == "f":
            gender = 0.0
        else:
            gender = 0.5

        pred, lt_hr, lt_idx, attention = real_model_predict(
            hr_seq=hr_seq, power_seq=power_seq, age=age, height=height,
            hrmax=hrmax, weight=weight, sport=sport, gender=gender
        )

        ax.clear()
        ax.set_facecolor(CARD)

        color = SPORT_COLORS[sport]
        x = np.array(hr_seq)

        # Sort arrays to plot correctly
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        pred = np.array(pred)[sort_idx]
        attention = np.array(attention)[sort_idx]

        ax.plot(x, pred, linewidth=3, color=color, label="Predicted Lactate")

        lt_lactate = pred[lt_idx]
        ax.scatter([lt_hr], [lt_lactate], s=180, color="#fbbf24", edgecolors="black", linewidths=2, zorder=10)
        ax.axvline(lt_hr, linestyle="--", linewidth=2, color="#fbbf24", label=f"LT HR = {lt_hr:.1f}")


        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-6)
        ax.scatter(x, pred * 0.15, c=attention, cmap="magma", s=140, alpha=0.95)

        ax.set_xlabel("Heart Rate (bpm)", fontsize=12, color=FG)
        ax.set_ylabel("Blood Lactate (mmol/L)", fontsize=12, color=FG)
        ax.tick_params(colors=FG)

        for spine in ax.spines.values():
            spine.set_color("#6b7280")

        leg = ax.legend()
        for text in leg.get_texts():
            text.set_color(FG)
        leg.get_frame().set_facecolor(CARD)

        canvas.draw()

    except Exception as e:
        traceback.print_exc()
        messagebox.showerror("Prediction Error", str(e))


btn = tk.Button(
    left, text="Predict Lactate", command=predict, bg=ACCENT, fg="black",
    relief="flat", font=("Segoe UI", 13, "bold"), pady=10
)
btn.pack(fill="x", padx=20, pady=25)

if not MODEL_READY:
    btn.config(state="disabled", text="Predict Lactate (model unavailable)")
else:
    predict()  # populate the initial chart with the default example values

root.mainloop()
