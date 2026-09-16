# Predicting Viewing Tasks from Eye Movements in Dynamic Scenes

Code for my Master's thesis in Human Factors at TU Berlin (2026).

> \\\\\\\*\\\\\\\*Thesis:\\\\\\\*\\\\\\\* \\\\\\\*Predicting Tasks from Eye-Tracking Data Using Machine Learning\\\\\\\*
> \\\\\\\*\\\\\\\*Author:\\\\\\\*\\\\\\\* Jonas Ebert · \\\\\\\*

## Overview

Can we tell *what someone is trying to do* just from how their eyes move?
Participants watched dynamic video scenes under four different viewing tasks:

|Task|Instruction|
|-|-|
|**Free viewing**|Watch the scene without a specific goal|
|**Search**|Find a target in the scene|
|**Memorization**|Remember the scene|
|**Balance**|Hybrid task combining search and memorization|

From the raw eye-tracking events (foveations, saccades, blinks, pupil size) this
project engineers \~60 trial-level features and trains several machine-learning
classifiers to predict the task. SHAP values are used to interpret which eye-movement
features drive the predictions.

The analysis consists of three experiments:

1. **Multiclass** – distinguish all four tasks.
2. **Binary** – distinguish search vs. memorization.
3. **Balance transfer** – apply the binary model zero-shot to the (unseen) balance
task, comparing the first and second half of each trial to reveal shifts in
viewing strategy over time.

## Key results

|Experiment|Best model|Accuracy|
|-|-|-|
|1 – Multiclass (4 tasks)|Logistic regression|\~54 % (chance ≈ 25 %)|
|2 – Search vs. memorization|Logistic regression|\~76 % (chance ≈ 50 %)|
|3 – Balance transfer|Logistic regression (from Exp. 2)|Shift in predicted strategy between first and second half of the trial|

Logistic regression generalised best among the five tested classifiers
(logistic regression, AdaBoost, HistGradientBoosting, LightGBM, MLP).

## Repository structure

```
├── 00\\\\\\\_preprocess.py      # Step 0: cleaning, splitting, feature engineering, scaling
├── 01\\\\\\\_multiclass.py      # Experiment 1: all four tasks
├── 02\\\\\\\_binary.py          # Experiment 2: search vs. memorization (saves LR model)
├── 03\\\\\\\_balance.py         # Experiment 3: zero-shot transfer to balance trials
├── src/
│   ├── config.py         # Paths, random seed, experiment modes
│   ├── preprocessing.py  # Event-level cleaning, outliers, KNN imputation
│   ├── features.py       # Trial-level feature engineering
│   └── modeling.py       # Models, hyperparameter grids, evaluation, SHAP
├── data/                 # Raw and processed data (not included, see data/README.md)
├── results/              # Generated figures and models (created when running)
└── requirements.txt
```

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/<username>/eye-tracking-task-classification.git
cd eye-tracking-task-classification

python -m venv .venv
# Windows:      .venv\\\\\\\\Scripts\\\\\\\\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

The data are not publicly available (see [`data/README.md`](data/README.md)).
With the raw data in place, run the scripts from the repository root in this order:

```bash
# Experiment 1
python 00\\\\\\\_preprocess.py --mode all
python 01\\\\\\\_multiclass.py

# Experiment 2
python 00\\\\\\\_preprocess.py --mode search\\\\\\\_vs\\\\\\\_memorization
python 02\\\\\\\_binary.py

# Experiment 3 (uses the model saved by Experiment 2)
python 00\\\\\\\_preprocess.py --mode balance\\\\\\\_classification
python 03\\\\\\\_balance.py
```

Metrics are printed to the console; figures are saved to `results/figures/`.

## Method

### Preprocessing (event level)

* Exclusion of incorrectly answered trials, the first five (practice) trials of each
task block, trials with fewer than 10 events and trials containing a foveation
longer than 5 s
* Pupil size expressed relative to each participant's mean pupil size
* Train/test split: 80/20 per participant × task (Exp. 1 and 2); in Exp. 3 the model is
trained on search and memorization and tested on balance trials only
* Implausible velocities set to missing; outliers winsorized at 1.5 × IQR
(bounds learned on the training set)
* Missing values imputed per trial with a KNN imputer (k = 3)

### Features (trial level)

|Group|Examples|
|-|-|
|Foveations|count, duration statistics, velocity, saliency|
|Saccades|amplitude, velocity, horizontal/vertical angle, scanpath length|
|Pupil|median, min, max (relative to baseline)|
|Semantic|foveations per object category, share of foveations on engaging objects, number of unique objects|
|Foveation categories|counts and mean durations per category, first gaze duration|
|Temporal|means per time bin, variance in the first bin, linear slope over the trial|
|Spatial|scanned area|
|Other|blink count|

Features are scaled with a `RobustScaler` fitted on the training set.

### Models

Logistic regression, AdaBoost, HistGradientBoosting, LightGBM and an MLP, each tuned
with 5-fold grid search on balanced accuracy. The logistic regression is interpreted with
SHAP (`LinearExplainer`).

## Reproducibility

The train/test split uses a fixed random seed (`src/config.py`). The results reported in
the thesis were computed with an unseeded split, so rerunning the code may give slightly
different numbers.

