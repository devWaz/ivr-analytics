# Project 4 (Group 4): IVR Call-Centre Analytics

**PAU MSc Data Science · DAT613 Python Programming for Data Science · Group Project: NumPy Pipelines**

Afribank Plus keeps only the **audio recordings** of its phone-menu (IVR) calls, not the keys callers pressed.
This project decodes the keypad presses from the audio and answers two business questions:

1. Which keys were pressed in each call?
2. How many callers chose each main-menu option first, how many asked for an agent, and what does this tell the bank?

Built with **NumPy**, `json` and **matplotlib** (charts) only. No pandas, SciPy or scikit-learn.

---

## Quick start

Run the script and give it the path of the `.npz` data file. There is no need to open or edit any code.

### Terminal / command prompt

```bash
python ivr_pipeline.py path/to/data.npz
```

Use quotes if the path contains spaces:

```bash
python ivr_pipeline.py "my folder/data.npz"
```

### Google Colab

Upload `ivr_pipeline.py` and the data file (folder icon on the left), then run:

```python
!python ivr_pipeline.py /content/data.npz
```

To show the charts inside the notebook instead:

```python
from ivr_pipeline import run_pipeline
results = run_pipeline("/content/data.npz")
```

### Options

| Option | What it does | Default |
|---|---|---|
| `--out FOLDER` | Folder where the two chart PNGs are saved | `ivr_outputs` |
| `--no-plots` | Skip drawing and saving charts | charts on |

---

## What it prints

1. **Data problems found**, with call numbers and evidence (different sample rate, offset and clipping, held key).
2. **Decoded keys for every call** and how each call ended.
3. **Effect of each fix**: how the results change if a fix is left out.
4. **SUMMARY** of the two business questions: first-choice counts and percentages, calls that asked for an agent, and how calls ended.

Two charts are saved in `ivr_outputs/`:

| File | Shows |
|---|---|
| `chart1_tones_and_keys.png` | Tone strengths over time for one call, with the decoded key underneath |
| `chart2_services_outcomes.png` | First menu choices and how calls ended |

If the file is missing or is not an IVR data file, a one-line `ERROR:` message is shown instead of a long traceback.

### Example of the SUMMARY block

```text
==============================================================
SUMMARY (answers to the two business questions)
==============================================================
Q1  Decoded keys: see the list of calls above.
Q2  First main-menu choice:
      Balance       6  (20%)
      Transfers    11  (37%)
      Cards         7  (23%)
      Loans         5  (17%)
      Agent         1  (3%)
    Asked for an agent at any point: 7 of 30 (23%)
    How calls ended: abandoned in amount 3, abandoned in sub 5, agent 7, service completed 12, transfer completed 3
```

---

## How it works

Each key on a phone plays two tones at once (one for its row, one for its column), so a key can be read from which two of seven frequencies are loud.

| Step | What happens | Main NumPy tools |
|---|---|---|
| 1. Load | Open the `.npz` file and check shapes | `np.load`, `assert` |
| 2. Split | Cut the long recording into separate calls | `np.split` |
| 3. Diagnose and clean | Find the data problems; remove the offset | `np.unique`, `np.median` |
| 4. Measure tones | Cut each call into 30 ms frames (every 10 ms) and measure the 7 keypad frequencies | `sliding_window_view`, `np.fft.rfft` |
| 5. Detect keys | Strongest row and column tones, compared with the call's background level | `argmax`, `np.where` |
| 6. Key presses | One run of identical frames is one press | `np.diff`, `np.flatnonzero` |
| 7. Menu rules | Follow the menu to find each caller's journey | small loop over each call's keys |
| 8. Count | Count first choices, agent requests and call endings | `np.unique` |

`run_pipeline(path)` runs all steps. It reads the number of calls, sample rates and offsets from the file, so it works on any file of the same type.

---

## Data problems handled

| Problem | Fix |
|---|---|
| One call recorded at a different sample rate | Use each call's own sample rate |
| A loud call with a large constant offset and clipping | Subtract the call's median |
| A key held down for a long time | Count a run of identical frames as one key press |

---

## Repository contents

| File | Purpose |
|---|---|
| `ivr_pipeline.py` | The script and the `run_pipeline(path)` function |
| `index.ipynb` | Step-by-step notebook version with comments |
| `make_test_data.py` | Builds a synthetic test file with known keys (testing only) |
| `README.md` | This file |

> **Note:** the data file (`group04_ivr.npz`) is **not** included in this repository. Pass its path when running the script.

---

## Requirements

- Python 3.9 or newer
- NumPy 1.20 or newer
- matplotlib (for charts only)

```bash
pip install numpy matplotlib
```

---

## Team

| Member | Part |
|---|---|
| Fawaz Odesanya | Data and structure; data problems |
| Emeka NWAOSA | Tone analysis and key decoding |
| Okechukwu Ifegwu | Menu logic and business answers |
| Temitope Omolade OBA | `run_pipeline`, report and submission |
