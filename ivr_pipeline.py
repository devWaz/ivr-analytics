"""
Group 4 - Banking: Call-Centre IVR Analytics (NumPy only + json).

Pipeline stages and array shapes (n_calls = 30 for the given file):
    raw        audio (N,)                      int16, all calls joined end to end
    reshape    list of n_calls 1-D arrays      lengths differ -> ragged, cannot be one 2-D array
    clean      per call (len_i,)               float64, median removed
    features   per call (n_frames_i, 7)        tone strength at the 7 keypad frequencies
    decode     per call (n_frames_i,)          key index 0..11, or -1 = no tone
    presses    per call (n_presses_i,)         one entry per key press
    answers    (n_calls,) arrays + counts      first choice, outcome, agent flag

Usage:  results = run_pipeline("group04_ivr.npz")
"""
import os
import sys
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FREQS = np.array([697, 770, 852, 941, 1209, 1336, 1477], dtype=float)
KEYPAD = np.array([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["*", "0", "#"]])
KEYS_FLAT = KEYPAD.ravel()                       # index = row*3 + col
MAIN_NAMES = {"1": "Balance", "2": "Transfers", "3": "Cards", "4": "Loans", "0": "Agent"}
FRAME_MS, HOP_MS = 30, 10


# ----------------------------------------------------------------- stage 1: load + structure
def load_data(path):
    """Read the .npz and validate its structure.

    (file) -> audio (N,), offsets (n_calls+1,), fs (n_calls,), menu (dict)
    """
    path = os.fspath(path)                       # accepts str or pathlib.Path
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path!r}. Check the path (in Colab, upload the file first).")
    d = np.load(path)
    missing = [k for k in ("audio", "call_offsets", "sample_rate_per_call", "menu_json") if k not in d.files]
    if missing:
        raise ValueError(f"{path!r} does not look like an IVR data file: missing array(s) {missing}.")
    audio = d["audio"]
    offsets = d["call_offsets"].astype(np.int64)
    fs = d["sample_rate_per_call"].astype(np.int64)
    raw = np.asarray(d["menu_json"]).item()
    menu = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
    assert audio.ndim == 1, audio.shape
    assert offsets.shape == (fs.size + 1,), (offsets.shape, fs.size)
    assert offsets[0] == 0 and offsets[-1] <= audio.size
    assert np.all(np.diff(offsets) > 0), "call offsets must be strictly increasing"
    return audio, offsets, fs, menu


def split_calls(audio, offsets):
    """Cut the long stream into calls.

    (N,), (n_calls+1,) -> list of n_calls arrays, each (offsets[i+1]-offsets[i],)
    """
    calls = np.split(audio[:offsets[-1]], offsets[1:-1])
    assert len(calls) == offsets.size - 1
    assert sum(c.size for c in calls) == offsets[-1]
    return calls


# ----------------------------------------------------------------- stage 2: data problems
def diagnose(calls, fs):
    """Find the three problems and return the evidence.

    calls: list of (len_i,) int16 ; fs: (n_calls,)
    -> dict of (n_calls,) arrays plus the modal sample rate
    """
    vals, counts = np.unique(fs, return_counts=True)
    modal_fs = int(vals[counts.argmax()])
    med = np.array([np.median(c) for c in calls])
    sd = np.array([c.astype(np.float64).std() for c in calls])
    clip = np.array([np.mean((c == c.max()) | (c == c.min())) for c in calls])
    assert med.shape == sd.shape == clip.shape == fs.shape
    return dict(modal_fs=modal_fs, odd_rate=np.flatnonzero(fs != modal_fs),
                median=med, std=sd, clip_frac=clip,
                offset_calls=np.flatnonzero(np.abs(med) > 0.25 * sd),
                clipped_calls=np.flatnonzero(clip > 0.005))


def clean_call(x, remove_offset=True):
    """(len,) int16 -> (len,) float64 with the constant offset (median) removed."""
    x = x.astype(np.float64)
    out = x - np.median(x) if remove_offset else x
    assert out.shape == x.shape
    return out


# ----------------------------------------------------------------- stage 3: analysis
def tone_strengths(x, fs, frame_ms=30, hop_ms=10):
    """(n,) -> (frames, 7)   [starter function from the brief, unchanged]
    columns: 697, 770, 852, 941, 1209, 1336, 1477 Hz"""
    freqs = np.array([697, 770, 852, 941, 1209, 1336, 1477])
    n = int(fs * frame_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    frames = sliding_window_view(x, n)[::hop]
    frames = frames * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    bins = np.fft.rfftfreq(n, 1 / fs)
    dist = np.abs(bins[None, :] - freqs[:, None])
    return spectrum[:, dist.argmin(axis=1)]


def detect_keys(S, k=6.0):
    """Frame-level key detection.

    S (frames, 7) -> key index (frames,) in 0..11, or -1 when no tone.
    A frame holds a tone if the strongest ROW frequency AND the strongest COLUMN
    frequency are both > k x the call's background level.
    Background = median over frames of the per-frame median of the 7 strengths
    (a frame has at most 2 tone bins, so the 4th-largest value is always background).
    """
    assert S.ndim == 2 and S.shape[1] == 7
    rows, cols = S[:, :4], S[:, 4:]
    floor = max(np.median(np.median(S, axis=1)), 1e-6)
    active = (rows.max(1) > k * floor) & (cols.max(1) > k * floor)
    idx = np.where(active, rows.argmax(1) * 3 + cols.argmax(1), -1)
    assert idx.shape == (S.shape[0],)
    return idx


def runs_to_presses(idx, min_frames=3):
    """Collapse runs of identical consecutive frames into one key press.

    idx (frames,) -> key_index (p,), length_in_frames (p,)
    Runs shorter than min_frames are transition glitches and are dropped.
    A held key is ONE long run, so it stays one press.
    """
    if idx.size == 0:
        return np.empty(0, int), np.empty(0, int)
    change = np.flatnonzero(idx[1:] != idx[:-1]) + 1
    starts = np.r_[0, change]
    lens = np.diff(np.r_[starts, idx.size])
    keep = (idx[starts] >= 0) & (lens >= min_frames)
    return idx[starts][keep], lens[keep]


def decode_call(x_int, fs, remove_offset=True):
    """(len,) int16, fs -> dict(S (frames,7), idx (frames,), keys str, lens (p,))"""
    x = clean_call(x_int, remove_offset)
    if x.size < int(fs * FRAME_MS / 1000):
        return dict(S=np.empty((0, 7)), idx=np.empty(0, int), keys="", lens=np.empty(0, int))
    S = tone_strengths(x, fs, FRAME_MS, HOP_MS)
    idx = detect_keys(S)
    p, lens = runs_to_presses(idx)
    return dict(S=S, idx=idx, keys="".join(KEYS_FLAT[p]), lens=lens)


# ----------------------------------------------------------------- stage 4: menu logic
def interpret_call(keys, menu=None):
    """Apply the menu rules to one key string (a small state machine: each key's meaning
    depends on the state, so this part is inherently sequential; 30 short strings only).

    keys: str of pressed keys ; menu: dict from menu_json (None -> brief's default rules)
    MAIN : 0 -> agent (end); a key with a sub-menu in the JSON (2, 3, 4) -> SUB;
           a main option with NO sub-menu in the JSON (1 Balance) -> service completed (end)
    SUB  : 9 -> back to MAIN, 0 -> agent (end), a listed option -> chosen (end),
           except Transfers (2) where the option starts AMOUNT entry
    AMOUNT: every digit is part of the amount (0 is NOT 'agent' here), '#' ends
    Keys not in the menu are ignored. If the keys run out mid-way the outcome is
    'abandoned_in_<state>'.
    """
    main_opts = set(menu[""]) if menu and "" in menu else set("12340")
    subs = {k: set(v) for k, v in menu.items() if k != ""} if menu else {k: set("12389") for k in "234"}
    state, menu_id, first, agent = "MAIN", None, None, False
    outcome, service, amount, done = None, None, "", False
    for ch in keys:
        if state == "MAIN":
            if ch not in main_opts:
                continue
            if first is None:
                first = ch
            if ch == "0":
                agent, outcome, done = True, "agent", True
            elif ch in subs:
                menu_id, state = ch, "SUB"
            else:
                service, outcome, done = (ch, None), "service_completed", True
        elif state == "SUB":
            if ch == "9":
                state = "MAIN"
            elif ch == "0":
                agent, outcome, done = True, "agent", True
            elif ch in subs[menu_id]:
                service = (menu_id, ch)
                if menu_id == "2":
                    state = "AMOUNT"
                else:
                    outcome, done = "service_completed", True
        else:  # AMOUNT
            if ch == "#":
                outcome, done = "transfer_completed", True
            elif ch.isdigit():
                amount += ch
        if done:
            break
    if outcome is None:
        outcome = "no_selection" if state == "MAIN" else f"abandoned_in_{state.lower()}"
    return dict(first=first, agent=agent, outcome=outcome, service=service,
                amount=int(amount) if amount and outcome == "transfer_completed" else None)


def summarise(key_strs, menu=None):
    """list of n_calls key strings -> counts for the two business questions."""
    res = [interpret_call(s, menu) for s in key_strs]
    first = np.array([r["first"] or "none" for r in res])
    agent = np.array([r["agent"] for r in res])
    outcome = np.array([r["outcome"] for r in res])
    f_vals, f_cnt = np.unique(first, return_counts=True)
    o_vals, o_cnt = np.unique(outcome, return_counts=True)
    assert f_cnt.sum() == len(key_strs) == o_cnt.sum()
    return dict(per_call=res, first=dict(zip(f_vals.tolist(), f_cnt.tolist())),
                outcome=dict(zip(o_vals.tolist(), o_cnt.tolist())),
                agent_calls=int(agent.sum()))


# ----------------------------------------------------------------- effect of each fix
def fix_effects(calls, fs, diag, baseline_keys, menu=None):
    """Re-decode WITHOUT each fix and report how the results change."""
    no_off = [decode_call(c, int(r), remove_offset=False)["keys"] for c, r in zip(calls, fs)]
    wrong_fs = [decode_call(c, diag["modal_fs"] if i in diag["odd_rate"] else int(r))["keys"]
                for i, (c, r) in enumerate(zip(calls, fs))]
    out = {}
    for name, ks in (("offset removal", no_off), ("the per-call sample rate", wrong_fs)):
        changed = [i for i, (a, b) in enumerate(zip(baseline_keys, ks)) if a != b]
        out[name] = dict(calls_changed=changed, summary=summarise(ks, menu))
    return out


# ----------------------------------------------------------------- charts
def _in_notebook():
    """True inside Jupyter/Colab, where charts can be shown inline."""
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def make_charts(calls, fs, decoded, summ, menu, out_dir, show=None):
    """Save two PNG charts to out_dir; also display them inline when run in a notebook."""
    import matplotlib.pyplot as plt                 # imported here so the rest works without matplotlib
    show = _in_notebook() if show is None else show
    os.makedirs(out_dir, exist_ok=True)

    # Chart 1: what the decoder sees in one call
    i = int(np.argmax([len(d["keys"]) for d in decoded]))
    S, idx = decoded[i]["S"], decoded[i]["idx"]
    t = np.arange(S.shape[0]) * HOP_MS / 1000
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[3, 1.4]))
    for j, f in enumerate(FREQS):
        ax[0].plot(t, S[:, j], lw=1, label=f"{int(f)} Hz")
    ax[0].set_ylabel("tone strength"); ax[0].legend(ncol=7, fontsize=6, loc="upper right")
    ax[0].set_title(f"Call {i}: tone strengths and decoded keys")
    ax[1].step(t, idx, where="post", color="k", lw=1)
    ax[1].set_yticks(range(12)); ax[1].set_yticklabels(KEYS_FLAT, fontsize=6)
    ax[1].set_ylabel("key"); ax[1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/chart1_tones_and_keys.png", dpi=150); plt.show() if show else plt.close(fig)

    # Chart 2: services + outcomes
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    order = ["1", "2", "3", "4", "0"]
    ax[0].bar([MAIN_NAMES[k] for k in order], [summ["first"].get(k, 0) for k in order],
              color="#2a6f97")
    ax[0].set_title("First main-menu choice"); ax[0].set_ylabel("callers")
    ax[0].tick_params(axis="x", labelrotation=30)
    o = summ["outcome"]
    ax[1].bar(list(o.keys()), list(o.values()), color="#52b788")
    ax[1].set_title(f"How calls end (agent asked at any point: {summ['agent_calls']})")
    ax[1].tick_params(axis="x", labelrotation=30)
    fig.tight_layout(); fig.savefig(f"{out_dir}/chart2_services_outcomes.png", dpi=150); plt.show() if show else plt.close(fig)


# ----------------------------------------------------------------- one function runs everything
def run_pipeline(path, out_dir="ivr_outputs", plot=True, verbose=True, show=None):
    """Run the whole analysis on any file of this type. Returns a dict of results."""
    audio, offsets, fs, menu = load_data(path)
    calls = split_calls(audio, offsets)
    diag = diagnose(calls, fs)
    decoded = [decode_call(c, int(r)) for c, r in zip(calls, fs)]
    key_strs = [d["keys"] for d in decoded]
    summ = summarise(key_strs, menu)
    fx = fix_effects(calls, fs, diag, key_strs, menu)

    all_lens = np.concatenate([d["lens"] for d in decoded]) if decoded else np.empty(0)
    med_len = np.median(all_lens) if all_lens.size else 0
    held = [i for i, d in enumerate(decoded) if d["lens"].size and d["lens"].max() > 4 * med_len]

    if plot:
        try:
            make_charts(calls, fs, decoded, summ, menu, out_dir, show)
        except Exception as e:                       # a chart problem must never stop the analysis
            print("Charts skipped:", repr(e))

    if verbose:
        print(f"Calls: {len(calls)}   raw audio shape: {audio.shape}   menu: {menu}")
        print(f"Modal sample rate {diag['modal_fs']} Hz | odd-rate calls: {diag['odd_rate'].tolist()} "
              f"({[int(fs[i]) for i in diag['odd_rate']]} Hz)")
        print(f"Offset calls: {diag['offset_calls'].tolist()} "
              f"(median {[int(diag['median'][i]) for i in diag['offset_calls']]})")
        print(f"Clipped calls: {diag['clipped_calls'].tolist()} "
              f"(clip frac {[round(float(diag['clip_frac'][i]), 3) for i in diag['clipped_calls']]})")
        print(f"Held-key calls: {held} (median press = {med_len*HOP_MS:.0f} ms, "
              f"longest = {[int(decoded[i]['lens'].max()*HOP_MS) for i in held]} ms)")
        print("\nDecoded keys per call:")
        for i, (s, r) in enumerate(zip(key_strs, summ["per_call"])):
            print(f"  call {i:2d}: {s:<22s} first={r['first']} outcome={r['outcome']} amount={r['amount']}")
        print("\nFirst choices:", summ["first"], "\nOutcomes:", summ["outcome"],
              "\nCalls asking for an agent at any point:", summ["agent_calls"])
        for name, e in fx.items():
            print(f"Without {name}: {len(e['calls_changed'])} calls decode differently "
                  f"{e['calls_changed']} -> first choices {e['summary']['first']}")
        n = len(key_strs)
        names = menu.get("", {})
        print("\n" + "=" * 62 + "\nSUMMARY (answers to the two business questions)\n" + "=" * 62)
        print("Q1  Decoded keys: see the list of calls above.")
        print("Q2  First main-menu choice:")
        for k in list(names) + ["none"]:
            if k in summ["first"]:
                v = summ["first"][k]
                print(f"      {names.get(k, 'no valid key'):<12}{v:>3}  ({100 * v / n:.0f}%)")
        print(f"    Asked for an agent at any point: {summ['agent_calls']} of {n} ({100 * summ['agent_calls'] / n:.0f}%)")
        print("    How calls ended:", ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in summ["outcome"].items()))
    return dict(diag=diag, decoded=decoded, keys=key_strs, summary=summ, fix_effects=fx, held=held)


def main(argv=None):
    """Command line:  python ivr_pipeline.py path/to/data.npz [--out FOLDER] [--no-plots]"""
    import argparse
    ap = argparse.ArgumentParser(description="Decode keypad presses from IVR call recordings and answer the business questions.")
    ap.add_argument("path", nargs="?", help="path to the .npz data file")
    ap.add_argument("--out", default="ivr_outputs", help="folder for the two chart PNGs (default: ivr_outputs)")
    ap.add_argument("--no-plots", action="store_true", help="do not draw or save charts")
    ap.add_argument("-f", dest="_kernel", help=argparse.SUPPRESS)      # swallows Jupyter/Colab's own '-f kernel.json'
    args, _ = ap.parse_known_args(argv)
    if not args.path:
        ap.print_usage()
        return 0
    try:
        run_pipeline(args.path, out_dir=args.out, plot=not args.no_plots)
    except (FileNotFoundError, ValueError) as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    if code and not _in_notebook():
        sys.exit(code)
