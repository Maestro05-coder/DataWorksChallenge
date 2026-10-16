"""
EDA starter: ML-assisted history matching (oil & gas hackathon)
================================================================

WHAT THIS SCRIPT ANSWERS
    "How do the 4 uncertain reservoir settings change the production curves,
     and which simulated cases look most like the real field?"

THE 8 EXCEL FILES AND WHAT THIS SCRIPT DOES WITH EACH

    01 Introduction      USED      sheet 'Uncertainty Values' = the 4 settings
                                   per case (the model INPUTS).
                                   ('Nomenclature' sheet is not needed.)
    02 Field Production  USED      the 6 production sheets (gas/oil/water x
                                   rate/cumulative) with Case_1..Case_N plus
                                   RAMP_History (the real field), and the
                                   'Field Production' sheet (incl. injection).
    03-06 Train Cases 1-4   only their SHEET NAMES are listed (an inventory:
    07 Validation Cases     how many case sheets per file). The heavy grid
    08 Test Cases           data inside is NOT read.

    Only 01 and 02 are required. Missing files 03-08 are simply skipped.

HOW TO RUN
    1. Put the Excel files in one folder (names like '01 Introduction.xlsx').
    2. python history_match_eda.py path/to/that/folder
       (or edit DATA_DIR below and run without arguments)
    3. Plots and CSVs are written to OUTPUT_DIR.

PLOTS (saved to OUTPUT_DIR) - each plot states its own verdict
    01  every simulated case vs the real field
    02  did the runs cover the 4 settings evenly?   GREEN / YELLOW / RED per pair
    03  for each result, the setting that controls it most (and the weakest links)
    03b every setting vs every result (full detail grid)
    04  who correlates with whom (settings, results, and each other) + printed
        list of all STRONG links in plain English
    05  the cases closest to the real history
    06  real field production and water injection
    07  WHEN each setting matters (correlation over time)
    08  funnel + what the history says about each setting's range

SINGLE WORKBOOK: if the 8 separate files are not found, the script falls back
to any .xlsx in DATA_DIR that contains the needed sheet names (e.g. one big
trainingData.xlsx).

Needs: pandas, numpy, matplotlib (3.4+), openpyxl
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# =============================================================================
# CONFIG - the only part you should need to edit
# =============================================================================
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUTPUT_DIR = Path("eda_output")

# File names WITHOUT the .xlsx extension. Matching ignores upper/lower case, and
# also accepts a name that merely STARTS with this text (e.g. "02 Field
# Production (1)"), in case a download added a suffix.
FILES = {
    "intro":      "01 Introduction",        # Nomenclature + Uncertainty Values
    "field":      "02 Field Production",    # production sheets + Field Production
    "train":      ["03 Train Cases 1", "04 Train Cases 2",
                   "05 Train Cases 3", "06 Train Cases 4"],
    "validation": "07 Validation Cases",
    "test":       "08 Test Cases",
}

# Colour the curves by one of the 4 settings so its effect is visible.
# Set to None for plain grey curves. Must match a column in Uncertainty Values.
COLOR_BY = "Aquifer Pore Volume"

TOP_N = 5            # how many best-matching cases to highlight
SHOW_PLOTS = True    # False = only save PNGs, do not open windows

# Sheet names (matching is case-insensitive, so 'Oil production cumulative'
# and 'Oil Production Cumulative' are treated as the same sheet).
PARAM_SHEET = "Uncertainty Values"
FIELD_SHEET = "Field Production"
PRODUCTION_SHEETS = {                      # short key -> sheet name
    "gas_cum":    "Gas Production Cumulative",
    "gas_rate":   "Gas Production Rate",
    "oil_cum":    "Oil production cumulative",
    "oil_rate":   "Oil Production Rate",
    "water_cum":  "Water Production Cumulative",
    "water_rate": "Water Production Rate",
}
RATE_KEYS = ["oil_rate", "gas_rate", "water_rate"]  # used to score history match

HISTORY_COL = "RAMP_History"   # column holding the real observed field
CASE_PREFIX = "Case_"          # simulated-run columns are Case_1, Case_2, ...

# --- thresholds that turn numbers into plain-English verdicts ----------------
STRONG_RHO = 0.6     # |rank correlation| at/above this  = "STRONG link"
MODERATE_RHO = 0.3   # at/above this = "moderate"; below = "weak / no link"
MATCH_TOL = 0.05     # a case "hits" the real value if within +-5% of it
MATCH_FRACTION = 0.3 # history-match funnel: best 30% of cases count as "good"
RANGE_TOP_N = 10     # best N cases used to show what history says per setting

NICE = {"gas_cum": "Gas cumulative", "gas_rate": "Gas rate",
        "oil_cum": "Oil cumulative", "oil_rate": "Oil rate",
        "water_cum": "Water cumulative", "water_rate": "Water rate"}


# =============================================================================
# STEP 1 - LOCATE THE EXCEL FILES AND READ SHEETS FROM THEM
# =============================================================================
class SheetNotFound(Exception):
    pass


def find_file(stem):
    """Return the .xlsx in DATA_DIR whose name matches `stem`, else None."""
    files = [p for p in sorted(DATA_DIR.glob("*.xlsx")) if not p.name.startswith("~$")]
    target = stem.strip().lower()
    for p in files:                                   # exact name first
        if p.stem.strip().lower() == target:
            return p
    for p in files:                                   # then 'starts with'
        if p.stem.strip().lower().startswith(target):
            return p
    return None


_NAMES_CACHE = {}


def sheet_names(path):
    """List the sheet names of a workbook WITHOUT loading its cells.

    openpyxl only reads the workbook's table of contents here, so this is fast
    even for the big train/validation/test grid files.
    """
    key = str(path)
    if key not in _NAMES_CACHE:
        with pd.ExcelFile(path) as xl:
            _NAMES_CACHE[key] = xl.sheet_names
    return _NAMES_CACHE[key]



def find_workbook_with_sheet(sheet_name):
    """Fallback for ONE combined workbook (e.g. trainingData.xlsx): return the
    first .xlsx in DATA_DIR that contains a sheet called `sheet_name`."""
    target = sheet_name.strip().lower()
    for p in sorted(DATA_DIR.glob("*.xlsx")):
        if p.name.startswith("~$"):
            continue
        if target in [n.strip().lower() for n in sheet_names(p)]:
            return p
    return None


def read_sheet(file_stem, sheet_name, header=None):
    """Read one sheet (matched ignoring upper/lower case) from one named file."""
    path = find_file(file_stem)
    if path is None:                      # maybe everything is in ONE workbook
        path = find_workbook_with_sheet(sheet_name)
    if path is None:
        have = [p.name for p in sorted(DATA_DIR.glob("*.xlsx"))]
        raise FileNotFoundError(
            f"Could not find a file named like '{file_stem}.xlsx' in "
            f"{DATA_DIR.resolve()}.\nExcel files I can see there: {have}"
        )
    names = sheet_names(path)
    lookup = {n.strip().lower(): n for n in names}
    real = lookup.get(sheet_name.strip().lower())
    if real is None:
        raise SheetNotFound(
            f"Sheet '{sheet_name}' not found in '{path.name}'.\n"
            f"Sheets in that file: {names}"
        )
    return pd.read_excel(path, sheet_name=real, header=header)


def inventory_case_files(n_case_columns):
    """List how many sheets (= cases) each train/validation/test file holds.

    Only sheet NAMES are read, never the grids, so this is cheap. It helps pin
    down how the case sheets relate to the Case_1..Case_N columns.
    """
    groups = {"train": FILES["train"],
              "validation": [FILES["validation"]],
              "test": [FILES["test"]]}
    totals = {}
    print("\n=== CASE-FILE INVENTORY (sheet names only, grids not read) ===")
    for group, stems in groups.items():
        count = 0
        for stem in stems:
            path = find_file(stem)
            if path is None:
                print(f"  {stem:22s} not found in {DATA_DIR} -> skipped")
                continue
            names = sheet_names(path)
            count += len(names)
            print(f"  {stem:22s} {len(names):3d} sheets  e.g. {names[:3]}")
        totals[group] = count
    total = sum(totals.values())
    print(f"  case sheets found: {totals}  (total {total})")
    print(f"  Case_ columns in the production sheets: {n_case_columns}")
    if total and total != n_case_columns:
        print("  NOTE: these two numbers differ. That can be fine, but confirm how a "
              "case sheet maps to a Case_N column / a row in Uncertainty Values.")
    print("==============================================================\n")


# =============================================================================
# STEP 2 - LOAD + CLEAN THE DATA
# =============================================================================
def load_parameters():
    """Uncertainty Values -> table with one row per case, one column per dial.

    Index is renamed 'Case 1' -> 'Case_1' so it matches the production sheets.
    """
    df = read_sheet(FILES["intro"], PARAM_SHEET, header=0)
    df = df.rename(columns={df.columns[0]: "Item"})
    df.columns = [str(c).strip() for c in df.columns]
    df["Item"] = df["Item"].astype(str).str.strip().str.replace(" ", "_", regex=False)
    df = df.set_index("Item").apply(pd.to_numeric, errors="coerce")
    return df.dropna(how="all")


def load_production(sheet_name):
    """Load one production sheet into a clean (dates x cases) table.

    The sheets have slightly different header layouts, e.g.
        row 0: Date | Case_1 | Case_2 ... | RAMP_History
        row 1: (blank) | units text ...
        row 2+: data
    but Water Production Rate has an EXTRA header row on top (X, Y1, Y2 ...).
    So instead of hard-coding row numbers we locate the row whose first cell
    says 'Date' and read relative to it.

    Returns (table, unit_label) e.g. (..., 'Oil production rate [STB/d]').
    """
    raw = read_sheet(FILES["field"], sheet_name, header=None)

    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")][0]
    unit_label = str(raw.iloc[header_row + 1, 1]).strip()       # the units row

    body = raw.iloc[header_row + 2:, :].copy()                   # real data rows
    body.columns = ["Date"] + [str(c).strip() for c in raw.iloc[header_row, 1:]]
    body["Date"] = pd.to_datetime(body["Date"], errors="coerce")
    body = body.dropna(subset=["Date"]).set_index("Date").sort_index()
    body = body.apply(pd.to_numeric, errors="coerce")            # text -> numbers
    # NOTE: we deliberately keep columns that are entirely empty. A Case_N
    # column with no data may be a failed simulation, and the sanity checks
    # below report it instead of hiding it.
    body = body.loc[:, ~body.columns.isin(["nan", ""])]          # junk header cells
    return body, unit_label


def case_cols(df):
    """Names of the simulated-run columns (Case_1 ... Case_N)."""
    return [c for c in df.columns if str(c).startswith(CASE_PREFIX)]


def last_valid(series):
    """Last non-empty value of a curve (its value at the final date)."""
    s = series.dropna()
    return s.iloc[-1] if len(s) else np.nan


def interpolate_to(series, dates):
    """Evaluate one case's curve at the requested dates.

    Each case may report only on ITS OWN dates (the rest of its column is
    empty), so cases cannot be compared row by row. We keep the case's valid
    points and linearly interpolate between them. Dates outside the range the
    case covers become NaN (no extrapolation).
    """
    s = series.dropna()
    if len(s) < 2:
        return pd.Series(np.nan, index=dates)
    x = s.index.values.astype("datetime64[ns]").astype("int64")
    xi = pd.DatetimeIndex(dates).values.astype("datetime64[ns]").astype("int64")
    return pd.Series(np.interp(xi, x, s.values, left=np.nan, right=np.nan), index=dates)


# =============================================================================
# STEP 3 - SANITY CHECKS (catch data problems BEFORE modelling)
# =============================================================================
def run_sanity_checks(data, params):
    print("\n=== SANITY CHECKS ===")
    reference_cases = None
    for key, df in data.items():
        cc = case_cols(df)
        n_hist = df[HISTORY_COL].notna().sum() if HISTORY_COL in df else "MISSING"
        span = f"{df.index.min().date()} -> {df.index.max().date()}" if len(df) else "no dates"
        nan_pct = df[cc].isna().mean().mean() * 100 if cc else float("nan")
        print(f"{key:11s} dates={len(df):4d} ({span}) cases={len(cc):3d} "
              f"history_points={n_hist} empty_cells={nan_pct:.1f}%")

        # Cases with NO data at all (e.g. a failed simulation run).
        empty = [c for c in cc if df[c].isna().all()]
        if empty:
            print(f"   WARNING: case columns with no data at all: {empty}")
        if HISTORY_COL in df and df[HISTORY_COL].isna().all():
            print("   WARNING: RAMP_History is completely empty in this sheet")

        # How many dates does each case actually report on?
        if cc:
            pts = df[cc].notna().sum()
            print(f"   points per case: min={pts.min()} median={int(pts.median())} "
                  f"max={pts.max()} (out of {len(df)} dates)")
            if pts.median() < 0.5 * len(df):
                print("   NOTE: cases are SPARSE (each reports on its own dates). "
                      "Plots draw lines through each case's own points, and "
                      "scoring interpolates onto the history dates.")

        # Cumulative production can never go down; if it does, the data is odd.
        # (checked on each case's own valid points, ignoring the gaps)
        if key.endswith("_cum") and cc:
            n_drops = int(sum((df[c].dropna().diff() < -1e-6).sum() for c in cc))
            if n_drops:
                print(f"   WARNING: {n_drops} places where cumulative DECREASES")

        # Every sheet should contain the same set of cases.
        if reference_cases is None:
            reference_cases = set(cc)
        elif set(cc) != reference_cases:
            print(f"   WARNING: case columns differ from the first sheet "
                  f"(difference: {len(set(cc) ^ reference_cases)} columns)")

    # Every case should have its 4 settings in Uncertainty Values.
    missing = sorted(reference_cases - set(params.index)) if reference_cases else []
    print(f"parameters: {params.shape[0]} cases x {params.shape[1]} settings "
          f"-> {list(params.columns)}")
    if missing:
        print(f"   WARNING: {len(missing)} cases have no parameter row "
              f"(e.g. {missing[:3]}). Is the Uncertainty Values sheet complete?")
    print("=====================\n")


# =============================================================================
# STEP 4 - PLOTS
# =============================================================================
def save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=130, bbox_inches="tight")
    print(f"saved {OUTPUT_DIR / name}")


def plot_ensemble(data, labels, params):
    """Plot 1: reproduce the competition figure (6 panels).

    Every thin line is one simulated case. Black dots are the real field.
    If COLOR_BY is set, lines are coloured by that setting's value, so you can
    SEE which curves it pushes up or down.
    """
    order = ["gas_cum", "gas_rate", "oil_cum", "oil_rate", "water_cum", "water_rate"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    use_color = COLOR_BY is not None and COLOR_BY in params.columns
    if use_color:
        vals = params[COLOR_BY]
        norm = Normalize(vals.min(), vals.max())
        cmap = plt.get_cmap("viridis")

    for ax, key in zip(axes.ravel(), order):
        df = data[key]
        # rates are step-like in the simulator, cumulatives are smooth
        style = "steps-post" if key.endswith("_rate") else "default"
        for c in case_cols(df):
            if use_color and c in params.index and pd.notna(params.loc[c, COLOR_BY]):
                color = cmap(norm(params.loc[c, COLOR_BY]))
            else:
                color = "lightgrey"
            s_c = df[c].dropna()        # drop gaps so the line connects the points
            ax.plot(s_c.index, s_c.values, color=color, lw=0.8, alpha=0.7,
                    drawstyle=style)
        if HISTORY_COL in df:
            ax.plot(df.index, df[HISTORY_COL], "ko", ms=3, label="History")
        ax.set_title(labels[key], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)

    axes[0, 0].legend(loc="upper left")
    if use_color:
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                     ax=axes.ravel().tolist(), shrink=0.6, label=COLOR_BY)
    fig.suptitle("Simulated cases vs real field history"
                 + (f"  (colour = {COLOR_BY})" if use_color else ""))
    save(fig, "01_ensemble_curves.png")


def pair_design_stats(x, y, bins):
    """Quality of the sampling of TWO settings: do they move together, and are
    there empty regions on the map of (x, y)?"""
    rho = x.rank().corr(y.rank())
    h, _, _ = np.histogram2d(x, y, bins=bins)
    empty = float((h == 0).mean())
    reasons = []
    if abs(rho) >= 0.3:
        verdict = "PROBLEM"
        reasons.append("the two settings move together")
    elif abs(rho) >= 0.15:
        verdict = "CHECK"
        reasons.append("slight tendency to move together")
    else:
        verdict = "OK"
    if empty > 0.25:
        verdict = "PROBLEM"
        reasons.append("large empty regions")
    elif empty > 0.12 and verdict == "OK":
        verdict = "CHECK"
        reasons.append("a few empty regions")
    return {"rho": rho, "empty": empty, "verdict": verdict,
            "why": "; ".join(reasons) if reasons else "evenly spread, independent"}


def plot_parameter_coverage(params):
    """Plot 2: did the runs cover the settings evenly?

    Lower-left : the real map of where the simulator was run (one dot = one
                 case), background colour = verdict for that pair.
    Upper-right: the same pair in numbers + a plain-English verdict.
    Diagonal   : how the values of ONE setting are spread (should be flat).
    GREEN = fine, YELLOW = look closer, RED = the ML model will struggle there.
    """
    p = params.dropna()
    cols = list(p.columns)
    n = len(cols)
    bins = 5 if len(p) >= 50 else 3
    tint = {"OK": "#d4edda", "CHECK": "#fff3cd", "PROBLEM": "#f8d7da"}
    edge = {"OK": "#1a7f37", "CHECK": "#b8860b", "PROBLEM": "#b22222"}
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.0 * n), constrained_layout=True)
    rows = []
    for i, yc in enumerate(cols):
        for j, xc in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(p[xc], bins=10, color="#9ecae1", edgecolor="k", lw=0.4)
                ax.set_title(xc, fontsize=10, weight="bold")
                ax.set_yticks([])
            else:
                st = pair_design_stats(p[xc], p[yc], bins)
                if i > j:                                   # lower: the map
                    ax.set_facecolor(tint[st["verdict"]])
                    ax.scatter(p[xc], p[yc], s=10, alpha=0.8, color="#1f77b4")
                    for sp in ax.spines.values():
                        sp.set_edgecolor(edge[st["verdict"]]); sp.set_linewidth(2)
                else:                                       # upper: the verdict
                    rows.append({"setting_x": xc, "setting_y": yc, "corr": st["rho"],
                                 "empty_share": st["empty"], "verdict": st["verdict"],
                                 "why": st["why"]})
                    ax.set_facecolor(tint[st["verdict"]])
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.5, 0.68, st["verdict"], ha="center", va="center",
                            fontsize=20, weight="bold", color=edge[st["verdict"]],
                            transform=ax.transAxes)
                    ax.text(0.5, 0.33,
                            f"{xc}\nvs {yc}\ncorrelation {st['rho']:+.2f}\n"
                            f"empty map area {st['empty']:.0%}\n{st['why']}",
                            ha="center", va="center", fontsize=8, transform=ax.transAxes)
            if i == n - 1 and i != j:
                ax.set_xlabel(xc)
            if j == 0 and i != j:
                ax.set_ylabel(yc)
            ax.tick_params(labelsize=7)
    fig.suptitle("Plot 2 - Did the runs cover the settings evenly?\n"
                 "GREEN = good  |  YELLOW = look closer  |  RED = ML model will struggle "
                 "(gaps = it must guess, correlated = cannot separate the two effects)",
                 fontsize=12)
    save(fig, "02_parameter_coverage.png")
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "parameter_design_check.csv", index=False)
    print("\nParameter design check (are the runs well spread?):")
    for r in rows:
        print(f"  {r['verdict']:8s} {r['setting_x']} vs {r['setting_y']}: "
              f"corr {r['corr']:+.2f}, empty map {r['empty_share']:.0%} -> {r['why']}")


def build_summary(data, params):
    """One row per case: its 4 settings + its FINAL value for each output."""
    # "Final" = value at the last date the real field has data, so every case
    # is compared at the same moment (cases that stop earlier become NaN).
    finals = {}
    for key, df in data.items():
        ref_date = df[HISTORY_COL].dropna().index.max()
        finals[f"final_{key}"] = pd.Series({
            c: interpolate_to(df[c], [ref_date]).iloc[0] for c in case_cols(df)
        })
    finals = pd.DataFrame(finals)
    summary = params.join(finals, how="inner")
    summary.to_csv(OUTPUT_DIR / "case_summary.csv")
    return summary


def plot_param_vs_outcome(data, labels, summary, param_cols):
    """Plot 3b (detail view): EVERY setting vs EVERY final result.

    A clear trend means that setting controls that result. A shapeless cloud
    means it barely matters. The dashed line is the REAL field's final value,
    so you can see which settings land near it.
    """
    keys = list(data)
    fig, axes = plt.subplots(len(keys), len(param_cols),
                             figsize=(16, 2.6 * len(keys)), constrained_layout=True,
                             squeeze=False)
    for i, key in enumerate(keys):
        hist_final = last_valid(data[key][HISTORY_COL]) if HISTORY_COL in data[key] else np.nan
        for j, p in enumerate(param_cols):
            ax = axes[i, j]
            ax.scatter(summary[p], summary[f"final_{key}"], s=14, alpha=0.7)
            ax.axhline(hist_final, color="k", ls="--", lw=1)
            if i == len(keys) - 1:
                ax.set_xlabel(p)
            if j == 0:
                ax.set_ylabel(labels[key], fontsize=7)
    fig.suptitle("Each setting vs each final result (dashed = real field)")
    save(fig, "03b_all_pairs_grid.png")


def strength_word(r):
    a = abs(r)
    return "STRONG" if a >= STRONG_RHO else ("moderate" if a >= MODERATE_RHO else "weak")


def nice_out(col):
    """'final_water_cum' -> 'Water cumulative'."""
    return NICE.get(col.replace("final_", ""), col)


STRENGTH_COLOR = {"STRONG": "#1a7f37", "moderate": "#b8860b", "weak": "#b22222"}


def _link_panel(ax, data, summary, p, o, r, show_legend=False):
    """One big setting-vs-result panel with a plain-English title."""
    key = o.replace("final_", "")
    x, y = summary[p], summary[o]
    m = x.notna() & y.notna()
    ax.scatter(x[m], y[m], s=18, alpha=0.7, color="#1f77b4", label="one simulated case")
    # typical trend: median of y inside 6 slices of x
    edges = np.quantile(x[m], np.linspace(0, 1, 7))
    cx, cy = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = m & (x >= a) & (x <= b)
        if sel.sum() >= 2:
            cx.append(x[sel].median()); cy.append(y[sel].median())
    ax.plot(cx, cy, color="#ff7f0e", lw=2.8, label="typical trend (median)")
    H = last_valid(data[key][HISTORY_COL]) if HISTORY_COL in data[key] else np.nan
    if np.isfinite(H):
        ax.axhline(H, color="k", ls="--", lw=1.3, label="real field")
        tol = MATCH_TOL * max(abs(H), 0.1 * (y[m].max() - y[m].min()))
        near = m & ((y - H).abs() <= tol)
        if near.any():
            ax.scatter(x[near], y[near], s=50, color="#d62728", edgecolor="k",
                       zorder=5, label=f"within {MATCH_TOL:.0%} of real")
            ax.axvspan(x[near].min(), x[near].max(), color="#d62728", alpha=0.08)
    w = strength_word(r)
    effect = ("no clear effect" if w == "weak" else
              f"higher setting gives {'HIGHER' if r > 0 else 'LOWER'} result")
    ax.set_title(f"{p} \u2192 {nice_out(o)}\n{w} link ({effect}, corr {r:+.2f})",
                 fontsize=10, weight="bold", color=STRENGTH_COLOR[w])
    ax.set_xlabel(f"{p}  (setting)")
    ax.set_ylabel(f"{nice_out(o)} at end of history")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7.5, loc="best")


def plot_top_links(data, labels, summary, param_cols):
    """Plot 3: for EVERY result, the setting that controls it most, drawn big.

    Top block   : strongest setting for each result (sorted strongest first).
    Bottom row  : the 3 weakest setting/result pairs, for contrast.
    Orange line = typical trend. Dashed = real field. Red = cases that land on
    the real value, and the pink band = the setting range those cases occupy.
    """
    out_cols = [c for c in summary.columns if c.startswith("final_")]
    corr = summary[param_cols + out_cols].rank().corr()
    best, allp = [], []
    for o in out_cols:
        r = corr.loc[param_cols, o].dropna()
        if len(r):
            best.append((r.abs().idxmax(), o, r[r.abs().idxmax()]))
        allp += [(p, o, corr.loc[p, o]) for p in param_cols if pd.notna(corr.loc[p, o])]
    best.sort(key=lambda t: -abs(t[2]))
    weak = sorted(allp, key=lambda t: abs(t[2]))[:3]

    n_top_rows = int(np.ceil(len(best) / 3))
    fig = plt.figure(figsize=(17, 4.6 * (n_top_rows + 1)), constrained_layout=True)
    top, bot = fig.subfigures(2, 1, height_ratios=[n_top_rows, 1])
    top.suptitle("STRONGEST LINKS - for each result, the setting that controls it most "
                 "(these are what the history can pin down)", fontsize=12, weight="bold")
    axes = top.subplots(n_top_rows, 3, squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k < len(best):
            _link_panel(ax, data, summary, *best[k], show_legend=(k == 0))
        else:
            ax.axis("off")
    bot.suptitle("WEAKEST LINKS - changing this setting barely changes this result "
                 "(history cannot tell you its value)", fontsize=12, weight="bold")
    for ax, t in zip(bot.subplots(1, 3, squeeze=False).ravel(), weak):
        _link_panel(ax, data, summary, *t)
    fig.suptitle("Plot 3 - Which settings control which results?", fontsize=14, weight="bold")
    save(fig, "03_top_links.png")


def plot_correlation(summary, param_cols):
    """Plot 4: who correlates with whom - EVERYTHING against EVERYTHING.

    Rank correlation: +1 = both rise together, -1 = one rises as the other
    falls, 0 = unrelated. Ranks are used so curved but steadily rising
    relationships still count. Bold boxed numbers are STRONG links.
    """
    out_cols = [c for c in summary.columns if c.startswith("final_")]
    cols = param_cols + out_cols
    corr = summary[cols].rank().corr()
    corr.to_csv(OUTPUT_DIR / "correlation_all.csv")
    corr.loc[param_cols, out_cols].to_csv(OUTPUT_DIR / "param_outcome_correlation.csv")
    n, k = len(cols), len(param_cols)
    names = param_cols + [nice_out(c) for c in out_cols]

    fig, ax = plt.subplots(figsize=(1.0 * n + 4, 0.9 * n + 1.5), constrained_layout=True)
    masked = np.ma.array(corr.values, mask=np.triu(np.ones((n, n), bool)))
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for (i, j), v in np.ndenumerate(corr.values):
        if i <= j or np.isnan(v):
            continue
        strong = abs(v) >= STRONG_RHO
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                weight="bold" if strong else "normal",
                color="white" if abs(v) > 0.75 else ("0.6" if abs(v) < MODERATE_RHO else "black"))
        if strong:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=2))
    # coloured outlines around the three blocks
    blocks = [((-.5, -.5), k, k, "#e67e22"),            # settings vs settings
              ((-.5, k - .5), k, n - k, "#2c3e50"),     # settings vs results
              ((k - .5, k - .5), n - k, n - k, "#27ae60")]  # results vs results
    for (x0, y0), w, h, c in blocks:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=c, lw=3.5, ls="--"))
    guide = ("HOW TO READ\n"
             "Each number = how two columns move together.\n"
             "+1 (red): both go up together.  -1 (blue): one up, other down.\n"
             "0 (white / grey text): unrelated.   Boxed bold = STRONG (|corr| >= "
             f"{STRONG_RHO}).\n\n"
             "ORANGE block: setting vs setting  -> should be ~0\n"
             "   (runs were sampled independently; big = bad design)\n"
             "DARK block: setting vs result  -> what you can tune and\n"
             "   what it controls. This is the block that matters.\n"
             "GREEN block: result vs result  -> big numbers = two results\n"
             "   carry the same information (redundant).")
    ax.text(0.98, 0.98, guide, transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(fc="#f8f9fa", ec="k"))
    fig.colorbar(im, ax=ax, label="rank correlation", shrink=0.7)
    ax.set_title("Plot 4 - Who correlates with whom?", fontsize=13, weight="bold")
    save(fig, "04_param_outcome_correlation.png")
    report_links(corr, param_cols, out_cols)


def report_links(corr, param_cols, out_cols):
    """Print (and save) every link in plain English, strongest first."""
    cols = param_cols + out_cols
    rows = []
    for a_i, a in enumerate(cols):
        for b in cols[a_i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if a in param_cols and b in param_cols:
                kind = "setting-setting"
            elif a in param_cols:
                kind = "setting->result"
            else:
                kind = "result-result"
            rows.append({"kind": kind, "a": a, "b": b, "corr": r,
                         "strength": strength_word(r)})
    links = pd.DataFrame(rows)
    links["abs"] = links["corr"].abs()
    links = links.sort_values("abs", ascending=False).drop(columns="abs")
    links.to_csv(OUTPUT_DIR / "links_ranked.csv", index=False)

    def nm(c):
        return nice_out(c) if c.startswith("final_") else c

    def sentence(r):
        if r["kind"] == "setting->result":
            d = "UP" if r["corr"] > 0 else "DOWN"
            return (f"  {r['strength']:8s} {nm(r['a'])} UP  ->  {nm(r['b'])} {d}"
                    f"   (corr {r['corr']:+.2f})")
        how = "rise and fall together" if r["corr"] > 0 else "move in OPPOSITE directions"
        return (f"  {r['strength']:8s} {nm(r['a'])} and {nm(r['b'])} {how}"
                f"   (corr {r['corr']:+.2f})")

    print("\n=== STRONG LINKS (|corr| >= %.1f) ===" % STRONG_RHO)
    for kind, title in [("setting->result", "Settings that drive results"),
                        ("result-result", "Results that carry the same information"),
                        ("setting-setting", "Settings that move together (design problem if any)")]:
        sub = links[(links["kind"] == kind) & (links["strength"] == "STRONG")]
        print(f"\n{title}:" + ("  none" if sub.empty else ""))
        for _, r in sub.iterrows():
            print(sentence(r))
    print("\nSettings with no strong effect on ANY result "
          "(history is unlikely to pin these down):")
    found = False
    for p in param_cols:
        mx = links[(links["kind"] == "setting->result") &
                   ((links["a"] == p) | (links["b"] == p))]["corr"].abs().max()
        if pd.notna(mx) and mx < STRONG_RHO:
            found = True
            print(f"  {p}  (strongest effect on any result: corr {mx:.2f})")
    if not found:
        print("  none - every setting strongly drives at least one result")


def misfit_by_stream(data):
    """Misfit of every case against history, SEPARATELY for oil, gas, water rate.

    For each stream: RMSE between the case and the history over the dates where
    history exists, divided by the largest history value of that stream (so
    oil, gas and water count equally despite having different units/sizes).
    Returns a table: one row per case, one column per stream (lower = better).
    """
    scores = {}
    for key in RATE_KEYS:
        df = data[key]
        if HISTORY_COL not in df or df[HISTORY_COL].notna().sum() == 0:
            raise ValueError(f"No usable '{HISTORY_COL}' in the {key} sheet, "
                             "cannot score cases against history.")
        hist = df[HISTORY_COL]
        h = hist[hist.notna()]
        rmse = pd.Series({
            c: np.sqrt(np.nanmean((interpolate_to(df[c], h.index).values - h.values) ** 2))
            if interpolate_to(df[c], h.index).notna().any() else np.nan
            for c in case_cols(df)
        })
        scores[key] = rmse / max(h.abs().max(), 1e-9)
    return pd.DataFrame(scores)


def rank_cases_by_misfit(data):
    """Total misfit per case = oil + gas + water scores added up (lower = better).

    A case that is empty in any stream gets NaN overall and is dropped instead
    of looking artificially good.
    """
    return misfit_by_stream(data).sum(axis=1, skipna=False).dropna().sort_values()


def sensitivity_over_time(df, params, n_grid=40):
    """Rank correlation between each setting and the result, date by date."""
    cc = [c for c in case_cols(df) if c in params.index]
    hist = df[HISTORY_COL].dropna().index if HISTORY_COL in df else df.index
    grid = pd.date_range(hist.min(), hist.max(), periods=n_grid)
    mat = pd.DataFrame({c: interpolate_to(df[c], grid).values for c in cc}, index=grid)
    out = pd.DataFrame(np.nan, index=grid, columns=params.columns)
    for p in params.columns:
        for d in grid:
            y = mat.loc[d]
            ok = y.notna() & params.loc[cc, p].notna()
            if ok.sum() >= 10 and y[ok].nunique() > 1:
                out.loc[d, p] = params.loc[cc, p][ok].rank().corr(y[ok].rank())
    return out


def plot_sensitivity_over_time(data, params):
    """Plot 7: WHEN does each setting matter?

    One line per setting. Line near 0 = no influence at that date. Line in the
    green band = strong influence (up = bigger setting gives bigger result,
    down = bigger setting gives smaller result). Lines that start at 0 and fan
    out show an effect that builds up over time - important for forecasting.
    """
    order = [k for k in ["oil_rate", "gas_rate", "water_rate",
                         "oil_cum", "gas_cum", "water_cum"] if k in data]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True, sharey=True)
    colors = dict(zip(params.columns, plt.get_cmap("tab10")(np.arange(len(params.columns)))))
    for ax, key in zip(axes.ravel(), order):
        sens = sensitivity_over_time(data[key], params)
        ax.axhspan(STRONG_RHO, 1.05, color="green", alpha=0.10)
        ax.axhspan(-1.05, -STRONG_RHO, color="green", alpha=0.10)
        ax.axhline(0, color="k", lw=0.8)
        for p in sens:
            ax.plot(sens.index, sens[p], lw=2.4, color=colors[p], label=p)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(NICE[key], fontsize=12, weight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
        ax.text(0.01, 0.97, "STRONG: bigger setting -> bigger result", transform=ax.transAxes,
                fontsize=8, va="top", color="green")
        ax.text(0.01, 0.03, "STRONG: bigger setting -> smaller result", transform=ax.transAxes,
                fontsize=8, va="bottom", color="green")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation of setting with result")
    fig.suptitle("Plot 7 - WHEN does each setting matter?  (line far from 0 = that setting "
                 "controls the result at that date)", fontsize=13, weight="bold")
    save(fig, "07_sensitivity_over_time.png")


def plot_history_match_summary(data, params):
    """Plot 8: what does matching the history actually tell us?

    Left : funnel - how many cases stay 'good' as you demand a match on more
           streams (oil, then gas, then water).
    Right: for each setting, the full range tested (grey) vs the range used by
           the best RANGE_TOP_N cases (colour). Narrow colour bar = history pins
           that setting down; full-width bar = history cannot tell.
    """
    ms = misfit_by_stream(data).dropna()
    thresh = ms.quantile(MATCH_FRACTION)
    run = pd.Series(True, index=ms.index)
    labels_f, counts = [], []
    for key in ms.columns:
        run = run & (ms[key] <= thresh[key])
        labels_f.append(("good on " if not labels_f else "... AND ") + NICE[key].lower())
        counts.append(int(run.sum()))
    total = ms.sum(axis=1).sort_values()
    best = total.index[:RANGE_TOP_N]
    pr = params.loc[[c for c in best if c in params.index]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 5.6), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    y = np.arange(len(counts))[::-1]
    a1.barh(y, counts, color="#6baed6", edgecolor="k")
    for yy, c in zip(y, counts):
        a1.text(c + 0.5, yy, f"{c} cases", va="center", fontsize=11, weight="bold")
    a1.set_yticks(y); a1.set_yticklabels(labels_f, fontsize=10)
    a1.set_xlim(0, max(counts) * 1.25)
    a1.set_xlabel(f"number of cases (out of {len(ms)})")
    a1.set_title(f"Funnel: 'good' = best {MATCH_FRACTION:.0%} of cases on that stream.\n"
                 "Every extra stream you must match removes more cases", fontsize=11)
    a1.grid(alpha=0.3, axis="x")

    rows = []
    names = list(params.columns)
    for i, p in enumerate(names[::-1]):
        lo, hi = params[p].min(), params[p].max()
        span = max(hi - lo, 1e-12)
        b0, b1 = pr[p].min(), pr[p].max()
        w = (b1 - b0) / span
        verdict = "PINNED DOWN" if w < 0.5 else ("partly constrained" if w < 0.8 else "NOT constrained")
        col = {"PINNED DOWN": "#1a7f37", "partly constrained": "#b8860b",
               "NOT constrained": "#b22222"}[verdict]
        a2.barh(i, 1, color="#e0e0e0", edgecolor="k")
        a2.barh(i, w, left=(b0 - lo) / span, color=col, edgecolor="k")
        a2.plot((pr[p].iloc[0] - lo) / span, i, "k*", ms=16, zorder=5)
        rows.append({"setting": p, "tested_min": lo, "tested_max": hi,
                     f"best{RANGE_TOP_N}_min": b0, f"best{RANGE_TOP_N}_max": b1,
                     "share_of_range_used": w, "verdict": verdict})
    a2.set_yticks(range(len(names)))
    a2.set_yticklabels([
        f"{r['setting']}  [{r['verdict']}]\nbest {RANGE_TOP_N}: {r[f'best{RANGE_TOP_N}_min']:.3g} to "
        f"{r[f'best{RANGE_TOP_N}_max']:.3g}   (tested {r['tested_min']:.3g} to {r['tested_max']:.3g})"
        for r in rows], fontsize=9)
    a2.set_xlim(0, 1); a2.set_xlabel("position inside the tested range (0 = lowest tested, 1 = highest)")
    a2.set_title(f"What history says about each setting: range used by the best {RANGE_TOP_N} cases\n"
                 "(grey = everything tested, colour = best cases, star = single best case)", fontsize=11)
    fig.suptitle("Plot 8 - What does matching history actually tell us?   "
                 "(rough guide only: just ~100 runs)", fontsize=13, weight="bold")
    save(fig, "08_history_match_summary.png")
    pd.DataFrame(rows[::-1]).to_csv(OUTPUT_DIR / "history_match_ranges.csv", index=False)
    print("\nWhat the history says about each setting (best %d cases):" % RANGE_TOP_N)
    for r in rows[::-1]:
        print(f"  {r['verdict']:18s} {r['setting']}: best cases use {r[f'best{RANGE_TOP_N}_min']:.3g} "
              f"to {r[f'best{RANGE_TOP_N}_max']:.3g} (tested {r['tested_min']:.3g} to {r['tested_max']:.3g})")


def plot_best_matches(data, labels, scores, params):
    """Plot 5: the TOP_N cases closest to history, drawn over all the others."""
    best = list(scores.index[:TOP_N])
    colors = plt.get_cmap("tab10")(np.arange(len(best)))
    fig, axes = plt.subplots(1, len(RATE_KEYS), figsize=(18, 5), constrained_layout=True)

    for ax, key in zip(np.atleast_1d(axes), RATE_KEYS):
        df = data[key]
        for c in case_cols(df):
            s_c = df[c].dropna()
            ax.plot(s_c.index, s_c.values, color="lightgrey", lw=0.6)
        for c, col in zip(best, colors):
            s_c = df[c].dropna()
            ax.plot(s_c.index, s_c.values, color=col, lw=1.8, label=c)
        ax.plot(df.index, df[HISTORY_COL], "ko", ms=3, label="History")
        ax.set_title(labels[key])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Top {TOP_N} cases closest to history")
    save(fig, "05_best_matching_cases.png")

    table = params.reindex(best).assign(misfit_score=scores.loc[best])
    scores.rename("misfit_score").to_csv(OUTPUT_DIR / "misfit_ranking.csv")
    print(f"\nTop {TOP_N} cases by misfit (lower = closer to history):")
    print(table.round(4).to_string())


def plot_field_history():
    """Plot 6 (optional): the real field's rates, incl. WATER INJECTION.

    Injection is a control the operators set, not a simulator result, so it
    matters for the forecast period.
    """
    try:
        df = read_sheet(FILES["field"], FIELD_SHEET, header=0)
    except SheetNotFound:
        print(f"(skipping field history plot: sheet '{FIELD_SHEET}' not found)")
        return
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.drop(columns=["Identifier"], errors="ignore").apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    df.plot(ax=ax, marker="o", ms=3)
    ax.set_title("Real field production and water injection")
    ax.grid(alpha=0.3)
    save(fig, "06_field_history.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) load the 4 settings (01 Introduction) and the 6 production sheets
    #    (02 Field Production)
    print(f"Reading from: {DATA_DIR.resolve()}")
    params = load_parameters()
    data, labels = {}, {}
    for key, sheet in PRODUCTION_SHEETS.items():
        data[key], labels[key] = load_production(sheet)

    # 2) inventory of the heavy case files (names only, nothing heavy is read)
    n_cases = len(case_cols(next(iter(data.values()))))
    inventory_case_files(n_cases)

    # 3) check the data before trusting it
    run_sanity_checks(data, params)

    # 4) plots and tables
    plot_ensemble(data, labels, params)
    plot_parameter_coverage(params)

    summary = build_summary(data, params)
    param_cols = list(params.columns)
    plot_top_links(data, labels, summary, param_cols)
    plot_param_vs_outcome(data, labels, summary, param_cols)
    plot_correlation(summary, param_cols)
    plot_sensitivity_over_time(data, params)
    plot_history_match_summary(data, params)

    scores = rank_cases_by_misfit(data)
    plot_best_matches(data, labels, scores, params)
    plot_field_history()

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()