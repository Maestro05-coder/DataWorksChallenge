"""
EDA starter: ML-assisted history matching (oil & gas hackathon)


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

Needs: pandas, numpy, matplotlib, openpyxl
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


def read_sheet(file_stem, sheet_name, header=None):
    """Read one sheet (matched ignoring upper/lower case) from one named file."""
    path = find_file(file_stem)
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

        # Cumulative production can never go down; if it does, the data is odd.
        if key.endswith("_cum") and cc:
            n_drops = int((df[cc].diff() < -1e-6).sum().sum())
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
        for c in case_cols(df):
            if use_color and c in params.index and pd.notna(params.loc[c, COLOR_BY]):
                color = cmap(norm(params.loc[c, COLOR_BY]))
            else:
                color = "lightgrey"
            ax.plot(df.index, df[c], color=color, lw=0.8, alpha=0.7)
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


def plot_parameter_coverage(params):
    """Plot 2: scatter matrix of the 4 settings.

    Shows how the 100 cases were sampled. Big empty regions mean the model
    will have nothing to learn from there.
    """
    pd.plotting.scatter_matrix(params, figsize=(9, 9), diagonal="hist", alpha=0.7)
    fig = plt.gcf()
    fig.suptitle("Coverage of the 4 uncertain settings")
    save(fig, "02_parameter_coverage.png")


def build_summary(data, params):
    """One row per case: its 4 settings + its FINAL value for each output."""
    finals = pd.DataFrame({
        f"final_{key}": df[case_cols(df)].apply(last_valid)
        for key, df in data.items()
    })
    summary = params.join(finals, how="inner")
    summary.to_csv(OUTPUT_DIR / "case_summary.csv")
    return summary


def plot_param_vs_outcome(data, labels, summary, param_cols):
    """Plot 3: each setting (columns) vs each final result (rows).

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
    save(fig, "03_param_vs_outcome.png")


def plot_correlation(summary, param_cols):
    """Plot 4: rank correlation between settings and final results.

    +1 = bigger setting -> bigger result, -1 = the opposite, 0 = no link.
    (Ranks are used so curved but steadily increasing relationships still count.)
    """
    out_cols = [c for c in summary.columns if c.startswith("final_")]
    corr = summary.rank().corr().loc[param_cols, out_cols]
    corr.to_csv(OUTPUT_DIR / "param_outcome_correlation.csv")

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(out_cols)))
    ax.set_xticklabels([c.replace("final_", "") for c in out_cols], rotation=30, ha="right")
    ax.set_yticks(range(len(param_cols)))
    ax.set_yticklabels(param_cols)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="rank correlation")
    ax.set_title("Which settings drive which results?")
    save(fig, "04_param_outcome_correlation.png")
    print("\nRank correlation (settings x final results):")
    print(corr.round(2).to_string())


def rank_cases_by_misfit(data):
    """Score every case by how far it is from the real field (lower = better).

    For oil, gas and water rate separately: RMSE between the case and the
    history over the dates where history exists, divided by the largest history
    value of that stream (so oil, gas and water count equally despite having
    different units/sizes). The three scores are added up.
    """
    total = None
    for key in RATE_KEYS:
        df = data[key]
        if HISTORY_COL not in df or df[HISTORY_COL].notna().sum() == 0:
            raise ValueError(f"No usable '{HISTORY_COL}' in the {key} sheet, "
                             "cannot score cases against history.")
        hist = df[HISTORY_COL]
        mask = hist.notna()
        cases = df[case_cols(df)]
        rmse = np.sqrt(((cases[mask].sub(hist[mask], axis=0)) ** 2).mean())
        score = rmse / max(hist[mask].abs().max(), 1e-9)
        # Plain '+' (not fill_value): a case that is empty in any stream gets
        # NaN overall and is ranked last instead of looking artificially good.
        total = score if total is None else total + score
    return total.dropna().sort_values()


def plot_best_matches(data, labels, scores, params):
    """Plot 5: the TOP_N cases closest to history, drawn over all the others."""
    best = list(scores.index[:TOP_N])
    colors = plt.get_cmap("tab10")(np.arange(len(best)))
    fig, axes = plt.subplots(1, len(RATE_KEYS), figsize=(18, 5), constrained_layout=True)

    for ax, key in zip(np.atleast_1d(axes), RATE_KEYS):
        df = data[key]
        for c in case_cols(df):
            ax.plot(df.index, df[c], color="lightgrey", lw=0.6)
        for c, col in zip(best, colors):
            ax.plot(df.index, df[c], color=col, lw=1.8, label=c)
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
    plot_param_vs_outcome(data, labels, summary, param_cols)
    plot_correlation(summary, param_cols)

    scores = rank_cases_by_misfit(data)
    plot_best_matches(data, labels, scores, params)
    plot_field_history()

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()