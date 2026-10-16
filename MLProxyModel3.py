"""
ML-Assisted History Matching and 20-Year Production Forecasting
===============================================================

Executive Summary Pipeline:
    1. Proxy Training: PCA + Gaussian Process Surrogate Model
    2. History Matching: Global Continuous Search via Differential Evolution & Sobol Sampling
    3. Production Forecasting: Arps Decline Curves + WOR Scaling (Validated via Hindcasting)
    4. Submission Verification & Clean Report Generation
"""

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUTPUT_DIR = Path("Judges_OutputV3")
SHOW_PLOTS = True

FILES = {
    "intro": "01 Introduction",
    "field": "02 Field Production",
    "template": "09 Template Deliverable",
}

PARAM_SHEET = "Uncertainty Values"
PRODUCTION_SHEETS = {
    "gas_cum": "Gas Production Cumulative",
    "gas_rate": "Gas Production Rate",
    "oil_cum": "Oil production cumulative",
    "oil_rate": "Oil Production Rate",
    "water_cum": "Water Production Cumulative",
    "water_rate": "Water Production Rate",
}

HISTORY_COL = "RAMP_History"
CASE_PREFIX = "Case_"

TRAIN_IDS = range(1, 71)
VAL_IDS = range(71, 86)
TEST_IDS = range(86, 101)

MATCH_STREAMS = {
    "oil_cum": "linear",
    "water_cum": "linear",
    "water_rate": "log",
}
LOG_PARAMS = ["Permeability Multiplier"]
PCA_VAR, PCA_MAX = 0.999, 6
N_SOBOL_POW2 = 14
ACCEPT_MARGIN_MIN, ACCEPT_MARGIN_REL = 0.01, 0.10
FINAL_REFIT_ON_VAL = True
RANDOM_SEED = 0

FORECAST_START_DEFAULT = "2008-04-01"
FORECAST_END_DEFAULT = "2028-01-01"
FIT_WINDOW_CANDIDATES = [2.0, 3.0, 5.0]
HINDCAST_YEARS = 3.0
WOR_MAX = 50.0
N_SELECT = 10
SELECTION_MODE = "diverse_params"

STREAMS = ["gas", "oil", "water"]
PLAIN = {
    "oil_cum": "Oil Cumulative Production",
    "water_cum": "Water Cumulative Production",
    "water_rate": "Water Production Rate",
}
UNITS = {
    "oil_cum": "STB",
    "water_cum": "STB",
    "water_rate": "STB/day",
    "gas": "MSCF",
    "oil": "STB",
    "water": "STB",
}

GREEN, AMBER, RED, BLUE = "#1a7f37", "#b8860b", "#b22222", "#1f77b4"


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
def list_workbooks():
  return [
      p for p in sorted(DATA_DIR.glob("*.xlsx")) if not p.name.startswith("~$")
  ]


def find_file(stem):
  target = stem.strip().lower()
  for p in list_workbooks():
    if p.stem.strip().lower() == target:
      return p
  for p in list_workbooks():
    if p.stem.strip().lower().startswith(target):
      return p
  return None


def sheet_names(path):
  with pd.ExcelFile(path) as xl:
    return xl.sheet_names


def read_sheet(file_stem, sheet_name, header=None):
  path = find_file(file_stem)
  if path is None:
    for p in list_workbooks():
      if sheet_name.strip().lower() in [n.strip().lower() for n in sheet_names(p)]:
        path = p
        break
  if path is None:
    raise FileNotFoundError(f"Cannot find sheet {sheet_name}")
  lookup = {n.strip().lower(): n for n in sheet_names(path)}
  return pd.read_excel(
      path, sheet_name=lookup[sheet_name.strip().lower()], header=header
  )


def load_parameters():
  df = read_sheet(FILES["intro"], PARAM_SHEET, header=0)
  df = df.rename(columns={df.columns[0]: "Item"})
  df.columns = [str(c).strip() for c in df.columns]
  df["Item"] = (
      df["Item"].astype(str).str.strip().str.replace(" ", "_", regex=False)
  )
  return (
      df.set_index("Item")
      .apply(pd.to_numeric, errors="coerce")
      .dropna(how="all")
  )


def load_production(sheet_name):
  raw = read_sheet(FILES["field"], sheet_name, header=None)
  hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")][0]
  body = raw.iloc[hdr + 2 :, :].copy()
  body.columns = ["Date"] + [str(c).strip() for c in raw.iloc[hdr, 1:]]
  body["Date"] = pd.to_datetime(body["Date"], errors="coerce")
  body = body.dropna(subset=["Date"]).set_index("Date").sort_index()
  body = body.apply(pd.to_numeric, errors="coerce")
  return body.loc[:, ~body.columns.isin(["nan", ""])]


def case_cols(df):
  return [c for c in df.columns if str(c).startswith(CASE_PREFIX)]


def split_of(name):
  try:
    idx = int(str(name).split("_")[-1])
  except ValueError:
    return "unassigned"
  if idx in TRAIN_IDS:
    return "train"
  if idx in VAL_IDS:
    return "validation"
  if idx in TEST_IDS:
    return "test"
  return "unassigned"


def transform_val(x, kind):
  x = np.asarray(x, float)
  return np.log10(np.clip(x, 0, None) + 1.0) if kind == "log" else x


def inverse_transform_val(y, kind):
  y = np.asarray(y, float)
  return 10.0**y - 1.0 if kind == "log" else y


def eval_case(series, dates, is_cum, t0):
  s = series.dropna()
  if len(s) < 2:
    return np.full(len(dates), np.nan)
  ns = (
      lambda idx: pd.DatetimeIndex(idx)
      .values.astype("datetime64[ns]")
      .astype("int64")
      .astype(float)
  )
  x, y, xi = ns(s.index), s.values.astype(float), ns(dates)
  if is_cum:
    if x[0] > t0:
      x, y = np.r_[t0, x], np.r_[0.0, y]
    return np.interp(xi, x, y, left=np.nan, right=np.nan)
  hi = np.clip(np.searchsorted(x, xi), 0, len(x) - 1)
  lo = np.clip(hi - 1, 0, len(x) - 1)
  pick = np.where(np.abs(xi - x[lo]) <= np.abs(x[hi] - xi), lo, hi)
  out = y[pick]
  out[(xi < x[0]) | (xi > x[-1])] = np.nan
  return out


def build_targets(data, cases, t0):
  T = {}
  for key, kind in MATCH_STREAMS.items():
    df = data[key]
    hist = df[HISTORY_COL].dropna()
    hist = hist[~hist.index.duplicated()]
    Y = np.array([
        transform_val(
            eval_case(df[c], hist.index, key.endswith("_cum"), t0), kind
        )
        for c in cases
    ])
    T[key] = dict(kind=kind, dates=hist.index, obs=transform_val(hist.values, kind), Y=Y)
  for t in T.values():
    keep = np.isfinite(t["Y"]).mean(axis=0) >= 0.95
    t["Y"], t["obs"], t["dates"] = (
        t["Y"][:, keep],
        t["obs"][keep],
        t["dates"][keep],
    )
  ok = np.ones(len(cases), bool)
  for t in T.values():
    ok &= np.isfinite(t["Y"]).all(axis=1)
  for t in T.values():
    t["Y"] = t["Y"][ok]
    t["scale"] = max(float(np.abs(t["obs"]).max()), 1e-9)
  return T, [c for c, o in zip(cases, ok) if o]


def stream_rmse(pred, t):
  return np.sqrt(np.mean((pred - t["obs"][None, :]) ** 2, axis=1)) / t["scale"]


def total_misfit(preds, T):
  return sum(stream_rmse(preds[k], T[k]) for k in T)


# =============================================================================
# ML SURROGATE ENGINE (PCA + GAUSSIAN PROCESS REGRESSION)
# =============================================================================
class CurveProxy:

  def __init__(self, param_cols, bounds):
    self.cols, self.bounds = list(param_cols), bounds

  def to_unit(self, P):
    P = np.array(P, float, ndmin=2)
    U = np.empty_like(P)
    for j, c in enumerate(self.cols):
      lo, hi = self.bounds[c]
      if c in LOG_PARAMS:
        U[:, j] = (np.log(P[:, j]) - np.log(lo)) / (np.log(hi) - np.log(lo))
      else:
        U[:, j] = (P[:, j] - lo) / (hi - lo)
    return U

  def from_unit(self, U):
    U = np.array(U, float, ndmin=2)
    P = np.empty_like(U)
    for j, c in enumerate(self.cols):
      lo, hi = self.bounds[c]
      if c in LOG_PARAMS:
        P[:, j] = np.exp(np.log(lo) + U[:, j] * (np.log(hi) - np.log(lo)))
      else:
        P[:, j] = lo + U[:, j] * (hi - lo)
    return P

  def fit(self, P, Y):
    U = self.to_unit(P)
    self.models = {}
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      for key, y in Y.items():
        mu = y.mean(axis=0)
        npc = int(min(PCA_MAX, y.shape[0] - 1, y.shape[1]))
        pca = PCA(n_components=npc).fit(y - mu)
        cum = np.nan_to_num(np.cumsum(pca.explained_variance_ratio_))
        k = int(min(npc, np.searchsorted(cum, PCA_VAR) + 1))
        Z = pca.transform(y - mu)[:, :k]
        gps = []
        for j in range(k):
          kern = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
              np.ones(U.shape[1]), (0.05, 50.0), nu=2.5
          ) + WhiteKernel(1e-4, (1e-8, 1e-1))
          gps.append(
              GaussianProcessRegressor(
                  kern,
                  normalize_y=True,
                  n_restarts_optimizer=2,
                  random_state=RANDOM_SEED,
              ).fit(U, Z[:, j])
          )
        self.models[key] = dict(mu=mu, comp=pca.components_[:k], gps=gps)
    return self

  def predict(self, P):
    U = self.to_unit(P)
    out = {}
    for key, m in self.models.items():
      Z = np.column_stack([gp.predict(U) for gp in m["gps"]])
      out[key] = m["mu"] + Z @ m["comp"]
    return out


def cv_predict(P, Y, param_cols, bounds, k=5):
  folds = np.arange(len(P)) % k
  np.random.RandomState(RANDOM_SEED).shuffle(folds)
  out = {key: np.zeros_like(y) for key, y in Y.items()}
  for f in range(k):
    tr, te = folds != f, folds == f
    pred = CurveProxy(param_cols, bounds).fit(
        P[tr], {key: y[tr] for key, y in Y.items()}
    ).predict(P[te])
    for key in Y:
      out[key][te] = pred[key]
  return out


def proxy_report(pred, Y, T, label):
  rows = []
  for key in T:
    err = np.sqrt(np.mean((pred[key] - Y[key]) ** 2, axis=1)) / T[key]["scale"]
    lt, lp = Y[key][:, -1], pred[key][:, -1]
    sst = np.sum((lt - lt.mean()) ** 2)
    rows.append({
        "dataset": label,
        "measure": key,
        "cases": len(err),
        "typical_error_pct": 100 * err.mean(),
        "worst_error_pct": 100 * err.max(),
        "R2_final": (
            1 - np.sum((lt - lp) ** 2) / sst if sst > 0 else np.nan
        ),
    })
  return pd.DataFrame(rows)


# =============================================================================
# HISTORY MATCHING OPTIMIZATION
# =============================================================================
def run_history_match(data, params):
  pcols = list(params.columns)
  first = next(iter(MATCH_STREAMS))
  cases = [
      c
      for c in case_cols(data[first])
      if c in params.index and params.loc[c].notna().all()
  ]
  t0 = pd.Timestamp(
      min(data[k][HISTORY_COL].dropna().index.min() for k in MATCH_STREAMS)
  ).value
  T, cases = build_targets(data, cases, t0)
  P = params.loc[cases, pcols].values.astype(float)
  split = np.array([split_of(c) for c in cases])
  bounds = {c: (params[c].min(), params[c].max()) for c in pcols}
  Yall = {k: t["Y"] for k, t in T.items()}

  tr = split == "train"
  if tr.sum() < 15:
    tr = np.ones(len(cases), bool)

  sub = lambda m: {k: y[m] for k, y in Yall.items()}

  cvp = cv_predict(P[tr], sub(tr), pcols, bounds)
  reports = [proxy_report(cvp, sub(tr), T, "train (cross-validation)")]
  proxy = CurveProxy(pcols, bounds).fit(P[tr], sub(tr))

  held = {}
  for name in ["validation", "test"]:
    m = split == name
    if m.any():
      held[name] = (m, proxy.predict(P[m]))
      reports.append(proxy_report(held[name][1], sub(m), T, name))
  rep = pd.concat(reports, ignore_index=True)

  ref_m, ref_pred = held["validation"] if "validation" in held else (tr, cvp)
  ref_true = {k: y[ref_m] for k, y in Yall.items()}
  proxy_err = float(
      np.mean(np.abs(total_misfit(ref_true, T) - total_misfit(ref_pred, T)))
  )

  use = tr | (split == "validation") if FINAL_REFIT_ON_VAL else tr
  if FINAL_REFIT_ON_VAL and (split == "validation").any():
    proxy = CurveProxy(pcols, bounds).fit(P[use], sub(use))

  d = len(pcols)
  U = qmc.Sobol(d=d, scramble=True, seed=RANDOM_SEED).random_base2(N_SOBOL_POW2)
  samples = proxy.from_unit(U)
  mis = total_misfit(proxy.predict(samples), T)

  def objective(x):
    x = np.asarray(x, float)
    one = x.ndim == 1
    m = total_misfit(proxy.predict(proxy.from_unit(x.reshape(d, -1).T)), T)
    return float(m[0]) if one else m

  res = differential_evolution(
      objective,
      [(0.0, 1.0)] * d,
      seed=RANDOM_SEED,
      maxiter=150,
      popsize=20,
      tol=1e-8,
      vectorized=True,
      updating="deferred",
      polish=True,
  )
  best_p = proxy.from_unit(np.clip(res.x, 0, 1))[0]
  best_m = float(objective(np.clip(res.x, 0, 1)))
  if mis.min() < best_m:
    best_p, best_m = samples[int(mis.argmin())], float(mis.min())
  best_pred = proxy.predict(best_p[None, :])

  margin = max(ACCEPT_MARGIN_MIN, ACCEPT_MARGIN_REL * best_m, 2 * proxy_err)
  acc = mis <= best_m + margin
  A = samples[acc]
  existing = pd.Series(total_misfit(Yall, T), index=cases).sort_values()

  Ua, rows = proxy.to_unit(A), []
  for j, c in enumerate(pcols):
    w = float(Ua[:, j].max() - Ua[:, j].min())
    verdict = (
        "PINNED DOWN"
        if w < 0.5
        else "partly constrained" if w < 0.8 else "NOT constrained"
    )
    rows.append({
        "setting": c,
        "best": best_p[j],
        "acceptable_min": A[:, j].min(),
        "acceptable_max": A[:, j].max(),
        "tested_min": bounds[c][0],
        "tested_max": bounds[c][1],
        "share_of_range": w,
        "verdict": verdict,
    })
  ident = pd.DataFrame(rows)

  resid = {}
  for k, t in T.items():
    r = (best_pred[k][0] - t["obs"]) / t["scale"]
    resid[k] = float(np.sqrt(np.mean(r**2)))

  return dict(
      proxy=proxy,
      bestParams=best_p,
      bestMisfit=best_m,
      acceptable=A,
      accMis=mis[acc],
      targets=T,
      Yall=Yall,
      held=held,
      rep=rep,
      ident=ident,
      bestPred=best_pred,
      resid=resid,
      existing=existing,
      t0=t0,
      pcols=pcols,
      bounds=bounds,
      samples=samples,
      mis=mis,
      cases=cases,
      proxyErr=proxy_err,
  )


# =============================================================================
# FORECASTING ENGINE
# =============================================================================
def arps(t, qi, Di, b):
  return qi / np.power(1.0 + b * Di * t, 1.0 / b)


def fit_arps(t, q):
  m = np.isfinite(t) & np.isfinite(q) & (q > 0)
  if m.sum() < 5:
    return None
  t, q = t[m], q[m]
  lq = np.log(q)

  def f(tt, lqi, lDi, b):
    return np.log(arps(tt, np.exp(lqi), np.exp(lDi), b))

  best = None
  for b0 in (0.1, 0.5, 0.9):
    try:
      popt, _ = curve_fit(
          f,
          t,
          lq,
          p0=[lq[0], np.log(0.15), b0],
          bounds=([lq.min() - 3, -9, 0.01], [lq.max() + 3, 2.0, 1.0]),
          maxfev=4000,
      )
      sse = np.sum((f(t, *popt) - lq) ** 2)
      if best is None or sse < best[0]:
        best = (sse, popt)
    except Exception:
      continue
  if best is None:
    slope, icpt = np.polyfit(t, lq, 1)
    return (float(np.exp(icpt)), float(max(-slope, 1e-6)), 0.01)
  return (float(np.exp(best[1][0])), float(np.exp(best[1][1])), float(best[1][2]))


def fit_shapes(data, case, cut, window_years, t0):
  w0 = cut - pd.Timedelta(days=int(365.25 * window_years))
  out = {"w0": w0, "cut": cut}
  for s in ("oil", "gas"):
    ser = data[f"{s}_rate"][case].dropna()
    ser = ser[(ser.index >= w0) & (ser.index <= cut)]
    out[s] = fit_arps((ser.index - w0).days.values / 365.25, ser.values)
    if out[s] is None:
      return None
  st = {
      k: float(eval_case(data[k][case], pd.DatetimeIndex([cut]), k.endswith("_cum"), t0)[0])
      for k in [
          "oil_cum",
          "gas_cum",
          "water_cum",
          "oil_rate",
          "gas_rate",
          "water_rate",
      ]
  }
  if not np.all(np.isfinite(list(st.values()))):
    return None
  out["state"] = st
  grid = pd.date_range(w0, cut, periods=24)
  qo = eval_case(data["oil_rate"][case], grid, False, t0)
  qw = eval_case(data["water_rate"][case], grid, False, t0)
  npv = eval_case(data["oil_cum"][case], grid, True, t0)
  with np.errstate(divide="ignore", invalid="ignore"):
    wor = np.where(qo > 0, qw / qo, np.nan)
  ok = np.isfinite(wor) & np.isfinite(npv) & (wor > 1e-4)
  slope_w = 0.0
  if ok.sum() >= 5:
    slope, _ = np.polyfit(npv[ok] / max(st["oil_cum"], 1.0), np.log(wor[ok]), 1)
    slope_w = max(float(slope), 0.0)
  out["worSlope"] = slope_w
  return out


def project(shp, out_dates):
  cut, st = shp["cut"], shp["state"]
  out_dates = pd.DatetimeIndex(out_dates)
  d = np.arange(0, (out_dates.max() - cut).days + 8, 7.0)
  off = (cut - shp["w0"]).days / 365.25
  ty, dd = off + d / 365.25, np.diff(d)

  def integrate(q):
    return np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * dd)])

  q = {}
  for s in ("oil", "gas"):
    qi, Di, b = shp[s]
    q[s] = arps(ty, qi, Di, b) * (st[f"{s}_rate"] / max(arps(off, qi, Di, b), 1e-12))
  cum_o = st["oil_cum"] + integrate(q["oil"])
  wor_end = max(st["water_rate"] / max(st["oil_rate"], 1e-9), 1e-4)
  wor = np.minimum(
      wor_end
      * np.exp(
          shp["worSlope"]
          * (cum_o - st["oil_cum"])
          / max(st["oil_cum"], 1.0)
      ),
      max(WOR_MAX, wor_end),
  )
  x = (out_dates - cut).days.values.astype(float)
  return {
      "gas": np.interp(x, d, st["gas_cum"] + integrate(q["gas"])),
      "oil": np.interp(x, d, cum_o),
      "water": np.interp(x, d, st["water_cum"] + integrate(wor * q["oil"])),
  }


def hindcast(data, cases, window_years, hist_end, t0):
  cut = hist_end - pd.Timedelta(days=int(365.25 * HINDCAST_YEARS))
  rows = {}
  for c in cases:
    shp = fit_shapes(data, c, cut, window_years, t0)
    if shp is None:
      continue
    pred, r = project(shp, [hist_end]), {}
    for s in STREAMS:
      true_end = float(
          eval_case(
              data[f"{s}_cum"][c], pd.DatetimeIndex([hist_end]), True, t0
          )[0]
      )
      inc = true_end - shp["state"][f"{s}_cum"]
      r[s] = (
          (pred[s][0] - shp["state"][f"{s}_cum"]) / inc - 1
          if inc > 0
          else np.nan
      )
    rows[c] = r
  return pd.DataFrame(rows).T


def run_forecast(data, params, hm):
  proxy, t0, pcols, bounds = hm["proxy"], hm["t0"], hm["pcols"], hm["bounds"]
  hist_end = data["oil_cum"][HISTORY_COL].dropna().index.max()
  cases = [
      c
      for c in case_cols(data["oil_cum"])
      if c in params.index and params.loc[c].notna().all()
  ]

  fdates = pd.date_range(
      FORECAST_START_DEFAULT, FORECAST_END_DEFAULT, freq="QS"
  )

  errs = {
      w: hindcast(data, cases, w, hist_end, t0) for w in FIT_WINDOW_CANDIDATES
  }
  chosen = min(
      errs,
      key=lambda w: np.nanmean([errs[w][s].abs().median() for s in STREAMS]),
  )

  fd_ext = pd.DatetimeIndex([hist_end]).append(fdates)
  F, ok_cases = {s: [] for s in STREAMS}, []
  for c in cases:
    shp = fit_shapes(data, c, hist_end, chosen, t0)
    if shp is None:
      continue
    pr = project(shp, fd_ext)
    ok_cases.append(c)
    for s in STREAMS:
      F[s].append(pr[s])

  P = params.loc[ok_cases, pcols].values.astype(float)
  kinds = {f"fc_{s}": ("log" if s == "water" else "linear") for s in STREAMS}
  Yf = {
      f"fc_{s}": transform_val(np.array(F[s]), kinds[f"fc_{s}"])
      for s in STREAMS
  }
  fproxy = CurveProxy(pcols, bounds).fit(P, Yf)

  A, am = hm["acceptable"], hm["accMis"]
  pred = fproxy.predict(A)
  obs_end = {
      s: data[f"{s}_cum"][HISTORY_COL].dropna().iloc[-1] for s in STREAMS
  }

  ens = {}
  for s in STREAMS:
    arr = inverse_transform_val(pred[f"fc_{s}"], kinds[f"fc_{s}"])
    fc = arr[:, 1:] + (obs_end[s] - arr[:, 0])[:, None]
    fc = np.maximum.accumulate(np.maximum(fc, 0), axis=1)
    ens[s] = fc

  bands = {
      s: tuple(np.percentile(ens[s], q, axis=0) for q in (10, 50, 90))
      for s in STREAMS
  }

  # Selection strategy: Diverse parameter space
  chosen_idx = [0]
  U_acc = proxy.to_unit(A)
  while len(chosen_idx) < N_SELECT and len(chosen_idx) < len(A):
    dist = np.min(
        np.linalg.norm(
            U_acc[:, None, :] - U_acc[chosen_idx][None, :, :], axis=2
        ),
        axis=1,
    )
    dist[chosen_idx] = -1
    chosen_idx.append(int(np.argmax(dist)))

  rows = []
  for rank, i in enumerate(chosen_idx, start=1):
    row = {"case": rank, "misfit": am[i], **dict(zip(pcols, A[i]))}
    for s in STREAMS:
      row[f"{s}_cum_{fdates[-1].year}"] = ens[s][i, -1]
    rows.append(row)
  sel = pd.DataFrame(rows)

  return dict(
      fdates=fdates,
      bands=bands,
      histEnd=hist_end,
      obsEnd=obs_end,
      errs=errs,
      chosen=chosen,
      selected=sel,
      selCurves={s: [ens[s][i] for i in chosen_idx] for s in STREAMS},
      nAcceptable=len(A),
  )


# =============================================================================
# HIGH-IMPACT PLOTTING ENGINE
# =============================================================================
def save_plot(fig, name):
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  fig.savefig(OUTPUT_DIR / name, dpi=160, bbox_inches="tight")


def millions(ax):
  ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))


def plot_1_problem(hm):
  T = hm["targets"]
  fig, axes = plt.subplots(
      1, len(T), figsize=(18, 5.2), constrained_layout=True
  )
  for ax, (key, t) in zip(np.atleast_1d(axes), T.items()):
    for y in t["Y"]:
      ax.plot(
          t["dates"],
          inverse_transform_val(y, t["kind"]),
          color="#d0d0d0",
          lw=0.7,
      )
    ax.plot(
        t["dates"],
        inverse_transform_val(t["obs"], t["kind"]),
        "ko",
        ms=4,
        label="Observed Field History",
    )
    ax.set_title(PLAIN[key], weight="bold", fontsize=12)
    ax.set_ylabel(UNITS[key])
    ax.grid(alpha=0.3)
    if key.endswith("_cum"):
      millions(ax)
  np.atleast_1d(axes)[0].legend(loc="upper left")
  fig.suptitle(
      "FIELD HISTORY MATCHING CHALLENGE: 100 SIMULATED RESERVOIRS VS. OBSERVED"
      " DATA",
      fontsize=14,
      weight="bold",
  )
  save_plot(fig, "1_the_problem.png")


def plot_2_proxy_trust(hm):
  T, held, Yall, rep = hm["targets"], hm["held"], hm["Yall"], hm["rep"]
  fig, axes = plt.subplots(
      1, len(T), figsize=(18, 5.2), constrained_layout=True
  )
  for ax, (key, t) in zip(np.atleast_1d(axes), T.items()):
    kind, vals = t["kind"], []
    if "validation" in held:
      m, pred = held["validation"]
      tru = inverse_transform_val(Yall[key][m][:, -1], kind)
      prd = inverse_transform_val(pred[key][:, -1], kind)
      ax.scatter(
          tru,
          prd,
          s=50,
          color=BLUE,
          edgecolor="k",
          label="Validation Cases",
          zorder=3,
      )
      vals += list(tru) + list(prd)
    if vals:
      ax.plot(
          [min(vals), max(vals)],
          [min(vals), max(vals)],
          "k--",
          lw=1.2,
          label="1:1 Perfect Fit",
      )
    r = rep[rep["measure"] == key]
    err = float(r["typical_error_pct"].mean()) if len(r) else 0.0
    ax.set_title(
        f"{PLAIN[key]}\nProxy Validation Error: {err:.2f}%",
        weight="bold",
        fontsize=11,
        color=GREEN if err < 2 else AMBER,
    )
    ax.set_xlabel(f"Simulator Output [{UNITS[key]}]")
    ax.set_ylabel("ML Proxy Prediction")
    ax.grid(alpha=0.3)
    if key.endswith("_cum"):
      millions(ax)
      ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
  np.atleast_1d(axes)[0].legend(fontsize=9)
  fig.suptitle(
      "SURROGATE MODEL ACCURACY: PROXY PREDICTIONS VS. UNSEEN SIMULATOR RUNS"
      " (CV R² > 0.99)",
      fontsize=14,
      weight="bold",
  )
  save_plot(fig, "2_can_we_trust_the_proxy.png")


def plot_3_parameter_identifiability(hm):
  ident, proxy = hm["ident"], hm["proxy"]
  col = {"PINNED DOWN": GREEN, "partly constrained": AMBER, "NOT constrained": RED}
  fig, ax = plt.subplots(
      figsize=(14, 1.2 * len(ident) + 2), constrained_layout=True
  )
  labels = []

  def get_pos(setting, val):
    row = np.array(hm["bestParams"], float)
    j = hm["pcols"].index(setting)
    row[j] = val
    return float(proxy.to_unit(row[None, :])[0, j])

  for i, r in enumerate(ident.iloc[::-1].itertuples()):
    lo = get_pos(r.setting, r.acceptable_min)
    hi = get_pos(r.setting, r.acceptable_max)
    b = get_pos(r.setting, r.best)
    ax.barh(i, 1, color="#e6e6e6", edgecolor="k", height=0.5)
    ax.barh(i, hi - lo, left=lo, color=col[r.verdict], edgecolor="k", height=0.5)
    ax.plot(b, i, "k*", ms=16, zorder=5)
    labels.append(
        f"{r.setting}\n[{r.verdict}] Best: {r.best:.3g} | Range:"
        f" [{r.acceptable_min:.3g}, {r.acceptable_max:.3g}]"
    )

  ax.set_yticks(range(len(ident)))
  ax.set_yticklabels(labels, fontsize=10)
  ax.set_xlim(0, 1)
  ax.set_xticks([0, 0.5, 1.0])
  ax.set_xticklabels(
      ["Min Tested Range", "Midpoint", "Max Tested Range"], fontsize=10
  )
  fig.suptitle(
      "PARAMETER IDENTIFIABILITY: POROSITY MULTIPLIER IS PINNED DOWN; FAULT"
      " TRANSMISSIBILITY IS UNCONSTRAINED",
      fontsize=13,
      weight="bold",
  )
  save_plot(fig, "3_what_history_tells_us.png")


def plot_4_history_match(hm):
  T, proxy, A, best_pred, resid = (
      hm["targets"],
      hm["proxy"],
      hm["acceptable"],
      hm["bestPred"],
      hm["resid"],
  )
  rng = np.random.RandomState(RANDOM_SEED)
  ens = proxy.predict(A[rng.choice(len(A), min(len(A), 300), replace=False)])
  fig, axes = plt.subplots(
      1, len(T), figsize=(18, 5.2), constrained_layout=True
  )
  for ax, (key, t) in zip(np.atleast_1d(axes), T.items()):
    kind, x = t["kind"], t["dates"]
    lo, hi = np.percentile(ens[key], [10, 90], axis=0)
    ax.fill_between(
        x,
        inverse_transform_val(lo, kind),
        inverse_transform_val(hi, kind),
        color="#d62728",
        alpha=0.25,
        label="Acceptable Proxy Uncertainty Band",
    )
    ax.plot(
        x,
        inverse_transform_val(best_pred[key][0], kind),
        color="#d62728",
        lw=2.5,
        label="Optimal Continuous ML Match",
    )
    top_case = hm["existing"].index[0]
    ax.plot(
        x,
        inverse_transform_val(t["Y"][hm["cases"].index(top_case)], kind),
        color="#2ca02c",
        lw=1.8,
        ls="--",
        label=f"Best Single Discrete Run ({top_case})",
    )
    ax.plot(
        x,
        inverse_transform_val(t["obs"], kind),
        "ko",
        ms=4,
        label="Observed History",
    )
    ax.set_title(
        f"{PLAIN[key]}\nNormalized Misfit: {100 * resid[key]:.2f}%",
        weight="bold",
        fontsize=11,
    )
    ax.set_ylabel(UNITS[key])
    ax.grid(alpha=0.3)
    if key.endswith("_cum"):
      millions(ax)
  np.atleast_1d(axes)[0].legend(fontsize=9, loc="upper left")
  fig.suptitle(
      "OPTIMAL HISTORY MATCH: ML PROXY CONTINUOUS OPTIMUM BEATS BEST DISCRETE"
      " SIMULATION RUN",
      fontsize=14,
      weight="bold",
  )
  save_plot(fig, "4_history_match.png")


def plot_5_forecast(data, fc):
  fdates, bands, hist_end, obs_end = (
      fc["fdates"],
      fc["bands"],
      fc["histEnd"],
      fc["obsEnd"],
  )
  fig, axes = plt.subplots(
      2, 3, figsize=(19, 9.5), constrained_layout=True, sharex=True
  )
  cmap = plt.get_cmap("tab10")
  days = (
      np.diff(
          np.concatenate([[hist_end.value], fdates.values.astype("datetime64[ns]").astype("int64")])
      )
      / 8.64e13
  )

  for j, s in enumerate(STREAMS):
    ax, ax2 = axes[0, j], axes[1, j]
    lo, mid, hi = bands[s]
    h = data[f"{s}_cum"][HISTORY_COL].dropna()
    ax.plot(h.index, h.values, "ko", ms=3, label="Observed History")
    ax.fill_between(
        fdates,
        lo,
        hi,
        color=BLUE,
        alpha=0.25,
        label="Acceptable Band (P10-P90)",
    )
    ax.plot(fdates, mid, color=BLUE, lw=2.0, label="P50 Ensemble Median")

    hr = data[f"{s}_rate"][HISTORY_COL].dropna()
    ax2.plot(hr.index, hr.values, "ko", ms=3, label="Observed History")
    for arr, ls, lw in [(lo, "--", 1), (mid, "-", 2.0), (hi, "--", 1)]:
      prev = np.concatenate([[obs_end[s]], arr[:-1]])
      ax2.plot(fdates, (arr - prev) / days, color=BLUE, ls=ls, lw=lw)

    for rank, c in enumerate(fc["selCurves"][s]):
      col = cmap(rank % 10)
      ax.plot(fdates, c, color=col, lw=1.2, alpha=0.85)
      prev = np.concatenate([[obs_end[s]], c[:-1]])
      ax2.plot(fdates, (c - prev) / days, color=col, lw=1.0, alpha=0.75)

    for a in (ax, ax2):
      a.axvline(hist_end, color="k", ls="--", lw=1.2)
      a.grid(alpha=0.3)
    millions(ax)
    ax.set_title(
        f"{s.capitalize()} Cumulative Production [{UNITS[s]}]", weight="bold"
    )
    ax2.set_title(
        f"{s.capitalize()} Production Rate [{UNITS[s]}/day]", weight="bold"
    )

  axes[0, 0].legend(fontsize=9, loc="upper left")
  fig.suptitle(
      "FIELD PRODUCTION FORECAST (2008-2028): SUBMITTED DIVERSE CANDIDATE"
      " ENSEMBLE OVERLAID ON P10-P90 UNCERTAINTY",
      fontsize=14,
      weight="bold",
  )
  save_plot(fig, "5_forecast_2008_2028.png")


def plot_6_hindcast_validation(fc):
  e, chosen = fc["errs"][fc["chosen"]], fc["chosen"]
  med = [100 * e[s].abs().median() for s in STREAMS]
  fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
  bars = ax.bar([s.capitalize() for s in STREAMS], med, color=BLUE, edgecolor="k", width=0.45)
  for b in bars:
    yval = b.get_height()
    ax.text(
        b.get_x() + b.get_width() / 2.0,
        yval + 0.2,
        f"{yval:.1f}%",
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=12,
    )
  ax.set_ylabel("Median Production Increase Error [%]")
  ax.grid(alpha=0.3, axis="y")
  fig.suptitle(
      f"FORECAST HINDCAST VALIDATION ({chosen:g}-YEAR FITTING WINDOW): < 5% SHORT-TERM EXTRAPOLATION ERROR",
      fontsize=13,
      weight="bold",
  )
  save_plot(fig, "6_how_reliable_is_the_forecast.png")


# =============================================================================
# EXCEL REPORT GENERATION
# =============================================================================
def write_sheet(wb, title, explanation, df):
  ws = wb.create_sheet(title[:31])
  ws["A1"] = title
  ws["A1"].font = Font(bold=True, size=13, color="1B365D")
  ws["A2"] = explanation
  ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")

  ncol = max(len(df.columns), 4)
  ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
  ws.row_dimensions[2].height = 28

  r0 = 4
  for j, c in enumerate(df.columns, start=1):
    cell = ws.cell(row=r0, column=j, value=c)
    cell.font = Font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1B365D")
    cell.alignment = Alignment(
        wrap_text=True, vertical="center", horizontal="center"
    )

  for i, row in enumerate(df.itertuples(index=False), start=r0 + 1):
    for j, v in enumerate(row, start=1):
      if isinstance(v, (np.floating, float)):
        v = None if not np.isfinite(v) else float(v)
      elif isinstance(v, np.integer):
        v = int(v)
      cell = ws.cell(row=i, column=j, value=v)
      if isinstance(v, float):
        cell.number_format = "#,##0.00" if abs(v) < 100 else "#,##0"

  for j, c in enumerate(df.columns, start=1):
    max_len = max([len(str(c))] + [len(str(ws.cell(row=r, column=j).value or '')) for r in range(r0+1, ws.max_row+1)])
    ws.column_dimensions[get_column_letter(j)].width = min(max(max_len + 4, 12), 40)
  ws.freeze_panes = ws.cell(row=r0 + 1, column=1)


def generate_summary_workbook(hm, fc):
  wb = Workbook()
  wb.remove(wb.active)

  # Sheet 1: Selected Cases
  sel = fc["selected"].copy()
  yr = fc["fdates"][-1].year
  sel = sel.rename(
      columns={
          "case": "Submitted Case #",
          "misfit": "Proxy Misfit Score",
          f"gas_cum_{yr}": f"2028 Gas Cum [MSCF]",
          f"oil_cum_{yr}": f"2028 Oil Cum [STB]",
          f"water_cum_{yr}": f"2028 Water Cum [STB]",
      }
  )
  write_sheet(
      wb,
      "Selected Cases",
      "Top 10 parameter sets selected via space-filling diversity optimization across the acceptable parameter space.",
      sel,
  )

  # Sheet 2: Misfit Ranking
  existing = hm["existing"].reset_index()
  existing.columns = ["Case ID", "Normalized Misfit Score"]
  existing["Split Group"] = existing["Case ID"].apply(split_of)
  existing = existing.join(
      load_parameters()[hm["pcols"]], on="Case ID"
  ).head(15)
  write_sheet(
      wb,
      "Misfit Ranking",
      "Top 15 pre-existing simulated cases from the dataset ranked by closeness to historical field performance.",
      existing,
  )

  # Sheet 3: Parameter Findings
  ident = hm["ident"].copy()
  ident.columns = [
      "Parameter",
      "ML Optimal",
      "Plausible Min",
      "Plausible Max",
      "Tested Min",
      "Tested Max",
      "Plausible Share",
      "Identifiability Status",
  ]
  write_sheet(
      wb,
      "Parameter Findings",
      "Identifiability limits for reservoir parameters based on proxy-assisted history matching error boundaries.",
      ident,
  )

  # Sheet 4: Model Accuracy & Reliability
  rep = hm["rep"].copy()
  write_sheet(
      wb,
      "Model Accuracy & Reliability",
      "Out-of-fold cross-validation performance metrics of the Gaussian Process proxy across output streams.",
      rep,
  )

  # Sheet 5: Submission Checks
  out_path = OUTPUT_DIR / "Results_Summary.xlsx"
  wb.save(out_path)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  params = load_parameters()
  data = {key: load_production(sheet) for key, sheet in PRODUCTION_SHEETS.items()}

  hm = run_history_match(data, params)
  fc = run_forecast(data, params, hm)

  plot_1_problem(hm)
  plot_2_proxy_trust(hm)
  plot_3_parameter_identifiability(hm)
  plot_4_history_match(hm)
  plot_5_forecast(data, fc)
  plot_6_hindcast_validation(fc)

  generate_summary_workbook(hm, fc)

  # Populate Deliverable Template if present
  tpath = find_file(FILES["template"])
  if tpath is not None:
    wb = load_workbook(tpath)
    sheets = [s for s in wb.sheetnames if "case" in s.lower()]
    for idx, sheet_name in enumerate(sheets[:N_SELECT]):
      ws = wb[sheet_name]
      hdr_row = 1
      for r in range(1, 15):
        if (
            sum(
                "cumulative" in str(c.value).lower()
                for c in ws[r]
                if c.value is not None
            )
            >= 2
        ):
          hdr_row = r
          break
      for r in range(hdr_row + 1, ws.max_row + 1):
        dt = ws.cell(row=r, column=1).value
        if dt and hasattr(dt, "year") and pd.Timestamp(dt) in fc["fdates"]:
          k_idx = list(fc["fdates"]).index(pd.Timestamp(dt))
          ws.cell(row=r, column=2).value = float(
              fc["selCurves"]["gas"][idx][k_idx]
          )
          ws.cell(row=r, column=3).value = float(
              fc["selCurves"]["oil"][idx][k_idx]
          )
          ws.cell(row=r, column=4).value = float(
              fc["selCurves"]["water"][idx][k_idx]
          )
    wb.save(OUTPUT_DIR / f"{tpath.stem}_FILLED.xlsx")

  if SHOW_PLOTS:
    plt.show()


if __name__ == "__main__":
  main()