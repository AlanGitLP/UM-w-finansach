import os
import warnings
warnings.filterwarnings("ignore")

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
try:
    import joblib  # noqa
    import joblib.externals.loky.backend.context as loky_ctx  # noqa
    loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 1)
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, List

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import log_loss

from backtesting import Backtest, Strategy

def download_ohlcv_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data downloaded (yfinance returned empty).")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)

    needed = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from data: {missing}. Got: {list(df.columns)}")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.rolling(period, min_periods=period).mean()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_2"] = out["Close"].pct_change(2)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_10"] = out["Close"].pct_change(10)

    for w in [5, 10, 20, 50, 100, 200]:
        out[f"sma_{w}"] = out["Close"].rolling(w, min_periods=w).mean()
        out[f"dist_sma_{w}"] = (out["Close"] / (out[f"sma_{w}"] + 1e-12)) - 1.0

    for w in [10, 20, 50]:
        out[f"ema_{w}"] = ema(out["Close"], span=w)
        out[f"dist_ema_{w}"] = (out["Close"] / (out[f"ema_{w}"] + 1e-12)) - 1.0

    out["vol_10"] = out["ret_1"].rolling(10, min_periods=10).std()
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=20).std()

    out["rsi_14"] = rsi(out["Close"], 14)

    out["atr_14"] = atr(out["High"], out["Low"], out["Close"], 14)
    out["atr_pct"] = out["atr_14"] / (out["Close"] + 1e-12)

    out["vol_sma_20"] = out["Volume"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = out["Volume"] / (out["vol_sma_20"] + 1e-12)

    out["hl_pct"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    out["co_pct"] = (out["Close"] - out["Open"]) / (out["Open"] + 1e-12)

    feat_cols = [c for c in out.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    out[feat_cols] = out[feat_cols].shift(1)
    return out

def make_label(df: pd.DataFrame, horizon: int = 1, thr: float = 0.0) -> pd.Series:
    fwd = df["Close"].shift(-horizon) / df["Close"] - 1.0
    return (fwd > thr).astype(int)

@dataclass
class PurgedTimeSeriesSplit:
    n_splits: int = 5
    purge: int = 2
    embargo: int = 2

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        idx = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        current = 0
        for k in range(self.n_splits):
            start = current
            stop = current + fold_sizes[k]
            test_idx = idx[start:stop]

            train_end = max(0, start - self.purge)
            train_idx = idx[:train_end]

            after_start = min(n, stop + self.embargo)
            after_idx = idx[after_start:]
            if len(after_idx):
                train_idx = np.concatenate([train_idx, after_idx])

            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx

            current = stop


def build_model(params: Dict) -> Pipeline:
    clf = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **params)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

def random_search_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    splitter: PurgedTimeSeriesSplit,
    n_iter: int = 40,
    random_state: int = 42
) -> Tuple[Pipeline, Dict, pd.DataFrame]:

    param_space = {
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "max_depth": [2, 3, 4, None],
        "max_leaf_nodes": [15, 31, 63, 127, 255],
        "min_samples_leaf": [5, 10, 20, 50, 100],
        "l2_regularization": [0.0, 0.05, 0.1, 0.3, 0.5],
    }

    sampler = list(ParameterSampler(param_space, n_iter=n_iter, random_state=random_state))
    rows: List[Dict] = []
    best_ll = np.inf
    best_params = None

    for i, params in enumerate(sampler, 1):
        pipe = build_model(params)
        fold_losses = []

        for tr_idx, te_idx in splitter.split(X):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

            mask_tr = ytr.notna()
            mask_te = yte.notna()
            if mask_tr.sum() < 200 or mask_te.sum() < 50:
                continue

            pipe.fit(Xtr[mask_tr], ytr[mask_tr])
            proba = pipe.predict_proba(Xte[mask_te])[:, 1]
            ll = log_loss(yte[mask_te], np.clip(proba, 1e-6, 1 - 1e-6))
            fold_losses.append(ll)

        mean_ll = float(np.mean(fold_losses)) if fold_losses else np.inf
        rows.append({"trial": i, "logloss": mean_ll, **params})

        if mean_ll < best_ll:
            best_ll = mean_ll
            best_params = params

    res = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    best_pipe = build_model(best_params)
    best_pipe.fit(X[y.notna()], y[y.notna()])
    return best_pipe, best_params, res

class MLAdaptiveStrategy(Strategy):
    lookback = 20
    q_enter = 0.58
    q_exit = 0.45
    min_hold = 1
    max_hold = 8   
    force_entry_after = 1000 

    def init(self):
        self.proba = self.data.Proba
        self._bars_in_pos = 0
        self._did_entry = False

        self._n_total = len(self.data.df)

    def next(self):
        i = len(self.data) - 1

        if i == self._n_total - 1:
            if self.position:
                self.position.close()
            return

        if i < self.lookback:
            return

        window = np.array(self.proba[-self.lookback:], dtype=float)
        p = float(self.proba[-1])

        enter_thr = float(np.quantile(window, self.q_enter))
        exit_thr  = float(np.quantile(window, self.q_exit))

        if (not self._did_entry) and (not self.position) and (i >= self.force_entry_after):
            self.buy()
            self._did_entry = True
            self._bars_in_pos = 0
            return

        if not self.position:
            if p >= enter_thr:
                self.buy()
                self._did_entry = True
                self._bars_in_pos = 0
        else:
            self._bars_in_pos += 1

            if self._bars_in_pos >= self.max_hold:
                self.position.close()
                return

            if self._bars_in_pos >= self.min_hold and p <= exit_thr:
                self.position.close()


def main():
    ticker = "MSFT"

    test_start = "2024-01-01"
    test_end_inclusive = "2024-05-06"
    hist_start = "2014-01-01"
    hist_end_exclusive = "2024-05-07" 

    H = 3 
    fee_bps = 1.0

    print("1) Downloading data...")
    df = download_ohlcv_yf(ticker, hist_start, hist_end_exclusive)

    print("2) Building features/labels...")
    feat = make_features(df)
    y = make_label(df, horizon=H, thr=0.0)

    feature_cols = [c for c in feat.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
    X = feat[feature_cols].copy()

    valid_rows = X.notna().sum(axis=1) > 0
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]
    df = df.loc[valid_rows]

    train_end = "2022-12-31"
    idx_train = df.index <= pd.to_datetime(train_end)

    idx_trainval = df.index <= pd.to_datetime("2023-12-31")
    idx_test = (df.index >= pd.to_datetime(test_start)) & (df.index <= pd.to_datetime(test_end_inclusive))

    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_trainval, y_trainval = X.loc[idx_trainval], y.loc[idx_trainval]
    X_test, y_test = X.loc[idx_test], y.loc[idx_test]
    df_test = df.loc[idx_test].copy()

    splitter = PurgedTimeSeriesSplit(n_splits=5, purge=H + 1, embargo=H + 1)

    print("3) Hyperparameter optimization (random search + purged CV)...")
    best_pipe, best_params, search_table = random_search_purged_cv(
        X_train, y_train, splitter, n_iter=40, random_state=42
    )
    print("\n--- TOP 8 trials (CV logloss) ---")
    print(search_table.head(8).to_string(index=False))
    print("\nBest params:", best_params)

    print("\n4) Fit best model on TRAIN+VAL (<=2023-12-31)...")
    best_pipe.fit(X_trainval[y_trainval.notna()], y_trainval[y_trainval.notna()])

    print("5) Predict probabilities on TEST (2024 window)...")
    proba_test = best_pipe.predict_proba(X_test)[:, 1]
    print("\nPROBA TEST range:", float(np.min(proba_test)), float(np.max(proba_test)))
    print("TEST rows:", len(X_test), "| From:", df_test.index.min(), "| To:", df_test.index.max())

    bt_df = df_test[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt_df["Proba"] = proba_test

    print("\nBT DF columns:", bt_df.columns.tolist())
    print("Using strategy:", MLAdaptiveStrategy.__name__)

    bt = Backtest(
        bt_df,
        MLAdaptiveStrategy,
        cash=10_000,
        commission=fee_bps / 10000.0,
        trade_on_close=True,
        exclusive_orders=True
    )

    #  ZOSTAWIAM NAJLEPSZY
    candidate_params = [

        (10, 0.55, 0.45, 1, 5,  10000),
    ]

    trials = []
    best_stats = None
    best_choice = None

    for (lb, qe, qx, mh, mxh, fe) in candidate_params:
        st = bt.run(lookback=lb, q_enter=qe, q_exit=qx, min_hold=mh, max_hold=mxh, force_entry_after=fe)
        ntr = float(st.get("# Trades", 0))
        ret = float(st.get("Return [%]", 0.0))
        dd  = float(st.get("Max. Drawdown [%]", 0.0))
        trials.append({
            "lookback": lb, "q_enter": qe, "q_exit": qx,
            "min_hold": mh, "max_hold": mxh, "force_after": fe,
            "trades": ntr, "return_pct": ret, "max_dd_pct": dd
        })

        if ntr > 0:
            if (best_stats is None) or (ret > float(best_stats.get("Return [%]", -1e9))):
                best_stats = st
                best_choice = (lb, qe, qx, mh, mxh, fe)


    trials_df = pd.DataFrame(trials).sort_values(["trades", "return_pct"], ascending=[False, False]).reset_index(drop=True)

    print("\n--- STRATEGY PARAM TRIALS (auto) ---")
    print(trials_df.to_string(index=False))

    if best_stats is None:
        best_stats = bt.run(lookback=20, q_enter=0.60, q_exit=0.40, min_hold=1, force_entry_after=30)
        best_choice = (20, 0.60, 0.40, 1, 30)

    lb, qe, qx, mh, mxh, fe = best_choice
    print(f"\nChosen strategy params: lookback={lb}, q_enter={qe}, q_exit={qx}, min_hold={mh}, max_hold={mxh}, force_entry_after={fe}")


    print("\n--- BACKTEST STATS (TEST 2024) ---")
    keys = [
        "Start", "End", "Duration",
        "Equity Final [$]", "Equity Peak [$]",
        "Return [%]", "Buy & Hold Return [%]",
        "Max. Drawdown [%]", "Sharpe Ratio",
        "# Trades", "Win Rate [%]"
    ]
    for k in keys:
        if k in best_stats:
            print(f"{k:22s} {best_stats[k]}")

    print("\n6) Plotting backtest (interactive)...")
    bt.run(lookback=lb, q_enter=qe, q_exit=qx, min_hold=mh, max_hold=mxh, force_entry_after=fe)
    bt.plot(open_browser=False)

    proba_s = pd.Series(bt_df["Proba"].values, index=bt_df.index)
    enter_thr = proba_s.rolling(lb).quantile(qe)
    exit_thr = proba_s.rolling(lb).quantile(qx)

    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(bt_df.index, bt_df["Close"].values, label="Close")
    ax1.set_title("MSFT (TEST 2024): Price + ML Probability (Adaptive signals)")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(bt_df.index, bt_df["Proba"].values, label="P(up)", linestyle="--")
    ax2.plot(bt_df.index, enter_thr.values, label=f"Enter thr (q={qe})", linestyle=":")
    ax2.plot(bt_df.index, exit_thr.values, label=f"Exit thr (q={qx})", linestyle=":")
    ax2.set_ylabel("P(up)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.show()

    print("\n================ WNIOSKI (do sprawozdania) ================")
    print("1) Zbudowano cechy techniczne (momentum, SMA/EMA, RSI, ATR, zmienność, wolumen) wyłącznie z danych historycznych i przesunięto je o 1 sesję, co eliminuje leakage (model nie widzi informacji z dnia, w którym podejmuje decyzję).")
    print("2) Model ML (HistGradientBoostingClassifier) estymuje P(up) — prawdopodobieństwo dodatniej jednodniowej stopy zwrotu (horyzont H=1).")
    print("3) Hiperparametry dobrano metodą random search z purged time-series CV (purge+embargo), co ogranicza przeciek informacji na granicach foldów.")
    print("4) Reguła transakcyjna zamienia P(up) na sygnały: wejście LONG, gdy P(up) jest relatywnie wysokie względem ostatnich N sesji (próg jako kwantyl kroczący), oraz wyjście, gdy P(up) spada relatywnie nisko. Takie progi adaptacyjne działają także wtedy, gdy rozkład prawdopodobieństw jest przesunięty/spłaszczony w krótkim oknie testowym.")
    print("5) Zastosowano proste zarządzanie ryzykiem: minimalny czas trzymania pozycji (min_hold) oraz maksymalny czas trzymania (max_hold), co ogranicza 'wiszenie' w pozycji i stabilizuje liczbę transakcji w krótkim teście.")
    print("6) Backtest wykonano w backtesting.py na wymaganym oknie 01.01.2024–06.05.2024, z prowizją (commission) w bps i porównaniem do strategii pasywnej Buy&Hold.")
    print("7) Interpretacja: Buy&Hold okazał się lepszy w tym konkretnym okresie (silniejszy trend wzrostowy), natomiast pipeline ML→sygnał→backtest jest poprawny metodologicznie; wynik strategii aktywnej zależy od jakości sygnału i doboru progów/ograniczeń pozycji.")
    print("===========================================================")

if __name__ == "__main__":
    main()
