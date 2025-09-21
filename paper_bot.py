import time, json, os
from pathlib import Path
from datetime import datetime
import pandas as pd, numpy as np
import ccxt, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SYMBOL = "BTC/USDT"
TF     = "1m"   # test quickly. Later change to "15m" if you prefer.
FEE    = 0.0010
SLIP   = 0.0002

STATE_F = Path("logs/state.json")
LOG_F   = Path("logs/paper_trades.csv")
MODEL_F = Path("models/logreg_h3.joblib")
CONF_F  = Path("config/regime_thresholds.json")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("config", exist_ok=True)

def fetch_ohlcv(limit=400):
    ex = ccxt.binance({
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"adjustForTimeDifference": True}
    })
    for attempt in range(5):
        try:
            print(f"[fetch] try {attempt+1}", flush=True)
            o = ex.fetch_ohlcv(SYMBOL, timeframe=TF, limit=limit)
            df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").tz_convert("Asia/Tokyo")
            return df
        except ccxt.NetworkError as e:
            print(f"[fetch] network error: {e}; backoff...", flush=True)
        except Exception as e:
            print(f"[fetch] error: {e}; backoff...", flush=True)
        time.sleep(2**attempt)
    raise RuntimeError("fetch_ohlcv failed after retries")

def make_features(df):
    ret = np.log(df["close"]).diff()
    X = pd.DataFrame(index=df.index)
    X["r1"] = ret
    X["r2"] = ret.shift(1)
    X["r3"] = ret.shift(2)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    X["macd"] = ema12 - ema26
    X["vol20"] = ret.rolling(20).std()
    X["vol60"] = ret.rolling(60).std()
    X["bbw"] = (df["close"].rolling(20).mean().sub(df["close"])) / df["close"].rolling(20).std()
    return X

def train_model_if_missing():
    if MODEL_F.exists():
        return
    df = fetch_ohlcv(limit=3000)
    X  = make_features(df).dropna()
    H  = 3
    y  = (np.log(df["close"]).shift(-H) - np.log(df["close"]) > 0).astype(int).reindex_like(X).dropna()
    X  = X.loc[y.index]
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=300, C=0.5))])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_F)
    print("model trained & saved:", MODEL_F, flush=True)

def build_conf_if_missing():
    if CONF_F.exists():
        return
    tmp = fetch_ohlcv(limit=800)
    v20 = np.log(tmp["close"]).diff().rolling(20).std()
    q1, q2 = float(v20.quantile(0.33)), float(v20.quantile(0.66))
    conf = {"q1": q1, "q2": q2, "thresholds": {"LOW": 0.55, "MID": 0.55, "HIGH": 0.60}}
    CONF_F.write_text(json.dumps(conf, indent=2))
    print("config created:", CONF_F, flush=True)

def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {"last_ts": None, "pos": 0, "equity": 1.0}

def save_state(s):
    STATE_F.write_text(json.dumps(s))

def append_log(row):
    hdr = not LOG_F.exists()
    import csv
    with LOG_F.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if hdr:
            w.writeheader()
        w.writerow(row)

def decide_threshold(v20, conf):
    q1, q2 = conf["q1"], conf["q2"]
    ths = conf["thresholds"]
    if v20 <= q1:
        return ths["LOW"]
    if v20 <= q2:
        return ths["MID"]
    return ths["HIGH"]

def _tf_seconds(tf):
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    return 900

def main():
    train_model_if_missing()
    build_conf_if_missing()
    model = joblib.load(MODEL_F)
    conf  = json.loads(CONF_F.read_text())
    state = load_state()
    print("paper bot start:", datetime.now(), flush=True)

    while True:
        try:
            print("[loop] start", datetime.now(), flush=True)
            df = fetch_ohlcv(limit=400)
            print("[loop] data ok", flush=True)
            X = make_features(df).dropna()
            print(f"[loop] features ok: {X.shape}", flush=True)
            if len(X) < 3:
                time.sleep(10)
                continue

            last_ts = str(X.index[-1])
            if state["last_ts"] == last_ts:
                print(f"[loop] no new bar yet ({last_ts}); sleeping...", flush=True)
                time.sleep(10)
                continue

            x = X.iloc[-1:]
            proba_up = float(model.predict_proba(x)[0, 1])
            v20 = float(x["vol20"].values[0])
            th  = float(decide_threshold(v20, conf))
            sig = int(1 if proba_up > th else (-1 if proba_up < (1 - th) else 0))

            close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2])
            ret = float(np.log(close) - np.log(prev_close))

            turnover = int(1 if sig != state["pos"] else 0)
            cost = float(turnover * (FEE + SLIP))

            pnl = float(state["pos"] * ret - cost)
            state["equity"] = float(state["equity"] * (1 + pnl))

            row = {
                "ts": last_ts, "price": close, "proba_up": proba_up,
                "vol20": v20, "th": th, "signal": sig, "prev_pos": int(state["pos"]),
                "ret": ret, "pnl": pnl, "equity": float(state["equity"])
            }
            print(f"[loop] proba_up={proba_up:.3f} v20={v20:.6f} th={th:.2f} sig={sig} ret={ret:.6f} pnl={pnl:.6f} eq={state['equity']:.6f}", flush=True)

            append_log(row)
            state["pos"] = sig
            state["last_ts"] = last_ts
            save_state(state)
            print("[loop] logged & state saved", flush=True)

            sec = _tf_seconds(TF)
            now = time.time()
            sleep_s = max(5, sec - (int(now) % sec) + 2)
            time.sleep(sleep_s)

        except KeyboardInterrupt:
            print("stopped by user", flush=True)
            break
        except Exception as e:
            print("error:", e, flush=True)
            time.sleep(10)

if __name__ == "__main__":
    main()
