"""Machine-learning core: data prep, training, evaluation and tree utilities.

Everything here is plain Python (no Streamlit state) so it can be tested and reused.
A "bundle" is the dict the app passes around for one model + its metadata.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import uuid
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
from sklearn.base import clone, is_classifier
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             median_absolute_error, precision_score, r2_score,
                             recall_score, roc_curve)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import theme as T

SEED = 42

# --------------------------------------------------------------------------- #
# Friendly names
# --------------------------------------------------------------------------- #
_FRIENDLY = {
    "pressure": ("Atmospheric pressure", "hPa"),
    "global_radiation": ("Global radiation", ""),
    "temp_mean(c)": ("Mean temperature", "°C"),
    "temp_min(c)": ("Minimum temperature", "°C"),
    "temp_max(c)": ("Maximum temperature", "°C"),
    "wind_speed": ("Wind speed", ""),
    "wind_bearing": ("Wind bearing", "°"),
}


def label_for(name: str) -> str:
    lab, unit = _FRIENDLY.get(str(name).lower(), (str(name), ""))
    return f"{lab} ({unit})" if unit else lab


def smart_step(lo: float, hi: float) -> float:
    """A sensible number_input step for a feature spanning [lo, hi]."""
    rng = float(hi) - float(lo)
    if not np.isfinite(rng) or rng <= 0:
        return 1.0
    for step in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0):
        if step >= rng / 200:
            return step
    return 100.0


# --------------------------------------------------------------------------- #
# Sample data (synthetic, same columns as the shipped model)
# --------------------------------------------------------------------------- #
SAMPLE_FEATURES = ["Pressure", "global_radiation", "temp_mean(c)", "temp_min(c)",
                   "Wind_Speed", "Wind_Bearing"]
SAMPLE_TARGET = "temp_max(c)"


def sample_weather_df(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic daily weather with seasonality, noise and a few missing readings."""
    rng = np.random.default_rng(seed)
    season = np.sin(2 * np.pi * (rng.uniform(0, 365, n) - 110) / 365)
    mean = 11 + 8.5 * season + rng.normal(0, 3.0, n)
    rad = np.clip(1.45 + 1.15 * season + rng.normal(0, 0.45, n), 0.03, 3.4)
    tmin = mean - rng.uniform(2, 8, n)
    tmax = mean + rng.uniform(2.5, 7, n) + 0.5 * (rad - 1.45) + rng.normal(0, 0.8, n)
    df = pd.DataFrame({
        "Pressure": np.clip(rng.normal(1010, 9, n), 985, 1035).round(1),
        "global_radiation": rad.round(2),
        "temp_mean(c)": mean.round(1),
        "temp_min(c)": tmin.round(1),
        "Wind_Speed": np.clip(rng.gamma(2.2, 2.4, n), 0.1, 40).round(1),
        "Wind_Bearing": rng.uniform(0, 360, n).round(0),
        SAMPLE_TARGET: tmax.round(1),
    })
    for col in ("global_radiation", "Wind_Speed"):  # ~1% gaps so cleaning tools have a job
        df.loc[rng.random(n) < 0.01, col] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Task detection, dataset preparation
# --------------------------------------------------------------------------- #
def detect_task(y: pd.Series) -> str:
    y = y.dropna()
    if y.empty or not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    int_like = bool(np.allclose(y % 1, 0))
    return "classification" if (int_like and y.nunique() <= 10) else "regression"


def _stats(s: pd.Series) -> dict:
    q = s.quantile([0.1, 0.5, 0.9])
    return dict(min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
                median=float(q[0.5]), p10=float(q[0.1]), p50=float(q[0.5]), p90=float(q[0.9]))


def encode_frame(b: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Raw feature rows -> the exact model matrix (one-hot, column order) for bundle b."""
    X = raw[b["raw_features"]].copy()
    cats = list(b["cat_info"])
    for c in cats:
        X[c] = pd.Categorical(X[c].astype(str), categories=b["cat_info"][c])
    X = pd.get_dummies(X, columns=cats, dtype=float) if cats else X.astype(float)
    return X.reindex(columns=b["features"], fill_value=0.0)


def prepare_dataset(df: pd.DataFrame, target: str, features: list[str], task: str,
                    test_size: float = 0.2, seed: int = SEED) -> dict:
    if not features:
        raise ValueError("Select at least one feature column.")
    cols = [target] + [f for f in features if f != target]
    d = df[cols].dropna().reset_index(drop=True)
    if len(d) < 30:
        raise ValueError(f"Only {len(d)} complete rows are left; at least 30 are needed. "
                         "Fill or drop missing values first.")
    feats = cols[1:]
    Xr = d[feats].copy()
    for c in Xr.columns:
        if pd.api.types.is_bool_dtype(Xr[c]):
            Xr[c] = Xr[c].astype(int)
    cat_cols = [c for c in Xr.columns if not pd.api.types.is_numeric_dtype(Xr[c])]
    for c in cat_cols:
        Xr[c] = Xr[c].astype(str)
    num_info = {c: _stats(Xr[c]) for c in Xr.columns if c not in cat_cols}
    cat_info = {c: sorted(Xr[c].unique().tolist()) for c in cat_cols}
    X = pd.get_dummies(Xr, columns=cat_cols, dtype=float) if cat_cols else Xr.astype(float)

    classes = None
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(d[target]), name=target)
        classes = [str(c) for c in le.classes_]
        if len(classes) < 2:
            raise ValueError("The target column has only one class; there is nothing to learn.")
    else:
        if not pd.api.types.is_numeric_dtype(d[target]):
            raise ValueError("Regression needs a numeric target column.")
        y = d[target].astype(float)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    try:
        parts = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        parts = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_test, y_train, y_test = parts
    return dict(df=d, X_all=X, y_all=y, X_train=X_train, X_test=X_test, y_train=y_train,
                y_test=y_test, features=X.columns.tolist(), raw_features=feats,
                num_info=num_info, cat_info=cat_info, classes=classes,
                target=target, dropped=len(df) - len(d))


def _score(task: str, y, pred) -> float:
    return float(accuracy_score(y, pred) if task == "classification" else r2_score(y, pred))


def fit_bundle(prep: dict, task: str, params: dict, source: str) -> dict:
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    model = cls(random_state=SEED, **params).fit(prep["X_train"], prep["y_train"])
    y_pred = model.predict(prep["X_test"])
    return dict(
        uid=uuid.uuid4().hex[:8], model=model, features=prep["features"],
        raw_features=prep["raw_features"], num_info=prep["num_info"], cat_info=prep["cat_info"],
        target=prep["target"], task=task, classes=prep["classes"], unit="",
        df=prep["df"], X_all=prep["X_all"], y_all=prep["y_all"],
        X_train=prep["X_train"], y_train=prep["y_train"],
        X_eval=prep["X_test"], y_eval=prep["y_test"], y_pred=y_pred,
        train_score=_score(task, prep["y_train"], model.predict(prep["X_train"])),
        test_score=_score(task, prep["y_test"], y_pred),
        params=params, source=source, created=dt.datetime.now(), n_rows=len(prep["df"]),
        dropped=prep["dropped"], notes=[],
    )


# --------------------------------------------------------------------------- #
# Pre-trained models
# --------------------------------------------------------------------------- #
def tree_feature_stats(model, features: list[str]) -> dict:
    """Approximate each feature's range/typical values from the tree's split points."""
    t = model.tree_
    out = {}
    for i, f in enumerate(features):
        th = t.threshold[t.feature == i]
        if th.size == 0:
            out[f] = dict(min=0.0, max=1.0, mean=0.5, median=0.5, p10=0.1, p50=0.5, p90=0.9)
            continue
        q = np.percentile(th, [10, 50, 90])
        out[f] = dict(min=float(th.min()), max=float(th.max()), mean=float(th.mean()),
                      median=float(q[1]), p10=float(q[0]), p50=float(q[1]), p90=float(q[2]))
    return out


def bundle_from_model(model, source: str, target: str = "Prediction", unit: str = "") -> dict:
    if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        raise TypeError(f"Expected a scikit-learn decision tree, got {type(model).__name__}.")
    feats = list(getattr(model, "feature_names_in_",
                         [f"feature_{i}" for i in range(model.n_features_in_)]))
    task = "classification" if is_classifier(model) else "regression"
    return dict(
        uid=uuid.uuid4().hex[:8], model=model, features=feats, raw_features=feats,
        num_info=tree_feature_stats(model, feats), cat_info={}, target=target, task=task,
        classes=[str(c) for c in model.classes_] if task == "classification" else None,
        unit=unit, df=None, X_all=None, y_all=None, X_train=None, y_train=None,
        X_eval=None, y_eval=None, y_pred=None, train_score=None, test_score=None,
        params=model.get_params(), source=source, created=dt.datetime.now(),
        n_rows=int(model.tree_.n_node_samples[0]), dropped=0, notes=[],
    )


def load_model_file(file, source: str, **kw) -> dict:
    """Load a pickled decision tree (path or file-like) into a bundle. Trusted files only."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(file)
    b = bundle_from_model(model, source, **kw)
    if any("InconsistentVersion" in type(w.message).__name__ for w in caught):
        b["notes"].append(f"Saved with a different scikit-learn version than the {sklearn.__version__} "
                          "running here. It works, but pin the version in requirements.txt.")
    return b


def demo_bundle() -> dict:
    """Fallback model trained on synthetic data when no model.pkl is present."""
    prep = prepare_dataset(sample_weather_df(), SAMPLE_TARGET, SAMPLE_FEATURES, "regression")
    b = fit_bundle(prep, "regression", dict(max_depth=8, min_samples_leaf=5),
                   "Demo model on synthetic data")
    b["unit"] = "°C"
    b["notes"].append("No model.pkl found next to app.py, so a demo model was trained on synthetic data.")
    return b


def attach_evaluation(b: dict, df: pd.DataFrame, target_col: str) -> dict:
    """Score a bundle on a user-supplied CSV that has its feature columns plus the target."""
    missing = [f for f in b["raw_features"] if f not in df.columns]
    if missing:
        raise ValueError("The file is missing model feature columns: " + ", ".join(missing))
    d = df[b["raw_features"] + [target_col]].dropna()
    if len(d) < 5:
        raise ValueError("Fewer than 5 complete rows to evaluate on.")
    X = encode_frame(b, d)
    y = d[target_col].to_numpy()
    if b["task"] == "regression":
        y = y.astype(float)
    b["X_eval"], b["y_eval"], b["y_pred"] = X, y, b["model"].predict(X)
    b["test_score"] = _score(b["task"], y, b["y_pred"])
    b["eval_rows"] = len(d)
    return b


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(b: dict) -> dict:
    y, p = np.asarray(b["y_eval"]), np.asarray(b["y_pred"])
    m = b["model"]
    if b["task"] == "classification":
        labels = list(m.classes_)
        names = b["classes"]
        out = dict(
            task="classification", n=len(y),
            accuracy=float(accuracy_score(y, p)),
            precision=float(precision_score(y, p, average="weighted", zero_division=0)),
            recall=float(recall_score(y, p, average="weighted", zero_division=0)),
            f1=float(f1_score(y, p, average="weighted", zero_division=0)),
            cm=confusion_matrix(y, p, labels=labels), names=names,
            report=pd.DataFrame(classification_report(y, p, labels=labels, target_names=names,
                                                      output_dict=True, zero_division=0)).T,
        )
        if len(labels) == 2 and hasattr(m, "predict_proba"):
            fpr, tpr, _ = roc_curve(y == labels[1], m.predict_proba(b["X_eval"])[:, 1])
            out.update(roc=(fpr, tpr), auc=float(auc(fpr, tpr)))
        return out
    y = y.astype(float)
    mse = float(mean_squared_error(y, p))
    return dict(task="regression", n=len(y), r2=float(r2_score(y, p)), mse=mse,
                rmse=float(np.sqrt(mse)), mae=float(mean_absolute_error(y, p)),
                medae=float(median_absolute_error(y, p)), y=y, p=p, resid=y - p)


def depth_sweep(prep: dict, task: str, params: dict, max_depth: int = 20,
                max_rows: int = 40000) -> pd.DataFrame:
    """Train/test score for each depth, to spot where the tree starts to overfit."""
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    base = {k: v for k, v in params.items() if k != "max_depth"}
    Xtr, ytr = prep["X_train"], prep["y_train"]
    if len(Xtr) > max_rows:
        idx = np.random.default_rng(SEED).choice(len(Xtr), max_rows, replace=False)
        Xtr, ytr = Xtr.iloc[idx], ytr.iloc[idx]
    rows = []
    for d in range(1, max_depth + 1):
        m = cls(random_state=SEED, max_depth=d, **base).fit(Xtr, ytr)
        rows.append((d, _score(task, ytr, m.predict(Xtr)),
                     _score(task, prep["y_test"], m.predict(prep["X_test"]))))
        if m.get_depth() < d:  # fully grown, deeper trees are identical
            break
    return pd.DataFrame(rows, columns=["Depth", "Train", "Test"])


def recommend_depth(sweep: pd.DataFrame, tol: float = 0.005) -> int:
    """Smallest depth whose test score is within `tol` of the best."""
    best = sweep["Test"].max()
    return int(sweep.loc[sweep["Test"] >= best - tol, "Depth"].min())


def cv_report(b: dict, cv: int = 5, max_rows: int = 30000) -> dict:
    X, y = b["X_all"], b["y_all"]
    if len(X) > max_rows:
        idx = np.random.default_rng(SEED).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    scoring = "accuracy" if b["task"] == "classification" else "r2"
    scores = cross_val_score(clone(b["model"]), X, y, cv=cv, scoring=scoring)
    sizes, tr, te = learning_curve(clone(b["model"]), X, y, cv=cv, scoring=scoring,
                                   train_sizes=np.linspace(0.15, 1.0, 6))
    return dict(scores=scores, scoring=scoring, sizes=sizes,
                train=tr.mean(axis=1), val=te.mean(axis=1), rows=len(X))


# --------------------------------------------------------------------------- #
# Tree inspection
# --------------------------------------------------------------------------- #
def tree_profile(model) -> dict:
    t = model.tree_
    leaf = t.children_left == -1
    n = t.n_node_samples[leaf]
    pure = float((t.impurity[leaf] <= 1e-12).mean())
    median = float(np.median(n))
    depth, n_leaves = int(model.get_depth()), int(leaf.sum())
    if depth > 15 and median <= 3:
        level, msg = "High", ("Most leaves hold only a handful of training rows, so the tree is "
                              "memorising rather than generalising. Limit depth or require more "
                              "samples per leaf when retraining.")
    elif depth > 10 and median <= 10:
        level, msg = "Moderate", "The tree is deep with small leaves; check it on held-out data."
    else:
        level, msg = "Low", "Depth and leaf sizes look reasonable."
    return dict(depth=depth, leaves=n_leaves, root_n=int(t.n_node_samples[0]),
                pure_share=pure, median_leaf=median, leaf_sizes=n, risk=level, risk_msg=msg)


def leaf_info(model, X: pd.DataFrame) -> dict:
    t = model.tree_
    leaf = int(model.apply(X)[0])
    return dict(leaf=leaf, n=int(t.n_node_samples[leaf]), impurity=float(t.impurity[leaf]))


def decision_path_rows(model, X: pd.DataFrame, features: list[str]) -> list[dict]:
    """Human-readable rules the input passed through, root to leaf."""
    t = model.tree_
    nodes = model.decision_path(X).indices
    row = X.iloc[0]
    out = []
    for step, node in enumerate(nodes[:-1], start=1):
        f = features[t.feature[node]]
        th = float(t.threshold[node])
        val = float(np.float32(row[f]))
        left = val <= th
        out.append({"Step": step, "Question": f"{f} ≤ {th:.4g}", "Input": round(float(row[f]), 4),
                    "Answer": "Yes, go left" if left else "No, go right",
                    "Samples": int(t.n_node_samples[node])})
    return out


def what_if_curve(b: dict, raw: dict, feature: str, points: int = 80):
    s = b["num_info"][feature]
    xs = np.linspace(s["min"], s["max"], points)
    frame = pd.DataFrame([raw] * points)
    frame[feature] = xs
    return xs, b["model"].predict(encode_frame(b, frame))


def out_of_range(b: dict, raw: dict) -> list[str]:
    return [label_for(f) for f, v in raw.items()
            if f in b["num_info"] and not (b["num_info"][f]["min"] <= v <= b["num_info"][f]["max"])]


# --------------------------------------------------------------------------- #
# Tree figure (interactive, palette-consistent)
# --------------------------------------------------------------------------- #
_VALUE_STOPS = ["#BFD7FB", "#DDEBFB", "#FDF1C9", "#FBC98A", "#F7A19B"]
_CLASS_FILLS = ["#C9D8FB", "#BDEBE4", "#FBE3B0", "#F7C9D2", "#DAD3F7", "#C5EBD6"]


def _short(text: str, n: int = 18) -> str:
    """Trim long names so node boxes keep a predictable width."""
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


NODE_PX = 150  # horizontal room reserved for every leaf column in the tree diagram


def tree_figure(model, features: list[str], classes: list[str] | None, max_depth: int = 3) -> go.Figure:
    t = model.tree_
    regression = classes is None
    pos: dict[int, tuple[float, int]] = {}
    counter = [0]

    def walk(node: int, depth: int) -> None:
        left, right = t.children_left[node], t.children_right[node]
        if left == -1 or depth == max_depth:
            pos[node] = (counter[0], -depth)
            counter[0] += 1
            return
        walk(left, depth + 1)
        walk(right, depth + 1)
        pos[node] = ((pos[left][0] + pos[right][0]) / 2, -depth)

    walk(0, 0)
    vals = t.value[:, 0, 0] if regression else None
    lo, hi = (float(vals.min()), float(vals.max())) if regression else (0.0, 1.0)

    fig = go.Figure()
    ex, ey = [], []
    for node, (x, y) in pos.items():
        for child in (t.children_left[node], t.children_right[node]):
            if child != -1 and child in pos:
                ex += [x, pos[child][0], None]
                ey += [y, pos[child][1], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip",
                             line=dict(color="#A9B7CC", width=1.4), showlegend=False))

    hx, hy, ht = [], [], []
    for node, (x, y) in pos.items():
        n = int(t.n_node_samples[node])
        is_leaf = t.children_left[node] == -1
        cut = (not is_leaf) and y == -max_depth
        if regression:
            v = float(vals[node])
            fill = T.lerp_color(_VALUE_STOPS, (v - lo) / (hi - lo + 1e-9))
            main = f"{v:.1f}"
            tip = f"mean {v:.2f} · {n:,} samples · variance {t.impurity[node]:.2f}"
        else:
            probs = t.value[node, 0] / t.value[node, 0].sum()
            k = int(np.argmax(probs))
            fill = _CLASS_FILLS[k % len(_CLASS_FILLS)]
            main = f"{_short(classes[k], 16)} {probs[k]:.0%}"
            tip = f"{classes[k]} · {probs[k]:.1%} · {n:,} samples · impurity {t.impurity[node]:.3f}"
        if is_leaf or cut:
            text = f"<b>{main}</b><br>n = {n:,}" + ("<br>…" if cut else "")
        else:
            f = _short(features[t.feature[node]])
            text = f"{f}<br>≤ {t.threshold[node]:.4g}<br>n = {n:,}<br><b>{main}</b>"
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, align="center",
                           bgcolor=fill, bordercolor="#8FA2BF", borderwidth=1, borderpad=6,
                           font=dict(size=11, color=T.INK), yanchor="middle")
        hx.append(x); hy.append(y); ht.append(tip)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", hovertext=ht, hoverinfo="text",
                             marker=dict(size=48, color="rgba(0,0,0,0)"), showlegend=False))
    leaves = max(counter[0], 1)
    span = leaves + 0.4  # x-axis range below is [-0.7, leaves - 0.3]
    fig.update_layout(
        width=int(span * NODE_PX + 8), height=150 * (max_depth + 1) + 40,
        margin=dict(l=4, r=4, t=10, b=4),
        xaxis=dict(visible=False, range=[-0.7, leaves - 0.3]),
        yaxis=dict(visible=False, range=[-max_depth - 0.6, 0.6]),
    )
    return fig


# --------------------------------------------------------------------------- #
# Weather bands and guidance
# --------------------------------------------------------------------------- #
DEFAULT_EDGES = dict(freeze=0.0, cool=12.0, warm=24.0, hot=30.0, extreme=35.0)
BAND_LABEL = dict(freezing="Freezing", cool="Cool", mild="Mild", warm="Warm",
                  hot="Hot", extreme="Extreme heat")
BAND_STATE = dict(freezing="cool", cool="cool", mild="mild", warm="warm", hot="hot", extreme="extreme")

_ADVISORY = {
    "freezing": [("Public health", "Cold exposure risk", "alert"), ("Outdoor events", "Reconsider", "caution"),
                 ("Energy demand", "High heating load", "caution"), ("Agriculture", "Frost risk", "alert")],
    "cool": [("Public health", "Normal", "ok"), ("Outdoor events", "Go ahead, bring layers", "ok"),
             ("Energy demand", "Moderate heating", "info"), ("Agriculture", "Low risk", "ok")],
    "mild": [("Public health", "Normal", "ok"), ("Outdoor events", "Ideal conditions", "ok"),
             ("Energy demand", "Low", "ok"), ("Agriculture", "Favourable", "ok")],
    "warm": [("Public health", "Stay hydrated", "info"), ("Outdoor events", "Go ahead, add shade", "ok"),
             ("Energy demand", "Rising cooling load", "info"), ("Agriculture", "Watch soil moisture", "info")],
    "hot": [("Public health", "Heat advisory", "caution"), ("Outdoor events", "Avoid midday", "caution"),
            ("Energy demand", "High cooling load", "caution"), ("Agriculture", "Irrigate", "caution")],
    "extreme": [("Public health", "Heat alert", "alert"), ("Outdoor events", "Postpone or cancel", "alert"),
                ("Energy demand", "Peak cooling load", "alert"), ("Agriculture", "Heat stress risk", "alert")],
}


def band_for(temp: float, e: dict) -> str:
    if temp < e["freeze"]:
        return "freezing"
    if temp < e["cool"]:
        return "cool"
    if temp < e["warm"]:
        return "mild"
    if temp < e["hot"]:
        return "warm"
    return "hot" if temp < e["extreme"] else "extreme"


def advisory(band: str) -> list[tuple[str, str, str]]:
    return _ADVISORY[band]


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def model_bytes(model) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.getvalue()


def model_card(b: dict, metrics: dict | None) -> dict:
    card = dict(
        source=b["source"], created=b["created"].isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__, task=b["task"], target=b["target"],
        features=b["raw_features"], classes=b["classes"],
        feature_ranges={f: {k: round(v[k], 4) for k in ("min", "max", "median")}
                        for f, v in b["num_info"].items()},
        categories=b["cat_info"], hyperparameters={k: (None if v is None else v)
                                                   for k, v in b["params"].items()
                                                   if isinstance(v, (int, float, str, type(None)))},
        tree=dict(depth=int(b["model"].get_depth()), leaves=int(b["model"].get_n_leaves()),
                  training_rows=b["n_rows"]),
    )
    if metrics:
        card["metrics"] = {k: round(v, 5) for k, v in metrics.items()
                           if k in ("r2", "rmse", "mae", "medae", "accuracy", "precision", "recall", "f1", "auc")}
    return card


def build_report(b: dict, metrics: dict | None, profile: dict) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# ModelForge Studio · Model report", f"Generated {now}", "",
        "## Model",
        f"- Source: {b['source']}",
        f"- Task: {b['task']}  |  Target: `{b['target']}`",
        f"- Features ({len(b['raw_features'])}): " + ", ".join(f"`{f}`" for f in b["raw_features"]),
        f"- Training rows: {b['n_rows']:,}",
        f"- Tree depth: {profile['depth']}  |  Leaves: {profile['leaves']:,}  |  "
        f"Median leaf size: {profile['median_leaf']:.0f}",
        f"- Overfitting risk: **{profile['risk']}**. {profile['risk_msg']}", "",
        "## Evaluation",
    ]
    if metrics is None:
        lines.append("No evaluation data has been supplied for this model yet.")
    elif metrics["task"] == "regression":
        lines += [f"- Evaluation rows: {metrics['n']:,}", f"- R²: {metrics['r2']:.4f}",
                  f"- RMSE: {metrics['rmse']:.4f}", f"- MAE: {metrics['mae']:.4f}",
                  f"- Median absolute error: {metrics['medae']:.4f}"]
    else:
        lines += [f"- Evaluation rows: {metrics['n']:,}", f"- Accuracy: {metrics['accuracy']:.4f}",
                  f"- Precision (weighted): {metrics['precision']:.4f}",
                  f"- Recall (weighted): {metrics['recall']:.4f}", f"- F1 (weighted): {metrics['f1']:.4f}"]
        if "auc" in metrics:
            lines.append(f"- ROC AUC: {metrics['auc']:.4f}")
    imp = pd.Series(b["model"].feature_importances_, index=b["features"]).sort_values(ascending=False)
    lines += ["", "## Feature importance"] + [f"- `{k}`: {v:.1%}" for k, v in imp.head(10).items()]
    if b.get("params"):
        lines += ["", "## Hyperparameters"] + [
            f"- {k}: {v}" for k, v in b["params"].items() if isinstance(v, (int, float, str, type(None)))]
    return "\n".join(lines) + "\n"


def card_json(card: dict) -> str:
    return json.dumps(card, indent=2, default=str)
