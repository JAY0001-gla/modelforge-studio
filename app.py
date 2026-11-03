"""ModelForge Studio: train, explain and deploy decision-tree models (includes a pre-trained weather model).

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.tree import export_text

import core as C
import theme as T

st.set_page_config(page_title="ModelForge Studio", page_icon="🌤️",
                   layout="wide", initial_sidebar_state="expanded")
T.inject_css()
T.register_plotly()
ss = st.session_state

MODEL_PATHS = [Path(__file__).parent / "model.pkl", Path(__file__).parent / "models" / "model.pkl"]
SAMPLE_NAME = "Sample weather data (synthetic)"


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def chart(fig: go.Figure, height: int | None = None, scroll: bool = False) -> None:
    """Draw a Plotly figure. scroll=True keeps the figure's own width (it scrolls sideways)."""
    if height:
        fig.update_layout(height=height)
    width_kw = T.content() if scroll else T.stretch(st.plotly_chart)
    st.plotly_chart(fig, theme=None, **width_kw)


def table(df: pd.DataFrame, **kw) -> None:
    st.dataframe(df, hide_index=kw.pop("hide_index", True), **T.stretch(st.dataframe), **kw)


def goto(page: str) -> None:
    ss["_goto"] = page


def flash() -> None:
    msg = ss.pop("_flash", None)
    if msg:
        getattr(st, msg[0])(msg[1])


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the pre-trained model…")
def load_default_bundle() -> dict:
    for path in MODEL_PATHS:
        if path.exists():
            try:
                # Assumption: the shipped model predicts a temperature in °C (range -8.5 to 37).
                return C.load_model_file(path, f"Pre-trained model ({path.name})",
                                         target="Temperature", unit="°C")
            except Exception as exc:  # corrupt/incompatible file: fall back, but say why
                fallback = C.demo_bundle()
                fallback["notes"].append(f"Could not load {path.name}: {exc}")
                return fallback
    return C.demo_bundle()


if "default_bundle" not in ss:
    ss.default_bundle = dict(load_default_bundle())  # shallow copy: per-session metadata
ss.setdefault("custom_bundle", None)
ss.setdefault("ctx_label", "Pre-trained model")
if "_ctx_next" in ss:
    ss.ctx_label = ss.pop("_ctx_next")
if "_goto" in ss:
    ss.nav = ss.pop("_goto")

PAGES = ["Weather forecast", "Train studio", "Model analytics", "Tree explorer",
         "Prediction center", "Model hub", "Data explorer", "Report"]


def active_bundle() -> dict:
    if ss.ctx_label == "My trained model" and ss.custom_bundle is not None:
        return ss.custom_bundle
    return ss.default_bundle


def ensure_work_df() -> None:
    if ss.get("work_df") is None:
        ss.work_df, ss.work_name = C.sample_weather_df(), SAMPLE_NAME


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    T.render('<div class="brand"><b>ModelForge Studio</b><span>Machine learning workbench</span></div>')
    st.radio("Active model", ["Pre-trained model", "My trained model"], key="ctx_label")
    if ss.ctx_label == "My trained model" and ss.custom_bundle is None:
        st.caption("No model of your own yet. Train one in Train studio or import one in Model hub.")
    page = st.radio("Go to", PAGES, key="nav")
    ab = active_bundle()
    prof_side = C.tree_profile(ab["model"])
    T.render(
        f'<div class="side-note"><b>{T.esc(ab["source"])}</b><br>'
        f'{T.esc(ab["task"].capitalize())} tree · depth {prof_side["depth"]} · '
        f'{prof_side["leaves"]:,} leaves<br>'
        f'{len(ab["raw_features"])} features · target {T.esc(ab["target"])}</div>'
    )


# --------------------------------------------------------------------------- #
# Shared input form
# --------------------------------------------------------------------------- #
def feature_inputs(b: dict, prefix: str, cols: int = 3) -> dict:
    """One input per raw feature; values live in session state under `prefix + name`."""
    values, grid = {}, st.columns(cols)
    for i, f in enumerate(b["raw_features"]):
        key = f"{prefix}{f}"
        with grid[i % cols]:
            if f in b["cat_info"]:
                opts = b["cat_info"][f]
                if ss.get(key) not in opts:
                    ss[key] = opts[0]
                values[f] = st.selectbox(C.label_for(f), opts, key=key)
            else:
                s = b["num_info"][f]
                if key not in ss:
                    ss[key] = float(round(s["median"], 4))
                values[f] = st.number_input(C.label_for(f), key=key, format="%.2f",
                                            step=C.smart_step(s["min"], s["max"]))
    return values


def range_note(b: dict, raw: dict) -> None:
    bad = C.out_of_range(b, raw)
    if bad:
        st.warning("Outside the range this model has seen: " + ", ".join(bad) +
                   ". A tree treats such values like the nearest edge of its data, so the result "
                   "is less trustworthy.")


# --------------------------------------------------------------------------- #
# Page: Weather forecast
# --------------------------------------------------------------------------- #
def run_forecast() -> None:
    b = ss.default_bundle
    raw = {f: ss[f"fc_{b['uid']}_{f}"] for f in b["raw_features"]}
    X = C.encode_frame(b, pd.DataFrame([raw]))
    ss.forecast = dict(uid=b["uid"], raw=raw, pred=float(b["model"].predict(X)[0]),
                       **C.leaf_info(b["model"], X))


def apply_preset(quantile: str) -> None:
    b = ss.default_bundle
    for f in b["raw_features"]:
        ss[f"fc_{b['uid']}_{f}"] = float(round(b["num_info"][f][quantile], 4))


def edges() -> dict:
    return {k: ss.get(f"edge_{k}", v) for k, v in C.DEFAULT_EDGES.items()}


def page_forecast() -> None:
    T.page_header("Weather forecast",
                  "Estimate the day's temperature from six atmospheric readings, then turn it "
                  "into guidance for health, events, energy and farming.")
    b = ss.default_bundle
    for n in b["notes"]:
        st.info(n)
    if b["task"] != "regression":
        st.error("The pre-trained model is a classifier. The forecast page expects a temperature "
                 "regressor; use Prediction center for other models.")
        return

    res = ss.get("forecast")
    res = res if res and res["uid"] == b["uid"] else None
    e = edges()
    if res is None:
        T.hero(state="idle", head="Forecast", title="Set the conditions below and run a forecast.",
               sub="The glow shifts colour to match the predicted weather.")
    else:
        band = C.band_for(res["pred"], e)
        lo, hi = -10.0, 40.0
        T.hero(state=C.BAND_STATE[band], head="Predicted temperature", value=f"{res['pred']:.1f}",
               unit=b["unit"], sub=f"{C.BAND_LABEL[band]} conditions, backed by {res['n']:,} "
               f"training record{'s' if res['n'] != 1 else ''} in the matching leaf.",
               meter=((res["pred"] - lo) / (hi - lo) * 100, ["-10°", "0°", "10°", "20°", "30°", "40°"]))

    st.subheader("Conditions")
    c1, c2, c3, _ = st.columns([1, 1, 1, 2])
    c1.button("Cool day", on_click=apply_preset, args=("p10",), **T.stretch(st.button))
    c2.button("Typical day", on_click=apply_preset, args=("p50",), **T.stretch(st.button))
    c3.button("Warm day", on_click=apply_preset, args=("p90",), **T.stretch(st.button))
    raw = feature_inputs(b, f"fc_{b['uid']}_")
    range_note(b, raw)
    st.button("Run forecast", type="primary", on_click=run_forecast, **T.stretch(st.button))

    if res is None:
        return
    st.markdown("&nbsp;")
    tab_a, tab_b, tab_c = st.tabs(["Guidance", "Decision path", "What-if"])
    with tab_a:
        T.rows(C.advisory(band))
        if res["n"] < 5:
            st.caption("This leaf rests on very few records, so treat the value as indicative.")
        with st.expander("Adjust temperature bands"):
            cols = st.columns(5)
            for col, (k, label) in zip(cols, [("freeze", "Freezing below"), ("cool", "Mild from"),
                                              ("warm", "Warm from"), ("hot", "Hot from"),
                                              ("extreme", "Extreme from")]):
                col.number_input(f"{label} (°C)", value=C.DEFAULT_EDGES[k], step=1.0, key=f"edge_{k}")
    with tab_b:
        X = C.encode_frame(b, pd.DataFrame([res["raw"]]))
        rows = C.decision_path_rows(b["model"], X, b["features"])
        st.caption(f"{len(rows)} questions led to leaf #{res['leaf']}. Each answer narrows the "
                   "set of historical records until the leaf's average is returned.")
        table(pd.DataFrame(rows), height=min(420, 38 * (len(rows) + 1)))
    with tab_c:
        feat = st.selectbox("Vary", [f for f in b["raw_features"] if f in b["num_info"]],
                            format_func=C.label_for, key="whatif_feature")
        xs, ys = C.what_if_curve(b, res["raw"], feat)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=T.COBALT, width=2.5, shape="hv"),
                                   name="Prediction", hovertemplate="%{x:.2f} → %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[res["raw"][feat]], y=[res["pred"]], mode="markers", name="Your input",
                                 marker=dict(size=12, color=T.SUN, line=dict(color="#fff", width=2))))
        fig.update_layout(xaxis_title=C.label_for(feat), yaxis_title=f"Predicted {b['unit']}")
        chart(fig, 360)
        st.caption("All other inputs stay fixed. Steps are normal for a tree: it predicts the "
                   "average of each leaf, not a smooth curve.")


# --------------------------------------------------------------------------- #
# Page: Train studio
# --------------------------------------------------------------------------- #
def _clean(kind: str) -> None:
    df = ss.work_df
    if kind == "dropna":
        ss.work_df = df.dropna().reset_index(drop=True)
    elif kind == "fill":
        num = df.select_dtypes("number").columns
        ss.work_df = df.fillna(df[num].median())
    else:
        ss.work_df = df.drop_duplicates().reset_index(drop=True)
    st.toast("Dataset updated.")


def _use_depth(d: int) -> None:
    ss.cfg_unlimited, ss.cfg_depth = False, int(d)


def page_train() -> None:
    T.page_header("Train studio", "Bring a dataset, tune a decision tree and check that it "
                  "generalises before you rely on it.")
    flash()
    st.subheader("Data")
    src = st.radio("Dataset", [SAMPLE_NAME, "Upload CSV"], horizontal=True,
                   label_visibility="collapsed", key="data_src")
    if src == SAMPLE_NAME:
        if ss.get("work_name") != SAMPLE_NAME:
            ss.work_df, ss.work_name = C.sample_weather_df(), SAMPLE_NAME
    else:
        up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
        if up is not None and ss.get("work_sig") != (up.name, up.size):
            try:
                ss.work_df, ss.work_name, ss.work_sig = pd.read_csv(up), up.name, (up.name, up.size)
            except Exception as exc:
                st.error(f"Could not read that file as CSV: {exc}")
        if ss.get("work_name") in (None, SAMPLE_NAME):
            st.info("Drop a CSV above to continue, or switch to the sample data.")
            return
    df: pd.DataFrame = ss.work_df

    T.strip([("Rows", f"{len(df):,}", ss.work_name), ("Columns", f"{df.shape[1]}", ""),
             ("Missing values", f"{int(df.isna().sum().sum()):,}", ""),
             ("Duplicate rows", f"{int(df.duplicated().sum()):,}", "")])
    table(df.head(8))
    c1, c2, c3 = st.columns(3)
    c1.button("Drop rows with missing values", on_click=_clean, args=("dropna",), **T.stretch(st.button))
    c2.button("Fill numeric gaps with the median", on_click=_clean, args=("fill",), **T.stretch(st.button))
    c3.button("Remove duplicate rows", on_click=_clean, args=("dups",), **T.stretch(st.button))

    st.subheader("Model setup")
    cols = df.columns.tolist()
    dkey = ss.work_name
    default_t = C.SAMPLE_TARGET if C.SAMPLE_TARGET in cols else cols[-1]
    target = st.selectbox("Target to predict", cols, index=cols.index(default_t), key=f"cfg_target_{dkey}")
    detected = C.detect_task(df[target])
    choice = st.radio("Task", ["Auto-detect", "Regression", "Classification"], horizontal=True, key="cfg_task")
    task = detected if choice == "Auto-detect" else choice.lower()
    st.caption(f"Detected: {detected}. Training as {task}.")

    cands = [c for c in cols if c != target]
    usable = [c for c in cands if pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 50]
    skipped = [c for c in cands if c not in usable]
    if skipped:
        st.caption("Left out by default (text with more than 50 distinct values): " + ", ".join(skipped))
    features = st.multiselect("Input features", cands, default=usable, key=f"cfg_feat_{dkey}")

    st.markdown("**Tree settings**")
    ss.setdefault("cfg_depth", 6)
    ss.setdefault("cfg_unlimited", False)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.toggle("Unlimited depth", key="cfg_unlimited")
        depth = st.slider("Max depth", 1, 30, key="cfg_depth", disabled=ss.cfg_unlimited)
        test_pct = st.slider("Held out for testing (%)", 10, 40, 20)
    with h2:
        leaf = st.slider("Min samples per leaf", 1, 100, 5)
        split = st.slider("Min samples to split", 2, 200, 10)
    with h3:
        crit_opts = ["gini", "entropy", "log_loss"] if task == "classification" else ["squared_error", "friedman_mse"]
        criterion = st.selectbox("Split criterion", crit_opts)
        alpha = st.number_input("Pruning strength (ccp_alpha)", 0.0, 1.0, 0.0, step=0.0005, format="%.4f")
    params = dict(max_depth=None if ss.cfg_unlimited else int(depth), min_samples_leaf=int(leaf),
                  min_samples_split=int(split), criterion=criterion, ccp_alpha=float(alpha))

    st.markdown("&nbsp;")
    a, b_ = st.columns([1, 1])
    if a.button("Suggest a depth", **T.stretch(st.button)):
        try:
            with st.spinner("Training trees from depth 1 upward…"):
                prep = C.prepare_dataset(df, target, features, task, test_pct / 100)
                sweep = C.depth_sweep(prep, task, params)
            ss.sweep = dict(df=sweep, rec=C.recommend_depth(sweep), key=(dkey, target, tuple(features), task))
        except ValueError as exc:
            st.error(str(exc))
    sw = ss.get("sweep")
    if sw and sw["key"] == (dkey, target, tuple(features), task):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sw["df"]["Depth"], y=sw["df"]["Train"], name="Training score",
                                 mode="lines+markers", line=dict(color=T.TEAL, width=2.5)))
        fig.add_trace(go.Scatter(x=sw["df"]["Depth"], y=sw["df"]["Test"], name="Test score",
                                 mode="lines+markers", line=dict(color=T.COBALT, width=2.5)))
        fig.add_vline(x=sw["rec"], line=dict(color=T.SUN, dash="dot", width=2))
        fig.update_layout(xaxis_title="Tree depth", yaxis_title="R²" if task == "regression" else "Accuracy")
        chart(fig, 320)
        s1, s2 = st.columns([3, 1], vertical_alignment="center")
        s1.caption(f"Test score stops improving around depth {sw['rec']}. Past that the tree fits the "
                   "training data better but not new data.")
        s2.button(f"Use depth {sw['rec']}", on_click=_use_depth, args=(sw["rec"],), **T.stretch(st.button))

    if st.button("Train model", type="primary", **T.stretch(st.button)):
        try:
            with st.spinner("Training…"):
                prep = C.prepare_dataset(df, target, features, task, test_pct / 100)
                bundle = C.fit_bundle(prep, task, params, f"Trained on {ss.work_name}")
        except ValueError as exc:
            st.error(str(exc))
        else:
            ss.custom_bundle, ss._ctx_next = bundle, "My trained model"
            for k in ("cv", "report_metrics"):
                ss.pop(k, None)
            note = f" {prep['dropped']:,} rows with gaps were skipped." if prep["dropped"] else ""
            ss._flash = ("success", f"Model trained and activated.{note}")
            st.rerun()

    cb = ss.custom_bundle
    if cb is not None:
        st.subheader("Latest trained model")
        metric = "R²" if cb["task"] == "regression" else "Accuracy"
        p = C.tree_profile(cb["model"])
        T.strip([(f"Test {metric}", fmt(cb["test_score"]), f"{len(cb['y_eval']):,} held-out rows"),
                 (f"Training {metric}", fmt(cb["train_score"]), f"{len(cb['y_train']):,} rows"),
                 ("Depth", str(p["depth"]), f"{p['leaves']:,} leaves"),
                 ("Median leaf size", f"{p['median_leaf']:.0f}", "rows per leaf")])
        gap = cb["train_score"] - cb["test_score"]
        (st.warning if gap > 0.1 else st.success)(
            f"Training beats test by {gap:.3f}. " +
            ("That gap suggests overfitting: lower the depth or raise the minimum samples per leaf."
             if gap > 0.1 else "Training and test scores are close, which is what you want."))
        n1, n2, _ = st.columns([1, 1, 2])
        n1.button("Open analytics", on_click=goto, args=("Model analytics",), **T.stretch(st.button))
        n2.button("Make a prediction", on_click=goto, args=("Prediction center",), **T.stretch(st.button))


# --------------------------------------------------------------------------- #
# Page: Model analytics
# --------------------------------------------------------------------------- #
def eval_uploader(b: dict) -> None:
    st.subheader("Score this model on your data")
    st.write("Upload a CSV containing the model's feature columns plus the true value to compare against.")
    up = st.file_uploader("Evaluation CSV", type=["csv"], key=f"evalfile_{b['uid']}")
    if up is None:
        return
    try:
        d = pd.read_csv(up)
    except Exception as exc:
        st.error(f"Could not read that file as CSV: {exc}")
        return
    extra = [c for c in d.columns if c not in b["raw_features"]]
    if not extra:
        st.error("The file has no column left to use as the true value.")
        return
    tgt = st.selectbox("True value column", extra, index=len(extra) - 1, key=f"evaltgt_{b['uid']}")
    if st.button("Evaluate", type="primary", **T.stretch(st.button)):
        try:
            C.attach_evaluation(b, d, tgt)
            ss.pop("report_metrics", None)
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))


def page_analytics() -> None:
    b = active_bundle()
    T.page_header("Model analytics", f"{b['source']} · target {b['target']}")
    flash()
    p = C.tree_profile(b["model"])
    if b["y_eval"] is None:
        T.strip([("Training rows", f"{p['root_n']:,}", "from the model file"), ("Depth", str(p["depth"]), ""),
                 ("Leaves", f"{p['leaves']:,}", f"median {p['median_leaf']:.0f} rows each"),
                 ("Pure leaves", f"{p['pure_share']:.1%}", "zero variance inside")])
        (st.error if p["risk"] == "High" else st.warning if p["risk"] == "Moderate" else st.success)(
            f"Overfitting risk: {p['risk']}. {p['risk_msg']}")
        st.caption("Test metrics need data the model has not seen, and none is stored with this model.")
        eval_uploader(b)
        return

    m = C.evaluate(b)
    ss.report_metrics = (b["uid"], m)
    if b["task"] == "regression":
        T.strip([("R²", fmt(m["r2"]), "share of variance explained"), ("RMSE", fmt(m["rmse"], 2), "typical error, penalises misses"),
                 ("MAE", fmt(m["mae"], 2), "average absolute error"), ("Median error", fmt(m["medae"], 2), f"{m['n']:,} rows")])
        l, r = st.columns(2)
        with l:
            st.markdown("**Actual vs predicted**")
            idx = np.random.default_rng(0).choice(m["n"], min(m["n"], 3000), replace=False)
            lim = [float(min(m["y"].min(), m["p"].min())), float(max(m["y"].max(), m["p"].max()))]
            fig = go.Figure(go.Scatter(x=m["y"][idx], y=m["p"][idx], mode="markers",
                                         marker=dict(size=5, color=T.COBALT, opacity=.4), name="Rows"))
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect",
                                     line=dict(color=T.SUN, dash="dot", width=2)))
            fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
            chart(fig, 360)
        with r:
            st.markdown("**Error distribution**")
            fig = px.histogram(x=m["resid"], nbins=50, color_discrete_sequence=[T.TEAL])
            fig.add_vline(x=0, line=dict(color=T.INK, width=1.5, dash="dot"))
            fig.update_layout(xaxis_title="Actual − predicted", yaxis_title="Rows", bargap=.05)
            chart(fig, 360)
    else:
        T.strip([("Accuracy", fmt(m["accuracy"]), f"{m['n']:,} rows"), ("Precision", fmt(m["precision"]), "weighted"),
                 ("Recall", fmt(m["recall"]), "weighted"), ("F1", fmt(m["f1"]), "weighted")])
        l, r = st.columns(2)
        with l:
            st.markdown("**Confusion matrix**")
            fig = go.Figure(go.Heatmap(z=m["cm"], x=m["names"], y=m["names"], colorscale=T.SEQUENTIAL,
                                       texttemplate="%{z}", showscale=False, xgap=3, ygap=3))
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
            chart(fig, 360)
        with r:
            if "roc" in m:
                st.markdown(f"**ROC curve** (AUC {m['auc']:.3f})")
                fig = go.Figure(go.Scatter(x=m["roc"][0], y=m["roc"][1], fill="tozeroy", mode="lines",
                                           line=dict(color=T.COBALT, width=2.5), fillcolor="rgba(42,91,215,.12)"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=T.MUTED, dash="dot")))
                fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", showlegend=False)
                chart(fig, 360)
            else:
                st.markdown("**Per-class report**")
                table(m["report"].round(3).reset_index().rename(columns={"index": "class"}))

    st.subheader("Does it generalise?")
    if b["X_all"] is None:
        st.caption("Cross-validation needs the training data, which is not stored with an imported model.")
        return
    if st.button("Run 5-fold cross-validation", **T.stretch(st.button)):
        try:
            with st.spinner("Cross-validating…"):
                ss.cv = dict(uid=b["uid"], **C.cv_report(b))
        except ValueError as exc:
            st.error(f"Cross-validation failed: {exc}")
    cv = ss.get("cv")
    if cv and cv["uid"] == b["uid"]:
        name = "R²" if cv["scoring"] == "r2" else "Accuracy"
        st.success(f"Mean {name} across 5 folds: {cv['scores'].mean():.3f} (± {cv['scores'].std() * 2:.3f})"
                   + (f", on a {cv['rows']:,}-row sample." if cv["rows"] < len(b["X_all"]) else "."))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cv["sizes"], y=cv["train"], name="Training", mode="lines+markers",
                                 line=dict(color=T.TEAL, width=2.5)))
        fig.add_trace(go.Scatter(x=cv["sizes"], y=cv["val"], name="Validation", mode="lines+markers",
                                 line=dict(color=T.COBALT, width=2.5)))
        fig.update_layout(xaxis_title="Training rows", yaxis_title=name)
        chart(fig, 320)
        st.caption("If validation keeps climbing as rows increase, more data will help. "
                   "A wide, steady gap means the tree is too flexible.")


# --------------------------------------------------------------------------- #
# Page: Tree explorer
# --------------------------------------------------------------------------- #
def page_tree() -> None:
    b = active_bundle()
    T.page_header("Tree explorer", "See which inputs drive the model and how it splits the data.")
    m, p = b["model"], C.tree_profile(b["model"])
    T.strip([("Depth", str(p["depth"]), "levels from root to deepest leaf"), ("Leaves", f"{p['leaves']:,}", "final answers"),
             ("Median leaf size", f"{p['median_leaf']:.0f}", "training rows"), ("Features", str(len(b["features"])), "after encoding")])
    st.markdown("**Feature importance**")
    imp = pd.Series(m.feature_importances_, index=b["features"]).sort_values().tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=[C.label_for(i) for i in imp.index], orientation="h",
                           marker=dict(color=imp.values, colorscale=T.SEQUENTIAL, cmin=0, cmax=max(imp.max(), 1e-9)),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(tickformat=".0%", title="Share of total error reduction"), yaxis_title=None)
    chart(fig, max(240, 34 * len(imp) + 70))

    st.markdown("**Tree diagram**")
    shown = st.slider("Levels to draw", 1, 4, min(3, max(1, p["depth"])))
    chart(C.tree_figure(m, b["features"], b["classes"], max_depth=min(shown, max(1, p["depth"]))), scroll=True)
    st.caption("Left branch = condition is true. Colour follows the predicted value"
               + (" (cool blue to warm red)." if b["task"] == "regression" else " (one colour per class).")
               + " Wide trees scroll sideways; hover a box for details.")

    st.markdown("**How big are the leaves?**")
    sizes = p["leaf_sizes"]
    fig = px.histogram(x=sizes, nbins=40, log_y=True, color_discrete_sequence=[T.COBALT])
    fig.update_layout(xaxis_title="Training rows in a leaf", yaxis_title="Leaves (log scale)", bargap=.05)
    chart(fig, 300)
    (st.error if p["risk"] == "High" else st.warning if p["risk"] == "Moderate" else st.success)(
        f"Overfitting risk: {p['risk']}. {p['risk_msg']}")

    with st.expander("Read the rules as text"):
        d = st.slider("Rule depth", 1, 6, 3, key="rules_depth")
        st.code(export_text(m, feature_names=b["features"], max_depth=d, decimals=2), language="text")


# --------------------------------------------------------------------------- #
# Page: Prediction center
# --------------------------------------------------------------------------- #
def run_prediction(b: dict) -> None:
    raw = {f: ss[f"pc_{b['uid']}_{f}"] for f in b["raw_features"]}
    X = C.encode_frame(b, pd.DataFrame([raw]))
    pred = b["model"].predict(X)[0]
    res = dict(uid=b["uid"], raw=raw, **C.leaf_info(b["model"], X))
    if b["task"] == "classification":
        k = list(b["model"].classes_).index(pred)
        res.update(label=b["classes"][k], probs=b["model"].predict_proba(X)[0])
    else:
        res["value"] = float(pred)
    ss.pred_result = res


def page_predict() -> None:
    b = active_bundle()
    T.page_header("Prediction center", "Enter values for each feature to get a prediction and see the "
                  "exact questions the tree asked.")
    res = ss.get("pred_result")
    res = res if res and res["uid"] == b["uid"] else None
    if res is None:
        T.hero(state="idle", head=b["target"], title="Fill in the inputs and run the model.")
    elif b["task"] == "classification":
        T.hero(state="idle", head=f"Predicted {b['target']}", value=res["label"],
               sub=f"Confidence {res['probs'].max():.1%}, from {res['n']:,} training rows in the leaf.")
    else:
        T.hero(state="idle", head=f"Predicted {b['target']}", value=f"{res['value']:.2f}", unit=b["unit"],
               sub=f"Average of {res['n']:,} training row{'s' if res['n'] != 1 else ''} in the matching leaf.")
    raw = feature_inputs(b, f"pc_{b['uid']}_")
    range_note(b, raw)
    st.button("Run model", type="primary", on_click=run_prediction, args=(b,), **T.stretch(st.button))
    if res is None:
        return
    st.markdown("&nbsp;")
    if b["task"] == "classification":
        fig = go.Figure(go.Bar(x=res["probs"], y=b["classes"], orientation="h",
                               marker_color=T.COBALT, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1], title="Probability"), yaxis_autorange="reversed")
        st.markdown("**Class probabilities**")
        chart(fig, max(160, 56 * len(b["classes"]) + 60))
    X = C.encode_frame(b, pd.DataFrame([res["raw"]]))
    rows = C.decision_path_rows(b["model"], X, b["features"])
    st.markdown("**Decision path**")
    st.caption(f"{len(rows)} questions to reach leaf #{res['leaf']}.")
    table(pd.DataFrame(rows), height=min(420, 38 * (len(rows) + 1)))


# --------------------------------------------------------------------------- #
# Page: Model hub
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=3)
def _model_bytes(uid: str, _model) -> bytes:
    return C.model_bytes(_model)


def page_hub() -> None:
    b = active_bundle()
    T.page_header("Model hub", "Export the active model, or bring in one you trained elsewhere.")
    left, right = st.columns(2)
    with left:
        st.subheader("Export")
        st.write(f"Active: **{b['source']}**")
        st.download_button("Download model (.pkl)", _model_bytes(b["uid"], b["model"]),
                           file_name=f"decision_tree_{b['uid']}.pkl", mime="application/octet-stream",
                           **T.stretch(st.download_button))
        met = ss.get("report_metrics")
        card = C.model_card(b, met[1] if met and met[0] == b["uid"] else None)
        st.download_button("Download model card (.json)", C.card_json(card), file_name=f"model_card_{b['uid']}.json",
                           mime="application/json", **T.stretch(st.download_button))
        st.caption("The model card records features, ranges, settings and metrics, so the "
                   "model can be understood without opening the pickle.")
    with right:
        st.subheader("Import")
        st.warning("Pickle files can run code when loaded. Import only files you created or trust.")
        up = st.file_uploader("Decision tree (.pkl)", type=["pkl", "joblib"], key="import_pkl")
        if up is not None and st.button("Activate imported model", type="primary", **T.stretch(st.button)):
            try:
                nb = C.load_model_file(up, f"Imported ({up.name})",
                                       target="Prediction" if b["task"] == "regression" else "Class")
            except Exception as exc:
                st.error(f"Could not load that file: {exc}")
            else:
                ss.custom_bundle, ss._ctx_next = nb, "My trained model"
                ss._flash = ("success", "Imported model activated. Add evaluation data in Model analytics to score it.")
                goto("Model analytics")
                st.rerun()
    st.subheader("Compatibility")
    import sklearn
    st.write(f"This app runs scikit-learn **{sklearn.__version__}**. Models are only guaranteed to load "
             "with the version that saved them, so keep `requirements.txt` pinned.")
    for n in b["notes"]:
        st.info(n)


# --------------------------------------------------------------------------- #
# Page: Data explorer
# --------------------------------------------------------------------------- #
def page_explore() -> None:
    ensure_work_df()
    df: pd.DataFrame = ss.work_df
    T.page_header("Data explorer", f"Exploring {ss.work_name}. Change the dataset in Train studio.")
    num = df.select_dtypes("number").columns.tolist()
    views = ["Correlations", "Distributions", "Relationships", "3D view", "Missing data"]
    # A radio (not st.tabs) so only the selected view is computed: st.tabs runs every tab on each click.
    view = st.radio("View", views, horizontal=True, label_visibility="collapsed", key="explore_view")

    def color_arg(label: str, key: str):
        pick = st.selectbox(label, ["None"] + df.columns.tolist(), key=key)
        if pick == "None":
            return None, {}
        if pick in num and df[pick].nunique() > 12:
            return pick, dict(color_continuous_scale=T.SEQUENTIAL)
        return pick, {}

    if view == views[0]:
        if len(num) < 2:
            st.info("Correlations need at least two numeric columns.")
        else:
            corr = df[num].corr()
            fig = go.Figure(go.Heatmap(z=corr.values, x=num, y=num, zmin=-1, zmax=1, zmid=0,
                                       colorscale=T.DIVERGING, texttemplate="%{z:.2f}", xgap=2, ygap=2,
                                       colorbar=dict(thickness=10, len=.8)))
            fig.update_layout(yaxis_autorange="reversed")
            chart(fig, max(380, 56 * len(num)))
    elif view == views[1]:
        col = st.selectbox("Column", df.columns, key="dist_col")
        fig = px.histogram(df, x=col, marginal="box" if col in num else None, color_discrete_sequence=[T.COBALT])
        fig.update_layout(bargap=.04)
        chart(fig, 420)
    elif view == views[2]:
        if len(num) < 2:
            st.info("Relationships need at least two numeric columns.")
        else:
            a, b_, c = st.columns(3)
            x = a.selectbox("X axis", num, key="sc_x")
            y = b_.selectbox("Y axis", num, index=1, key="sc_y")
            with c:
                col, extra = color_arg("Colour by", "sc_c")
            d = df.sample(min(len(df), 3000), random_state=0)
            fig = px.scatter(d, x=x, y=y, color=col, opacity=.6, render_mode="svg", **extra)
            if col is None and x != y:
                ok = d[[x, y]].dropna()
                k, m0 = np.polyfit(ok[x], ok[y], 1)
                xs = np.array([ok[x].min(), ok[x].max()])
                fig.add_trace(go.Scatter(x=xs, y=k * xs + m0, mode="lines", name="Linear trend",
                                         line=dict(color=T.SUN, width=2.5)))
            chart(fig, 460)
    elif view == views[3]:
        if len(num) < 3:
            st.info("The 3D view needs at least three numeric columns.")
        else:
            a, b_, c, d_ = st.columns(4)
            x = a.selectbox("X", num, key="d3_x")
            y = b_.selectbox("Y", num, index=1, key="d3_y")
            z = c.selectbox("Z", num, index=2, key="d3_z")
            with d_:
                col, extra = color_arg("Colour by", "d3_c")
            d = df.sample(min(len(df), 3000), random_state=0)
            fig = px.scatter_3d(d, x=x, y=y, z=z, color=col, opacity=.75, **extra)
            fig.update_traces(marker_size=3)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            chart(fig, 560)
    elif view == views[4]:
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            st.success("No missing values in this dataset.")
        else:
            fig = go.Figure(go.Bar(x=miss.values, y=miss.index, orientation="h", marker_color=T.SUN))
            fig.update_layout(xaxis_title="Missing values", yaxis_title=None)
            chart(fig, max(200, 40 * len(miss) + 80))


# --------------------------------------------------------------------------- #
# Page: Report
# --------------------------------------------------------------------------- #
def page_report() -> None:
    b = active_bundle()
    T.page_header("Report", "A plain-text summary of the active model you can save or share.")
    metrics = C.evaluate(b) if b["y_eval"] is not None else None
    report = C.build_report(b, metrics, C.tree_profile(b["model"]))
    st.download_button("Download report (.md)", report, file_name=f"model_report_{b['uid']}.md",
                       mime="text/markdown", type="primary", **T.stretch(st.download_button))
    st.markdown("&nbsp;")
    with st.container(border=True):
        st.markdown(report)


# --------------------------------------------------------------------------- #
ROUTES = {"Weather forecast": page_forecast, "Train studio": page_train, "Model analytics": page_analytics,
          "Tree explorer": page_tree, "Prediction center": page_predict, "Model hub": page_hub,
          "Data explorer": page_explore, "Report": page_report}
ROUTES[page]()
