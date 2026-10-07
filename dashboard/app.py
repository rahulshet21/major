"""
app.py — Wildfire Burned-Area Dashboard (v2).
Run: python3 -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import json
import zipfile
import io
import subprocess
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_DASH_DIR  = Path(__file__).parent.resolve()
_PROJ_ROOT = _DASH_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_DASH_DIR))

from config import (
    PROJECT_ROOT, DEFAULT_RESULTS_DIR, DEFAULT_CANADA_DIR,
    DEFAULT_WILDFIRE_DIR, DEFAULT_WILDFIRE_PROC, BEST_MODEL_PATH,
    MODELS, MODEL_KEYS, MODEL_LABELS, MODEL_COLORS,
    CH12_TABLE, PHYSICS_TABLE, DEFAULT_PIXEL_SIZE_M, RENDER_MAX_PX, UPLOADS_DIR,
)
from io_utils import (
    discover_events, discover_models_for_event, get_event_folder,
    load_prob, load_mask, load_metrics_csv, load_prediction_png, load_all_models_png,
    get_canada_h5_path, read_canada_groups, load_event_h5,
    discover_wildfire_events, maybe_downsample, make_rgb, disk_size_str,
    align, scan_all_results,
)
from metrics import (
    compute_metrics, best_f1_threshold, pr_roc_curves, dice_vs_threshold,
    ensemble_prob, agreement_map, burned_area_ha, build_leaderboard,
)
import viz
from runner import detect_device, build_predict_cmd, run_predict_streaming, get_error_hint

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Burned-Area Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
css_path = _DASH_DIR / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Pending navigation: must be consumed BEFORE any widget with key="section" ──
# Buttons can't write directly to a key already bound to a widget in the same run.
# Callers write st.session_state["_nav_to"] + st.rerun(); we apply it here first.
_pending_nav = st.session_state.pop("_nav_to", None)
if _pending_nav is not None:
    st.session_state["section"] = _pending_nav

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔥 Wildfire Dashboard")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Section switch
    section = st.radio(
        "Section",
        ["Analyze", "Results", "Library", "Upload & Run"],
        horizontal=False,
        key="section",
    )

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Results dir
    results_dir_str = st.text_input("Results directory", value=str(DEFAULT_RESULTS_DIR), key="results_dir")
    results_dir = Path(results_dir_str)

    if st.button("Reload results", use_container_width=True):
        st.cache_data.clear()
        st.toast("Cache cleared — results reloaded!", icon="♻️")

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    if section == "Analyze":
        # Dataset mode
        dataset_mode = st.radio("Active Dataset", ["Canada (12-ch)", "Wildfire (30-ch)"], horizontal=True, key="dataset_mode")
        is_canada = dataset_mode == "Canada (12-ch)"

        # Event picker
        with st.spinner("Discovering events…"):
            all_events = discover_events(results_dir)
        if not all_events:
            st.warning("No events found in results directory.")
            all_events = ["(none)"]
        selected_event = st.selectbox("Event", all_events, key="selected_event")

        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
        pixel_size_m  = st.number_input("Pixel size (m)", value=DEFAULT_PIXEL_SIZE_M, min_value=1.0, step=1.0)

    # Device badge
    device_str = detect_device()
    st.markdown(f'<div class="device-badge">Device: {device_str}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _fmt_params(v) -> str:
    try:
        f = float(v)
        if f > 1e6:
            return f"{f/1e6:.2f}M"
        elif f > 1e3:
            return f"{f/1e3:.1f}K"
        return str(int(f))
    except Exception:
        return str(v) if v not in (None, "", "N/A") else "N/A"


def _metric_row(m_dict: dict, cols):
    for col, (lbl, key) in zip(cols, [("Dice/F1", "dice"), ("IoU", "iou"), ("Precision", "precision"), ("Recall", "recall")]):
        val = m_dict.get(key)
        col.metric(lbl, f"{float(val):.3f}" if val is not None else "N/A")


def _degenerate_warning(model_key: str, dice_val: float, prob: np.ndarray | None):
    if dice_val < 0.01:
        max_p = float(prob.max()) if prob is not None else 0.0
        st.markdown(
            f'<div class="degenerate-warn">⚠ <b>{MODEL_LABELS.get(model_key, model_key)}</b> predicts ~no burned pixels '
            f'(Dice={dice_val:.4f}, max prob={max_p:.3f}). This model may have a domain mismatch '
            f'or needs threshold tuning in the What-If expander below.</div>',
            unsafe_allow_html=True,
        )


def _load_gt_for_event(event: str) -> np.ndarray | None:
    """Load GT mask in priority order — never crash."""
    folder = get_event_folder(results_dir, event)
    for name in [f"cmp_{event}_gt.npy", "mask.npy", f"{event}_mask.npy"]:
        p = folder / name
        if p.exists():
            try:
                return np.load(str(p), mmap_mode="r").astype(np.uint8)
            except Exception:
                pass
    # Canada HDF5
    if section == "Analyze" and st.session_state.get("dataset_mode", "").startswith("Canada"):
        h5p = get_canada_h5_path(Path(st.session_state.get("canada_dir_input", str(DEFAULT_CANADA_DIR))))
        if h5p:
            _, gt = load_event_h5(str(h5p), event)
            if gt is not None:
                return gt.astype(np.uint8)
    # Wildfire processed
    wf_mask = Path(st.session_state.get("wf_proc_input", str(DEFAULT_WILDFIRE_PROC))) / f"{event}_mask.npy"
    if wf_mask.exists():
        try:
            return np.load(str(wf_mask), mmap_mode="r").astype(np.uint8)
        except Exception:
            pass
    return None


def _build_leaderboard_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format leaderboard for display."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    for c in ["dice", "iou", "precision", "recall"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(3)
    if "params" in d.columns:
        d["params"] = d["params"].apply(_fmt_params)
    if "inference_seconds" in d.columns:
        d["inference_seconds"] = pd.to_numeric(d["inference_seconds"], errors="coerce").round(2)
    rename = {
        "model": "Model", "dice": "Dice", "iou": "IoU",
        "precision": "Precision", "recall": "Recall",
        "params": "Params", "inference_seconds": "Inference (s)"
    }
    d = d.rename(columns=rename)
    d["Model"] = d["Model"].apply(lambda k: MODEL_LABELS.get(str(k), str(k)))
    cols = [c for c in ["Model", "Dice", "IoU", "Precision", "Recall", "Params", "Inference (s)"] if c in d.columns]
    return d[cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
if section == "Analyze":
    event = selected_event if selected_event != "(none)" else None

    if not event:
        st.info("No events found. Run a prediction first or adjust the Results directory.")
        st.stop()

    metrics_df = load_metrics_csv(results_dir_str, event)
    lb = build_leaderboard(metrics_df)
    gt = _load_gt_for_event(event)

    # ── Tab structure ──────────────────────────────────────────────────────────
    TAB_NAMES = [
        "Overview", "U-Net", "DeepLabV3+", "FC-Siam", "SegFormer",
        "Random Forest", "CA-LMoETransUNet",
        "Compare", "Threshold Lab", "Model Info", "Export",
    ]
    tabs = st.tabs(TAB_NAMES)

    # ── Tab 0: Overview ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown(f"### Event: `{event}`")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        if lb.empty:
            st.info("No metrics CSV found for this event. Images may still be available in the model tabs.")
        else:
            best_row   = lb.iloc[0]
            best_model = str(best_row.get("model", ""))
            best_dice  = float(best_row.get("dice", 0) or 0)
            best_iou   = float(best_row.get("iou",  0) or 0)

            # KPI cards
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Best Model", MODEL_LABELS.get(best_model, best_model))
            c2.metric("Best Dice", f"{best_dice:.3f}")
            c3.metric("Best IoU",  f"{best_iou:.3f}")
            if gt is not None:
                burned_px = int(gt.sum())
                total_px  = gt.size
                c4.metric("GT Burned px", f"{burned_px:,} ({burned_px/total_px*100:.1f}%)")
                c5.metric("GT Area", f"{burned_area_ha(gt, pixel_size_m):,.0f} ha")

            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

            # Leaderboard
            st.markdown("#### Model Leaderboard")
            lb_disp = _build_leaderboard_display(lb)
            if not lb_disp.empty:
                try:
                    styled = lb_disp.style.background_gradient(
                        subset=[c for c in ["Dice", "IoU"] if c in lb_disp.columns],
                        cmap="YlOrRd"
                    )
                    st.dataframe(styled, use_container_width=True, hide_index=True)
                except Exception:
                    st.dataframe(lb_disp, use_container_width=True, hide_index=True)

            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

            col_r, col_b = st.columns(2)
            with col_r:
                st.plotly_chart(viz.radar_chart(lb), use_container_width=True)
            with col_b:
                st.plotly_chart(viz.grouped_bar(lb), use_container_width=True)
            st.plotly_chart(viz.bubble_chart(lb), use_container_width=True)

            # Auto insights
            st.markdown("#### Insights")
            insights = [f"**Best model:** {MODEL_LABELS.get(best_model, best_model)} — Dice={best_dice:.3f}, IoU={best_iou:.3f}."]
            if "dice" in lb.columns:
                zero_models = lb[pd.to_numeric(lb["dice"], errors="coerce").fillna(0) < 0.01]["model"].tolist()
                if zero_models:
                    insights.append(f"**Degenerate (≈0 Dice):** {', '.join(MODEL_LABELS.get(m, m) for m in zero_models)} — try lower threshold in Threshold Lab.")
            st.markdown('<div class="insights-box">' + "<br>".join(insights) + "</div>", unsafe_allow_html=True)

        # All-models comparison image
        all_img = load_all_models_png(results_dir_str, event)
        if all_img:
            with st.expander("All-Models Comparison Image (click to expand)", expanded=False):
                st.image(all_img, use_container_width=True)

    # ── Tabs 1–6: Per-Model ────────────────────────────────────────────────────
    for tab_idx, model_info in enumerate(MODELS):
        mkey   = model_info["key"]
        mlabel = model_info["label"]
        mcolor = model_info["color"]
        tab    = tabs[tab_idx + 1]

        with tab:
            st.markdown(f"### {mlabel}")
            st.caption(model_info["blurb"])
            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

            # Metric cards from CSV
            m_row = {}
            if metrics_df is not None:
                row_mask = metrics_df["model"] == mkey
                if row_mask.any():
                    m_row = metrics_df[row_mask].iloc[0].to_dict()

            if m_row:
                dice_val = float(m_row.get("dice", 0) or 0)
                _degenerate_warning(mkey, dice_val, load_prob(results_dir_str, event, mkey))
                metric_cols = st.columns(5)
                metric_cols[0].metric("Dice / F1", f"{dice_val:.3f}")
                metric_cols[1].metric("IoU",       f"{float(m_row.get('iou', 0) or 0):.3f}")
                metric_cols[2].metric("Precision",  f"{float(m_row.get('precision', 0) or 0):.3f}")
                metric_cols[3].metric("Recall",     f"{float(m_row.get('recall', 0) or 0):.3f}")
                metric_cols[4].metric("Params",     _fmt_params(m_row.get("params")))
                if not lb.empty:
                    delta = dice_val - float(lb.iloc[0].get("dice", 0) or 0)
                    if mkey != lb.iloc[0].get("model"):
                        st.caption(f"Δ vs best: {delta:+.3f}")

            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

            # ── MAIN VIEW: Saved prediction PNG ───────────────────────────────
            pred_png = load_prediction_png(results_dir_str, event, mkey)
            if pred_png:
                st.markdown("#### Saved Prediction (4-panel: S2 Pre | GT | Probability | Overlay)")
                st.image(pred_png, use_container_width=True)
                dl_buf = io.BytesIO()
                pred_png.save(dl_buf, format="PNG")
                st.download_button(
                    f"Download {mkey}_prediction.png",
                    data=dl_buf.getvalue(),
                    file_name=f"{event}_{mkey}_prediction.png",
                    key=f"dl_pred_png_{mkey}",
                )
            else:
                st.info(f"No saved prediction PNG for **{mlabel}** on `{event}`.")

            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

            # ── Interactive probability heatmap ────────────────────────────────
            prob = load_prob(results_dir_str, event, mkey)
            if prob is not None:
                st.markdown("#### Interactive Probability Heatmap")
                # Downsample only for rendering — no GT needed here
                prob_ds = maybe_downsample(prob, RENDER_MAX_PX)
                fig_heat = viz.prob_heatmap(prob_ds, title=f"{mlabel} — Probability")
                st.plotly_chart(fig_heat, use_container_width=True)

                # Download prob npy
                st.download_button(
                    f"Download {mkey}_prob.npy",
                    data=prob.tobytes(),
                    file_name=f"{event}_{mkey}_prob.npy",
                    key=f"dl_prob_{mkey}",
                )
            else:
                st.info("No probability .npy file for this model/event.")

            # ── What-if analysis (only if GT + prob both available) ────────────
            if prob is not None and gt is not None:
                with st.expander("What-if analysis (recomputed, not the saved result)", expanded=False):
                    local_t = st.slider(
                        "Threshold", 0.0, 1.0, 0.5, 0.01,
                        key=f"thresh_{mkey}",
                    )
                    # Align prob and gt before any indexing
                    prob_aligned, gt_aligned = align(prob, gt, max_side=0)  # max_side=0 = no downsample
                    if prob_aligned is not None and gt_aligned is not None:
                        from metrics import compute_metrics_at_threshold
                        wm = compute_metrics_at_threshold(prob_aligned, gt_aligned, local_t)
                        wc = st.columns(4)
                        for col, (lbl, key) in zip(wc, [("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"), ("Recall", "recall")]):
                            col.metric(lbl, f"{wm.get(key, 0):.3f}")

                        if st.button(f"Find best-F1 threshold", key=f"bf1_{mkey}"):
                            bt = best_f1_threshold(prob_aligned, gt_aligned)
                            st.info(f"Best-F1 threshold = **{bt:.2f}**")

                        st.markdown("**Probability Distribution**")
                        # Align for histogram — use downsampled version
                        prob_dsh, gt_dsh = align(prob, gt, max_side=800)
                        if prob_dsh is not None and gt_dsh is not None:
                            fig_h = viz.prob_histogram(prob_dsh, gt_dsh)
                            st.plotly_chart(fig_h, use_container_width=True)

                        st.markdown("**PR / ROC Curves**")
                        thrs, precs, recalls, fprs = pr_roc_curves(prob_aligned, gt_aligned)
                        col_pr, col_roc = st.columns(2)
                        col_pr.plotly_chart(viz.pr_curve_fig(thrs, precs, recalls, mlabel, mcolor), use_container_width=True)
                        col_roc.plotly_chart(viz.roc_curve_fig(fprs, recalls, mlabel, mcolor), use_container_width=True)

    # ── Tab 7: Compare ────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("### Model Comparison")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        all_img = load_all_models_png(results_dir_str, event)
        if all_img:
            st.markdown("#### All-Models Comparison Grid")
            st.image(all_img, use_container_width=True)
            dl_buf = io.BytesIO()
            all_img.save(dl_buf, format="PNG")
            st.download_button("Download comparison grid",
                               data=dl_buf.getvalue(),
                               file_name=f"{event}_ALL_MODELS_comparison.png")
        else:
            st.info("No ALL_MODELS_comparison.png found for this event.")

        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        avail = discover_models_for_event(results_dir, event)
        if avail:
            st.markdown("#### Side-by-side Saved Predictions")
            sel = st.multiselect(
                "Select models to view side by side",
                avail,
                default=avail[:min(3, len(avail))],
                format_func=lambda k: MODEL_LABELS.get(k, k),
                key="compare_sel",
            )
            if sel:
                cmp_cols = st.columns(len(sel))
                for k, col in zip(sel, cmp_cols):
                    png = load_prediction_png(results_dir_str, event, k)
                    col.markdown(f"**{MODEL_LABELS.get(k, k)}**")
                    if png:
                        col.image(png, use_container_width=True)
                    else:
                        col.info("PNG not available")

        # Ensemble from probs (when available)
        probs_avail = {k: load_prob(results_dir_str, event, k) for k in avail}
        probs_valid = [p for p in probs_avail.values() if p is not None]
        if len(probs_valid) >= 2:
            st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### Ensemble Probability (Mean)")
            # Align all probs to each other first
            aligned_probs = align(*probs_valid, max_side=RENDER_MAX_PX)
            aligned_probs = [a for a in aligned_probs if a is not None]
            ens = ensemble_prob(aligned_probs)
            st.plotly_chart(viz.prob_heatmap(ens, "Ensemble (Mean Probability)"), use_container_width=True)

            agr = agreement_map(aligned_probs)
            st.plotly_chart(viz.agreement_map_fig(agr, len(aligned_probs)), use_container_width=True)

    # ── Tab 8: Threshold Lab ───────────────────────────────────────────────────
    with tabs[8]:
        st.markdown("### Threshold Lab")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        if gt is None:
            st.info("Ground truth mask not available for this event. Threshold analysis requires GT.")
        else:
            avail_tl = discover_models_for_event(results_dir, event)
            with st.spinner("Computing Dice vs threshold curves…"):
                curves = {}
                best_thresholds = {}
                for mk in avail_tl:
                    p = load_prob(results_dir_str, event, mk)
                    if p is not None:
                        # align before any per-pixel op
                        pa, ga = align(p, gt, max_side=0)
                        if pa is not None and ga is not None:
                            thrs, dices = dice_vs_threshold(pa, ga)
                            curves[mk] = (thrs, dices)
                            best_thresholds[mk] = float(thrs[np.argmax(dices)])

            if curves:
                st.plotly_chart(viz.dice_threshold_chart(curves), use_container_width=True)
                rows_t = [
                    {"Model": MODEL_LABELS.get(k, k), "Best Threshold": f"{v:.2f}",
                     "Best Dice": f"{curves[k][1][np.argmax(curves[k][1])]:.3f}"}
                    for k, v in best_thresholds.items()
                ]
                st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

    # ── Tab 9: Model Info ──────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown("### Model Architectures & Training Details")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        st.markdown("#### CA-LMoETransUNet (Existing Primary Model)")
        st.markdown("""
**Architecture:** 12→64→128→256→128 encoder | MoE Transformer bottleneck (3 experts, 4 heads, Linear-ReLU attention) | Skip-connection decoder | 2-class head | **3.78M params**

**Training:** CrossEntropy + Dice loss | AdamW (lr=1e-4, wd=1e-4) | CosineAnnealing → 1e-6 | Flip augmentation | 80/20 split seed=42 | Best epoch 19 → **Dice=0.6924, IoU=0.5337**

**Synthetic test:** Acc=99.70%, Prec=99.99%, Recall=98.90%, F1=99.45%, IoU=98.90%
        """)

        st.markdown("#### 12-Channel Input Layout")
        ch_df = pd.DataFrame(CH12_TABLE, columns=["Channel", "Sensor", "Description"])
        ch_df.insert(0, "Index", range(len(ch_df)))
        st.dataframe(ch_df, use_container_width=True, hide_index=True)

        st.markdown("#### Fire Physics Signatures")
        ph_df = pd.DataFrame(PHYSICS_TABLE, columns=["Band", "Change", "Explanation"])
        st.dataframe(ph_df, use_container_width=True, hide_index=True)

        c_col, l_col = st.columns(2)
        c_col.markdown("**Can:**\n- Detect post-fire burn scars from S1+S2\n- Generalize across boreal Canada fires\n- Handle large images (tiled inference)\n- Compare against 5 alternative architectures")
        l_col.markdown("**Cannot:**\n- Detect active fire\n- Handle cloud-obscured optical\n- Guarantee tropical/urban generalization\n- Accept 30-ch wildfire inputs directly")

        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Alternative Models")
        for m in MODELS:
            if m["key"] == "calmoe":
                continue
            with st.expander(m["label"], expanded=False):
                st.markdown(f"**Architecture:** {m['architecture']}")
                st.markdown(m["blurb"])

    # ── Tab 10: Export ─────────────────────────────────────────────────────────
    with tabs[10]:
        st.markdown("### Export")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"**Event:** `{event}`")

        if metrics_df is not None:
            st.download_button(
                "Download Metrics CSV",
                data=metrics_df.to_csv(index=False).encode(),
                file_name=f"{event}_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if st.button("Build PNG Gallery ZIP", use_container_width=True):
            with st.spinner("Zipping…"):
                zip_buf = io.BytesIO()
                folder = get_event_folder(results_dir, event)
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for png in folder.glob(f"cmp_{event}*.png"):
                        zf.write(png, png.name)
                zip_buf.seek(0)
            st.download_button("Download PNG Gallery ZIP",
                               data=zip_buf,
                               file_name=f"{event}_gallery.zip",
                               use_container_width=True)

        if st.button("Generate HTML Report", use_container_width=True):
            lb_html = _build_leaderboard_display(lb).to_html(index=False) if not lb.empty else "<p>No metrics.</p>"
            html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Wildfire Report — {event}</title>
<style>body{{font-family:Inter,sans-serif;background:#0e1117;color:#e8eaf0;padding:40px}}
h1,h2{{color:#ff6b35}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #2a2e45;padding:8px 12px;text-align:left}}th{{background:#1a1d26}}</style>
</head><body>
<h1>Wildfire Report: {event}</h1>
{lb_html}
<p><i>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</i></p>
</body></html>"""
            st.download_button("Download HTML Report", data=html.encode(),
                               file_name=f"{event}_report.html", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: LIBRARY  (analytics, gallery, cross-event, import)
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Library":
    st.markdown("## Library")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    with st.spinner("Scanning all events…"):
        lib_df = scan_all_results(results_dir)

    if lib_df.empty:
        st.info("No results found in the results directory.")
        st.stop()

    # Search & filter
    fcol1, fcol2, fcol3 = st.columns([3, 1, 1])
    search = fcol1.text_input("Search event", value="", placeholder="e.g. CA_2018 or BC", key="lib_search")
    years  = ["All"] + sorted(lib_df["year"].dropna().unique().tolist(), reverse=True)
    provs  = ["All"] + sorted(lib_df["province"].dropna().unique().tolist())
    year_f = fcol2.selectbox("Year", years, key="lib_year")
    prov_f = fcol3.selectbox("Province", provs, key="lib_prov")

    filtered = lib_df.copy()
    if search:
        filtered = filtered[filtered["event"].str.contains(search, case=False, na=False)]
    if year_f != "All":
        filtered = filtered[filtered["year"] == year_f]
    if prov_f != "All":
        filtered = filtered[filtered["province"] == prov_f]

    # Display table
    disp = filtered.copy()
    disp["last_modified"] = disp["last_modified"].apply(
        lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "")
    disp["best_dice"] = disp["best_dice"].apply(lambda x: f"{x:.3f}" if x else "—")
    disp["CSV"] = disp["has_csv"].apply(lambda x: "✓" if x else "✗")
    disp["Grid"] = disp["has_grid"].apply(lambda x: "✓" if x else "✗")
    disp["Best Model"] = disp["best_model"].apply(lambda k: MODEL_LABELS.get(k, k))
    show_cols = ["event", "year", "province", "Best Model", "best_dice", "n_models", "CSV", "Grid", "last_modified"]
    show_cols = [c for c in show_cols if c in disp.columns]
    st.dataframe(disp[show_cols].rename(columns={"best_dice": "Best Dice", "n_models": "#Models",
                                                   "last_modified": "Last Modified", "event": "Event",
                                                   "year": "Year", "province": "Province"}),
                 use_container_width=True, hide_index=True)

    st.markdown(f"**{len(filtered)} events** found")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Thumbnail gallery
    st.markdown("### Comparison Image Gallery")
    thumb_events = filtered[filtered["has_grid"]]["event"].tolist()
    if thumb_events:
        n_cols = 3
        rows_g = [thumb_events[i:i+n_cols] for i in range(0, len(thumb_events), n_cols)]
        for row_evs in rows_g:
            cols_g = st.columns(n_cols)
            for col, ev in zip(cols_g, row_evs):
                img = load_all_models_png(results_dir_str, ev)
                if img:
                    thumb = img.copy()
                    thumb.thumbnail((400, 400), Image.LANCZOS)
                    col.image(thumb, caption=ev, use_container_width=True)
                    if col.button("Open in Analyze", key=f"open_{ev}"):
                        st.session_state["selected_event"] = ev
                        st.session_state["_nav_to"] = "Analyze"
                        st.rerun()
    else:
        st.info("No ALL_MODELS_comparison.png files found.")

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Cross-event performance
    st.markdown("### Cross-Event Model Performance")
    if not filtered[filtered["has_csv"]].empty:
        # Collect per-event metrics
        all_rows = []
        for ev in filtered["event"]:
            df_ev = load_metrics_csv(results_dir_str, ev)
            if df_ev is not None:
                df_ev = df_ev.copy()
                df_ev["event"] = ev
                all_rows.append(df_ev)

        if all_rows:
            agg = pd.concat(all_rows, ignore_index=True)
            agg["dice"] = pd.to_numeric(agg.get("dice", 0), errors="coerce")
            agg["iou"]  = pd.to_numeric(agg.get("iou", 0),  errors="coerce")

            # Heatmap table: event × model
            pivot = agg.pivot_table(index="event", columns="model", values="dice", aggfunc="first")
            st.markdown("#### Dice Score Heatmap (event × model)")
            try:
                styled_pivot = pivot.style.background_gradient(cmap="YlOrRd", axis=None)
                st.dataframe(styled_pivot, use_container_width=True)
            except Exception:
                st.dataframe(pivot, use_container_width=True)

            # Per-model summary
            st.markdown("#### Per-Model Summary")
            summary = agg.groupby("model").agg(
                Mean_Dice=("dice", "mean"),
                Median_Dice=("dice", "median"),
                Std_Dice=("dice", "std"),
                Mean_IoU=("iou", "mean"),
                Events=("event", "count"),
            ).reset_index()
            # Win count
            if not pivot.empty:
                wins = pivot.idxmax(axis=1).value_counts().reset_index()
                wins.columns = ["model", "Wins"]
                summary = summary.merge(wins, on="model", how="left").fillna({"Wins": 0})
                summary["Wins"] = summary["Wins"].astype(int)
            summary["Mean_Dice"] = summary["Mean_Dice"].round(3)
            summary["Median_Dice"] = summary["Median_Dice"].round(3)
            summary["Std_Dice"] = summary["Std_Dice"].round(3)
            summary["Model"] = summary["model"].apply(lambda k: MODEL_LABELS.get(k, k))
            st.dataframe(summary.drop(columns=["model"]), use_container_width=True, hide_index=True)

            # Box plot
            fig_box = go.Figure()
            for mk in agg["model"].unique():
                dvals = agg[agg["model"] == mk]["dice"].dropna().tolist()
                if dvals:
                    fig_box.add_trace(go.Box(
                        y=dvals, name=MODEL_LABELS.get(mk, mk),
                        marker_color=MODEL_COLORS.get(mk, "#aaa"),
                        boxmean=True,
                    ))
            fig_box.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                   title="Dice Score Distribution per Model",
                                   yaxis=dict(title="Dice", range=[0, 1.05]))
            st.plotly_chart(fig_box, use_container_width=True)

            # Download aggregate
            st.download_button("Download Aggregate CSV",
                               data=agg.to_csv(index=False).encode(),
                               file_name="aggregate_metrics.csv")

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Import old outputs
    st.markdown("### Import Previous Outputs")
    st.caption("Drag and drop old cmp_* files (PNG, CSV, NPY) to add them to the Results Library.")
    imp_files = st.file_uploader("Import files",
                                  accept_multiple_files=True,
                                  type=["png", "csv", "npy"],
                                  key="lib_import")
    overwrite_ok = st.checkbox("Overwrite existing files", value=False, key="lib_overwrite")
    if imp_files:
        saved, skipped = [], []
        for f in imp_files:
            dest = results_dir / f.name
            if dest.exists() and not overwrite_ok:
                skipped.append(f.name)
            else:
                dest.write_bytes(f.read())
                saved.append(f.name)
        if saved:
            st.success(f"Saved: {', '.join(saved)}")
        if skipped:
            st.warning(f"Skipped (already exist): {', '.join(skipped)}")
        if saved:
            st.cache_data.clear()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: RESULTS  (search + browse + Files button)
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Results":
    st.markdown("## Results")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # Files button — opens results folder in Finder
    col_files, col_reload = st.columns([1, 1])
    if col_files.button("📂  Open Files Folder", use_container_width=True, key="open_finder"):
        subprocess.Popen(["open", str(results_dir)])
        st.toast(f"Opening {results_dir} in Finder", icon="📂")
    col_reload.button("Reload", use_container_width=True,
                      on_click=lambda: st.cache_data.clear(), key="res_reload")

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    with st.spinner("Scanning events…"):
        res_df = scan_all_results(results_dir)

    if res_df.empty:
        st.info("No results found. Run a prediction first.")
        st.stop()

    # ── Search bar + Dice range slider ────────────────────────────────
    st.markdown("### Search & Filter")
    res_search = st.text_input("Search events",
                               placeholder="e.g. CA_2018, BC, EMSR…",
                               key="res_search")

    # Fill NaN dice so slider comparison never crashes
    res_df["best_dice"] = pd.to_numeric(res_df["best_dice"], errors="coerce").fillna(0.0)
    dice_min_all = float(res_df["best_dice"].min())
    dice_max_all = float(res_df["best_dice"].max())
    if dice_min_all == dice_max_all:
        dice_max_all = min(dice_min_all + 0.01, 1.0)
    dice_range = st.slider(
        "Best Dice range",
        min_value=0.0, max_value=1.0,
        value=(round(dice_min_all, 2), round(dice_max_all, 2)),
        step=0.01, key="res_dice_slider",
    )

    years_r = ["All"] + sorted(res_df["year"].dropna().unique().tolist(), reverse=True)
    provs_r = ["All"] + sorted(res_df["province"].dropna().unique().tolist())
    yf, pf = st.columns(2)
    year_rf = yf.selectbox("Year", years_r, key="res_year")
    prov_rf = pf.selectbox("Province", provs_r, key="res_prov")

    # Apply filters
    res_filtered = res_df.copy()
    if res_search:
        res_filtered = res_filtered[res_filtered["event"].str.contains(res_search, case=False, na=False)]
    if year_rf != "All":
        res_filtered = res_filtered[res_filtered["year"] == year_rf]
    if prov_rf != "All":
        res_filtered = res_filtered[res_filtered["province"] == prov_rf]
    res_filtered = res_filtered[
        (res_filtered["best_dice"] >= dice_range[0]) &
        (res_filtered["best_dice"] <= dice_range[1])
    ]

    st.markdown(f"**{len(res_filtered)} events match**")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # ── Event table ─────────────────────────────────────────────────────────
    disp_r = res_filtered.copy()
    disp_r["Last Modified"] = disp_r["last_modified"].apply(
        lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "")
    disp_r["Best Dice"]  = disp_r["best_dice"].apply(lambda x: f"{x:.3f}" if x else "—")
    disp_r["Best Model"] = disp_r["best_model"].apply(lambda k: MODEL_LABELS.get(k, k))
    disp_r["CSV"]  = disp_r["has_csv"].apply(lambda x: "✓" if x else "✗")
    disp_r["Grid"] = disp_r["has_grid"].apply(lambda x: "✓" if x else "✗")
    disp_r = disp_r.rename(columns={"event": "Event", "year": "Year",
                                     "province": "Province", "n_models": "#Models"})
    table_cols = [c for c in ["Event", "Year", "Province", "Best Model", "Best Dice",
                               "#Models", "CSV", "Grid", "Last Modified"] if c in disp_r.columns]
    st.dataframe(disp_r[table_cols], use_container_width=True, hide_index=True)

    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

    # ── Event detail panel ──────────────────────────────────────────────────
    st.markdown("### Event Detail")
    # Use original lowercase 'event' column from res_filtered (disp_r renamed it but res_filtered wasn't changed)
    ev_list = res_filtered["event"].tolist() if not res_filtered.empty else []
    if not ev_list:
        st.info("No events match the current filters.")
    else:
        detail_ev = st.selectbox("Select an event to inspect", ev_list, key="res_detail_ev")
        mdf_r = load_metrics_csv(results_dir_str, detail_ev)
        avail_r = discover_models_for_event(results_dir, detail_ev)

        info_col, img_col = st.columns([1, 2])
        with info_col:
            st.markdown(f"**Event:** `{detail_ev}`")
            if mdf_r is not None:
                lb_r = build_leaderboard(mdf_r)
                best_r = lb_r.iloc[0] if not lb_r.empty else {}
                st.metric("Best Model", MODEL_LABELS.get(str(best_r.get("model","")),str(best_r.get("model","N/A"))))
                st.metric("Best Dice",  f"{float(best_r.get('dice', 0) or 0):.3f}")
                st.metric("Best IoU",   f"{float(best_r.get('iou',  0) or 0):.3f}")
            else:
                st.info("No metrics CSV")

            st.markdown(f"**Available models:** {', '.join(MODEL_LABELS.get(k,k) for k in avail_r) or 'none'}")

            folder_r = get_event_folder(results_dir, detail_ev)
            st.markdown(f"**Folder:** `{folder_r}`")
            if st.button("📂 Open in Finder", key=f"finder_{detail_ev}"):
                subprocess.Popen(["open", str(folder_r)])
                st.toast(f"Opening {folder_r}", icon="📂")

            if st.button("📊 Open in Analyze", type="primary",
                         use_container_width=True, key=f"res_analyze_{detail_ev}"):
                st.session_state["selected_event"] = detail_ev
                st.session_state["_nav_to"] = "Analyze"
                st.rerun()

        with img_col:
            grid_img = load_all_models_png(results_dir_str, detail_ev)
            if grid_img:
                st.image(grid_img, caption="All-Models Comparison", use_container_width=True)
            else:
                # Fall back to first available model PNG
                for mk in avail_r:
                    png_r = load_prediction_png(results_dir_str, detail_ev, mk)
                    if png_r:
                        st.image(png_r, caption=f"{MODEL_LABELS.get(mk,mk)} prediction",
                                 use_container_width=True)
                        break
                else:
                    st.info("No prediction images saved for this event.")

        if mdf_r is not None:
            st.markdown("**Metrics:**")
            lb_disp_r = _build_leaderboard_display(build_leaderboard(mdf_r))
            if not lb_disp_r.empty:
                st.dataframe(lb_disp_r, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: UPLOAD & RUN
# ═══════════════════════════════════════════════════════════════════════════════
elif section == "Upload & Run":
    st.markdown("## Upload & Run")
    st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"**Device:** `{device_str}` | **Results dir:** `{results_dir_str}`")

    # ── Inline results preview helper ──────────────────────────────────────
    def _inline_results_preview(event_name: str):
        """Show run results inline without navigating away."""
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"### 🔍 Results Preview — `{event_name}`")

        # Toggle: show/hide preview
        show_preview = st.toggle("Show output preview", value=True, key=f"toggle_preview_{event_name}")
        if not show_preview:
            st.caption("Toggle on to inspect results, or switch to **Analyze** in the sidebar for the full dashboard.")
            return

        st.cache_data.clear()

        # Comparison grid
        all_img = load_all_models_png(results_dir_str, event_name)
        if all_img:
            st.markdown("#### All-Models Comparison Grid")
            st.image(all_img, use_container_width=True)
            dl_buf = io.BytesIO()
            all_img.save(dl_buf, format="PNG")
            st.download_button("Download grid", data=dl_buf.getvalue(),
                               file_name=f"{event_name}_ALL_MODELS.png",
                               key=f"dl_grid_{event_name}")
        else:
            st.info("Comparison grid not saved yet (may still be generating).")

        # Metrics leaderboard
        mdf = load_metrics_csv(results_dir_str, event_name)
        if mdf is not None:
            st.markdown("#### Metrics Leaderboard")
            lb_p = build_leaderboard(mdf)
            lb_disp = _build_leaderboard_display(lb_p)
            if not lb_disp.empty:
                try:
                    st.dataframe(
                        lb_disp.style.background_gradient(
                            subset=[c for c in ["Dice", "IoU"] if c in lb_disp.columns],
                            cmap="YlOrRd"
                        ),
                        use_container_width=True, hide_index=True
                    )
                except Exception:
                    st.dataframe(lb_disp, use_container_width=True, hide_index=True)

        # Per-model saved PNGs
        avail_m = discover_models_for_event(results_dir, event_name)
        if avail_m:
            st.markdown("#### Per-Model Predictions")
            tab_labels = [MODEL_LABELS.get(k, k) for k in avail_m]
            model_tabs = st.tabs(tab_labels)
            for mk, mtab in zip(avail_m, model_tabs):
                with mtab:
                    png = load_prediction_png(results_dir_str, event_name, mk)
                    if png:
                        st.image(png, use_container_width=True)
                        # Interactive heatmap
                        prob_p = load_prob(results_dir_str, event_name, mk)
                        if prob_p is not None:
                            prob_ds_p = maybe_downsample(prob_p, RENDER_MAX_PX)
                            st.plotly_chart(
                                viz.prob_heatmap(prob_ds_p, f"{MODEL_LABELS.get(mk, mk)} Probability"),
                                use_container_width=True
                            )
                    else:
                        st.info(f"No prediction PNG for {MODEL_LABELS.get(mk, mk)}.")

        # Jump button — uses _nav_to to avoid 'widget key already instantiated' error
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)
        if st.button("Open full analysis in Analyze tab ↗",
                     type="primary", use_container_width=True,
                     key=f"jump_analyze_{event_name}"):
            st.session_state["_nav_to"] = "Analyze"
            st.session_state["selected_event"] = event_name
            st.rerun()

    # Show persisted preview from last run (survives widget interactions)
    _last_run_event = st.session_state.get("last_run_event_upload", "")

    card_a, card_b = st.columns(2)

    # ── Card A: Existing dataset ───────────────────────────────────────────────
    with card_a:
        st.markdown("### Card A — Existing Dataset")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        ds_choice = st.radio("Dataset", ["Canada 12-ch", "Wildfire 30-ch"], key="run_ds_choice")
        is_canada_run = ds_choice == "Canada 12-ch"

        if is_canada_run:
            ca_dir_str = st.text_input("Canada data dir", value=str(DEFAULT_CANADA_DIR), key="canada_dir_input")
            ca_dir = Path(ca_dir_str)
            h5p = get_canada_h5_path(ca_dir)
            if h5p:
                st.success(f"Found ({disk_size_str(h5p)})")
            else:
                st.error("HDF5 not found — select the correct path above.")
                st.caption("Note: The 21 GB .h5 cannot be browser-uploaded. Specify the path above.")

            ca_events = read_canada_groups(ca_dir)
            if ca_events:
                run_event_a = st.selectbox("Event", ca_events, key="run_event_a")
                if st.button("Inspect", key="inspect_a") and h5p:
                    with st.spinner("Loading…"):
                        img12, gt12 = load_event_h5(str(h5p), run_event_a)
                    if img12 is not None:
                        st.json({"shape": list(img12.shape), "dtype": str(img12.dtype),
                                 "range": [float(img12.min()), float(img12.max())],
                                 "burned_px": int(gt12.sum()) if gt12 is not None else None})
                    else:
                        st.error("Could not load event.")
            else:
                run_event_a = st.text_input("Event name", key="run_event_a_text")
        else:
            wf_dir_str = st.text_input("Wildfire dataset dir", value=str(DEFAULT_WILDFIRE_DIR), key="wf_dir_input")
            wf_proc_str = st.text_input("Wildfire processed dir", value=str(DEFAULT_WILDFIRE_PROC), key="wf_proc_input")
            wf_dir = Path(wf_dir_str)
            if wf_dir.is_dir():
                st.success(f"Found ({disk_size_str(wf_dir)})")
            else:
                st.warning("Wildfire dir not found — falling back to processed dir.")
            wf_events = discover_wildfire_events(Path(wf_proc_str))
            if wf_events:
                run_event_a = st.selectbox("Event", wf_events, key="run_event_wf")
            else:
                run_event_a = st.text_input("Event name", key="run_event_wf_text")

        run_weights_a  = st.text_input("Weights path", value=str(BEST_MODEL_PATH), key="run_weights_a")
        run_ckpt_a     = st.text_input("Checkpoint dir", value=str(PROJECT_ROOT / "checkpoints_compare"), key="run_ckpt_a")
        run_thresh_a   = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="run_thresh_a")
        run_patch_a    = st.select_slider("Patch size", [128, 256, 512], value=256, key="run_patch_a")
        _SCRIPT_MODELS = ["unet", "deeplab", "fcsiam", "segformer", "rf"]
        run_models_a   = st.multiselect("Models to run", _SCRIPT_MODELS, default=_SCRIPT_MODELS,
                                         format_func=lambda k: MODEL_LABELS.get(k, k), key="run_models_a")
        st.caption("CA-LMoETransUNet always runs automatically via best_model.pth.")

        if st.button("Run all 6 models", type="primary", use_container_width=True, key="btn_run_a"):
            ev = run_event_a if isinstance(run_event_a, str) else ""
            cmd = build_predict_cmd(
                dataset="canada" if is_canada_run else "wildfire",
                event=ev,
                weights_path=run_weights_a,
                ckpt_dir=run_ckpt_a,
                results_dir=results_dir_str,
                threshold=run_thresh_a,
                patch_size=run_patch_a,
                overlap=32,
                device=device_str,
                models_filter=run_models_a,
            )
            st.code(" ".join(cmd))
            # Persistent log: stored in session_state so it survives reruns
            st.session_state["run_log_a"] = ""
            log_status_a = st.empty()
            pbar = st.progress(0)
            done_c = 0
            n_m = len(run_models_a) or 6
            for line in run_predict_streaming(cmd):
                st.session_state["run_log_a"] += line
                log_status_a.markdown(
                    f'<div class="log-box">{st.session_state["run_log_a"][-4000:]}</div>',
                    unsafe_allow_html=True
                )
                if "Dice=" in line:
                    done_c += 1
                    pbar.progress(min(done_c / n_m, 1.0))
            pbar.progress(1.0)
            log_txt = st.session_state["run_log_a"]
            if "EXIT CODE" in log_txt and "EXIT CODE 0" not in log_txt:
                hint = get_error_hint(log_txt)
                st.error("Run failed.")
                if hint:
                    st.warning(f"Fix hint: {hint}")
            else:
                st.success("✅ Run complete!")
                st.session_state["last_run_event_upload"] = ev
                st.cache_data.clear()
                st.toast("Results ready — see preview below!", icon="✅")
                st.rerun()

        # Always show last Card A log (persists after run)
        if st.session_state.get("run_log_a"):
            with st.expander("📋 Last run log (Card A)", expanded=False):
                st.markdown(
                    f'<div class="log-box">{st.session_state["run_log_a"][-6000:]}</div>',
                    unsafe_allow_html=True
                )

    # ── Card B: Upload outside data ────────────────────────────────────────────
    with card_b:
        st.markdown("### Card B — Upload Outside Data")
        st.markdown('<div class="fire-divider"></div>', unsafe_allow_html=True)

        upload_method = st.radio("Input method", ["Upload files", "Folder path on disk"], key="upload_method")

        if upload_method == "Upload files":
            st.caption("Upload 4 files: s1_pre, s1_post, s2_pre, s2_post (+ optional mask). .npy or .tif")
            up_files = st.file_uploader("Upload satellite files",
                                         accept_multiple_files=True,
                                         type=["npy", "tif", "tiff"],
                                         key="up_files")

            if up_files:
                # Match files by keywords
                KEYWORDS = {
                    "s1_pre":  ["s1_pre", "sentinel1_pre", "s1_before", "s1pre"],
                    "s1_post": ["s1_post", "sentinel1_post", "s1_after", "s1post"],
                    "s2_pre":  ["s2_pre", "sentinel2_pre", "s2_before", "s2pre"],
                    "s2_post": ["s2_post", "sentinel2_post", "s2_after", "s2post"],
                    "mask":    ["mask", "gt", "ground_truth", "label"],
                }
                match: dict[str, str] = {}
                for f in up_files:
                    stem = f.name.lower().replace("-", "_").replace(" ", "_")
                    for role, kws in KEYWORDS.items():
                        if any(kw in stem for kw in kws):
                            match[role] = f.name
                            break

                # Show match table with dropdowns to fix
                st.markdown("**File matching:**")
                fname_map = {f.name: f for f in up_files}
                file_names = [f.name for f in up_files]
                for role in ["s1_pre", "s1_post", "s2_pre", "s2_post", "mask"]:
                    current = match.get(role, "— none —")
                    options = ["— none —"] + file_names
                    idx = options.index(current) if current in options else 0
                    chosen = st.selectbox(f"{role}", options, index=idx, key=f"match_{role}")
                    if chosen != "— none —":
                        match[role] = chosen

                event_name_b = st.text_input("Event name", value="custom_upload", key="event_name_b")

                if st.button("Save & Run all 6 models", type="primary", use_container_width=True, key="btn_run_b"):
                    # Save files to a temp input folder
                    input_dir = UPLOADS_DIR / "input" / event_name_b
                    input_dir.mkdir(parents=True, exist_ok=True)

                    for role, fname in match.items():
                        if fname in fname_map:
                            ext = Path(fname).suffix
                            dest_name = f"{role}{ext}"
                            (input_dir / dest_name).write_bytes(fname_map[fname].read())

                    # Run predict_compare with --input_dir
                    cmd_b = [
                        sys.executable,
                        str(PROJECT_ROOT / "predict_compare.py"),
                        "--input_dir", str(input_dir),
                        "--weights",   str(BEST_MODEL_PATH),
                        "--out-dir",   results_dir_str,
                        "--threshold", "0.5",
                        "--patch",     "256",
                        "--overlap",   "32",
                    ]
                    st.code(" ".join(cmd_b))
                    # Persistent log B
                    st.session_state["run_log_b"] = ""
                    log_status_b = st.empty()
                    for line in run_predict_streaming(cmd_b):
                        st.session_state["run_log_b"] += line
                        log_status_b.markdown(
                            f'<div class="log-box">{st.session_state["run_log_b"][-4000:]}</div>',
                            unsafe_allow_html=True
                        )
                    log_txt_b = st.session_state["run_log_b"]
                    if "EXIT CODE" in log_txt_b and "EXIT CODE 0" not in log_txt_b:
                        hint = get_error_hint(log_txt_b)
                        st.error("Run failed.")
                        with st.expander("Full traceback"):
                            st.text(log_txt_b)
                        if hint:
                            st.warning(f"Fix hint: {hint}")
                    else:
                        st.success("✅ Run complete!")
                        st.session_state["last_run_event_upload"] = event_name_b
                        st.cache_data.clear()
                        st.toast("Results ready — see preview below!", icon="✅")
                        st.rerun()

            # Always show last Card B log
            if st.session_state.get("run_log_b"):
                with st.expander("📋 Last run log (Card B)", expanded=False):
                    st.markdown(
                        f'<div class="log-box">{st.session_state["run_log_b"][-6000:]}</div>',
                        unsafe_allow_html=True
                    )
        else:
            folder_path = st.text_input("Folder path (on disk, large data)", key="folder_path")
            event_name_folder = st.text_input("Event name", value=Path(folder_path).name if folder_path else "folder_event", key="event_folder_name")

            if folder_path and Path(folder_path).is_dir():
                contents = list(Path(folder_path).iterdir())
                st.markdown(f"Found {len(contents)} files in folder:")
                st.code("\n".join(f.name for f in contents[:20]))

            if st.button("Run on this folder", type="primary", use_container_width=True, key="btn_run_folder"):
                if not folder_path:
                    st.error("Enter a folder path.")
                else:
                    cmd_f = [
                        sys.executable,
                        str(PROJECT_ROOT / "predict_compare.py"),
                        "--input_dir", folder_path,
                        "--weights",   str(BEST_MODEL_PATH),
                        "--out-dir",   results_dir_str,
                    ]
                    # Persistent log F
                    st.session_state["run_log_f"] = ""
                    log_status_f = st.empty()
                    for line in run_predict_streaming(cmd_f):
                        st.session_state["run_log_f"] += line
                        log_status_f.markdown(
                            f'<div class="log-box">{st.session_state["run_log_f"][-4000:]}</div>',
                            unsafe_allow_html=True
                        )
                    log_txt_f = st.session_state["run_log_f"]
                    if "EXIT CODE 0" in log_txt_f or "EXIT CODE" not in log_txt_f:
                        st.success("✅ Run complete!")
                        st.session_state["last_run_event_upload"] = event_name_folder
                        st.cache_data.clear()
                        st.toast("Results ready — see preview below!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Run failed.")
                        with st.expander("Full output"):
                            st.text(log_txt_f)

            # Always show last folder log
            if st.session_state.get("run_log_f"):
                with st.expander("📋 Last run log (Folder)", expanded=False):
                    st.markdown(
                        f'<div class="log-box">{st.session_state["run_log_f"][-6000:]}</div>',
                        unsafe_allow_html=True
                    )

    # ── Inline preview rendered below both cards ──────────────────────────────
    if _last_run_event:
        _inline_results_preview(_last_run_event)
