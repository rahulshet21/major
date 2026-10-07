"""
viz.py — Plotly builders: overlay, heatmap, error map, radar, bars, PR/ROC.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import MODEL_COLORS, MODEL_LABELS, MODELS

_PLOTLY_TEMPLATE = "plotly_dark"
_PAPER_BG = "rgba(0,0,0,0)"
_PLOT_BG  = "rgba(20,22,34,0.8)"


def _base_layout(**kwargs) -> dict:
    return dict(
        template=_PLOTLY_TEMPLATE,
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(family="Inter, system-ui, sans-serif", color="#e8eaf0"),
        margin=dict(l=10, r=10, t=40, b=10),
        **kwargs,
    )


# ── Probability heatmap ────────────────────────────────────────────────────────
def prob_heatmap(prob: np.ndarray, title: str = "Probability Heatmap") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=prob,
        colorscale="Inferno",
        zmin=0, zmax=1,          # always: 0=black, 1=bright-yellow
        colorbar=dict(
            title="Prob",
            tickfont=dict(color="#e8eaf0"),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0", "0.25", "0.5", "0.75", "1"],
        ),
    ))
    fig.update_layout(**_base_layout(title=title))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    return fig


# ── RGB overlay ───────────────────────────────────────────────────────────────
def rgb_overlay(rgb_hwc: np.ndarray, mask: np.ndarray,
                opacity: float = 0.5,
                title: str = "Prediction Overlay") -> go.Figure:
    """RGB image with burned mask overlaid in orange."""
    fig = px.imshow(rgb_hwc, title=title)
    # Add mask overlay
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask == 1] = [255, 107, 53, int(opacity * 255)]  # fire-orange
    fig.add_trace(go.Image(z=overlay, opacity=opacity))
    fig.update_layout(**_base_layout(title=title))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ── Error map ─────────────────────────────────────────────────────────────────
def error_map_fig(rgba: np.ndarray, title: str = "Error Map") -> go.Figure:
    fig = px.imshow(rgba, title=title)
    # Manual legend items
    for label, color in [("TP", "#32c850"), ("FP", "#dc3232"), ("FN", "#3250dc"), ("TN", "#646464")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=label,
        ))
    fig.update_layout(**_base_layout(title=title), showlegend=True,
                      legend=dict(orientation="h", y=-0.05))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


# ── Ground truth ──────────────────────────────────────────────────────────────
def gt_fig(gt: np.ndarray, title: str = "Ground Truth") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=gt.astype(np.float32),
        colorscale=[[0, "#1a1d26"], [1, "#ff6b35"]],
        zmin=0, zmax=1,
        showscale=False,
    ))
    fig.update_layout(**_base_layout(title=title))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    return fig


# ── 2×2 model panel ───────────────────────────────────────────────────────────
def model_2x2_panel(rgb_pre: np.ndarray, prob: np.ndarray,
                    mask: np.ndarray, gt: Optional[np.ndarray],
                    rgba_err: Optional[np.ndarray],
                    opacity: float = 0.5, model_label: str = "") -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["S2 Pre + Overlay", "Probability", "Error Map", "Ground Truth"],
        shared_xaxes=True, shared_yaxes=True,
    )

    # (a) RGB + overlay
    if rgb_pre is not None:
        fig.add_trace(go.Image(z=rgb_pre), row=1, col=1)
    ovl = np.zeros((*mask.shape, 4), dtype=np.uint8)
    ovl[mask == 1] = [255, 107, 53, int(opacity * 255)]
    fig.add_trace(go.Image(z=ovl, opacity=opacity), row=1, col=1)

    # (b) Probability
    fig.add_trace(go.Heatmap(z=prob, colorscale="Inferno", zmin=0, zmax=1,
                              showscale=True, colorbar=dict(x=1.02, len=0.5, y=0.75)), row=1, col=2)

    # (c) Error map
    if rgba_err is not None:
        fig.add_trace(go.Image(z=rgba_err), row=2, col=1)
    else:
        fig.add_trace(go.Heatmap(z=np.zeros_like(mask, dtype=np.float32),
                                  colorscale="gray", showscale=False), row=2, col=1)

    # (d) GT
    if gt is not None:
        fig.add_trace(go.Heatmap(z=gt.astype(np.float32),
                                  colorscale=[[0, "#1a1d26"], [1, "#ff6b35"]],
                                  zmin=0, zmax=1, showscale=False), row=2, col=2)
    else:
        fig.add_trace(go.Heatmap(z=np.zeros_like(mask, dtype=np.float32),
                                  colorscale="gray", showscale=False), row=2, col=2)

    fig.update_layout(**_base_layout(title=f"{model_label} — Detailed View"), height=700)
    fig.update_yaxes(autorange="reversed")
    return fig


# ── Radar chart ───────────────────────────────────────────────────────────────
def radar_chart(metrics_df, title: str = "Model Comparison — Radar") -> go.Figure:
    cats = ["Dice", "IoU", "Precision", "Recall"]
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        key = str(row.get("model", "")).lower()
        color = MODEL_COLORS.get(key, "#aaaaaa")
        label = MODEL_LABELS.get(key, key)
        vals = [
            float(row.get("dice", 0) or 0),
            float(row.get("iou", 0) or 0),
            float(row.get("precision", 0) or 0),
            float(row.get("recall", 0) or 0),
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself", name=label,
            line=dict(color=color, width=2),
            fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if color.startswith("rgb") else color + "20",
        ))
    fig.update_layout(**_base_layout(title=title),
                      polar=dict(bgcolor=_PLOT_BG,
                                 radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#888")),
                                 angularaxis=dict(tickfont=dict(color="#e8eaf0"))))
    return fig


# ── Grouped bar chart ─────────────────────────────────────────────────────────
def grouped_bar(metrics_df, title: str = "Model Metrics Comparison") -> go.Figure:
    metrics = ["dice", "iou", "precision", "recall"]
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        key = str(row.get("model", "")).lower()
        color = MODEL_COLORS.get(key, "#aaaaaa")
        label = MODEL_LABELS.get(key, key)
        fig.add_trace(go.Bar(
            name=label,
            x=[m.capitalize() for m in metrics],
            y=[float(row.get(m, 0) or 0) for m in metrics],
            marker_color=color,
        ))
    fig.update_layout(**_base_layout(title=title), barmode="group",
                      yaxis=dict(range=[0, 1.05]))
    return fig


# ── Bubble chart (params vs Dice) ─────────────────────────────────────────────
def bubble_chart(metrics_df, title: str = "Params vs Dice (bubble = inference time)") -> go.Figure:
    fig = go.Figure()
    for _, row in metrics_df.iterrows():
        key = str(row.get("model", "")).lower()
        color = MODEL_COLORS.get(key, "#aaaaaa")
        label = MODEL_LABELS.get(key, key)
        params = row.get("params", 0)
        try:
            params = float(params) if params != "N/A" else 0
        except Exception:
            params = 0
        infer_s = float(row.get("inference_seconds", 1) or 1)
        dice = float(row.get("dice", 0) or 0)
        fig.add_trace(go.Scatter(
            x=[params / 1e6],
            y=[dice],
            mode="markers+text",
            text=[label],
            textposition="top center",
            marker=dict(size=max(10, infer_s * 8), color=color, opacity=0.85,
                        line=dict(width=1, color="white")),
            name=label,
        ))
    fig.update_layout(**_base_layout(title=title),
                      xaxis=dict(title="Parameters (M)"),
                      yaxis=dict(title="Dice Score", range=[0, 1.05]))
    return fig


# ── PR Curve ──────────────────────────────────────────────────────────────────
def pr_curve_fig(thresholds, precisions, recalls, model_label: str = "",
                 color: str = "#ff6b35") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines",
                              line=dict(color=color, width=2),
                              name=model_label,
                              hovertemplate="Recall=%{x:.3f}<br>Precision=%{y:.3f}"))
    fig.update_layout(**_base_layout(title=f"PR Curve — {model_label}"),
                      xaxis=dict(title="Recall", range=[0, 1]),
                      yaxis=dict(title="Precision", range=[0, 1.05]))
    return fig


# ── ROC Curve ─────────────────────────────────────────────────────────────────
def roc_curve_fig(fprs, recalls, model_label: str = "",
                  color: str = "#ff6b35") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(color="#555", dash="dash"), name="Random"))
    fig.add_trace(go.Scatter(x=fprs, y=recalls, mode="lines",
                              line=dict(color=color, width=2),
                              name=model_label,
                              hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}"))
    fig.update_layout(**_base_layout(title=f"ROC Curve — {model_label}"),
                      xaxis=dict(title="FPR", range=[0, 1]),
                      yaxis=dict(title="TPR (Recall)", range=[0, 1.05]))
    return fig


# ── Dice vs Threshold (all models) ────────────────────────────────────────────
def dice_threshold_chart(curves_dict: dict, title: str = "Dice vs Threshold") -> go.Figure:
    """curves_dict: {model_key: (thresholds, dices)}"""
    fig = go.Figure()
    for key, (thresholds, dices) in curves_dict.items():
        color = MODEL_COLORS.get(key, "#aaaaaa")
        label = MODEL_LABELS.get(key, key)
        fig.add_trace(go.Scatter(x=thresholds, y=dices, mode="lines",
                                  line=dict(color=color, width=2), name=label))
    fig.update_layout(**_base_layout(title=title),
                      xaxis=dict(title="Threshold"),
                      yaxis=dict(title="Dice Score", range=[0, 1.05]))
    return fig


# ── Agreement map ─────────────────────────────────────────────────────────────
def agreement_map_fig(agr: np.ndarray, n_models: int, title: str = "Model Agreement") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=agr,
        colorscale="YlOrRd",
        zmin=0, zmax=n_models,
        colorbar=dict(title=f"# models (/{n_models})", tickfont=dict(color="#e8eaf0")),
    ))
    fig.update_layout(**_base_layout(title=title))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    return fig


# ── Probability histogram ─────────────────────────────────────────────────────
def prob_histogram(prob: np.ndarray, gt: Optional[np.ndarray] = None,
                   title: str = "Probability Distribution") -> go.Figure:
    fig = go.Figure()
    flat = prob.ravel()
    if gt is not None:
        fg = flat[gt.ravel() == 1]
        bg = flat[gt.ravel() == 0]
        fig.add_trace(go.Histogram(x=fg, name="Inside GT (burned)",
                                    marker_color="#ff6b35", opacity=0.7,
                                    nbinsx=50, histnorm="probability"))
        fig.add_trace(go.Histogram(x=bg, name="Outside GT (unburned)",
                                    marker_color="#4fc3f7", opacity=0.7,
                                    nbinsx=50, histnorm="probability"))
    else:
        fig.add_trace(go.Histogram(x=flat, nbinsx=50, marker_color="#ff6b35",
                                    histnorm="probability"))
    fig.update_layout(**_base_layout(title=title), barmode="overlay",
                      xaxis=dict(title="Probability"), yaxis=dict(title="Density"))
    return fig
