#!/usr/bin/env python3
"""
DeformCT Viewer — interactive Dash app for comparing original and deformed CTs.

Usage:
    # After running generate_ensemble, point at both dirs:
    python -m viewer.app --orig-ct exampledata/ct_063 --ensemble output/ensemble

    # Original CT only (no comparison):
    python -m viewer.app --orig-ct exampledata/ct_063

Then open http://127.0.0.1:8050 in your browser.

Features:
  • Side-by-side: Original | Deformed | Difference (ΔHU)
  • Axial / Coronal / Sagittal orientation
  • Per-state selector populated from manifest.json
  • Window / Level / Gamma sliders
  • Difference colour range slider (±HU)
  • HU value probe on hover (click any panel)
  • State provenance info (tx, ty, tz, rx, ry, rz, cluster weight)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pydicom

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ─── Colour scales ────────────────────────────────────────────────────────────
GRAY = [[0.0, "rgb(0,0,0)"], [1.0, "rgb(255,255,255)"]]
DIFF_CSCALE = "RdBu_r"   # red = positive HU, blue = negative HU

# ─── Defaults ─────────────────────────────────────────────────────────────────
_DEF_WINDOW = 400
_DEF_LEVEL  = 40
_DEF_GAMMA  = 1.0
_DEF_DIFF   = 100   # ± HU


# ─── Module-level volume cache (single-user tool — safe) ──────────────────────
_orig: np.ndarray | None = None          # (nz, ny, nx) int16
_deformed: dict[str, np.ndarray] = {}   # ct_dir key → (nz, ny, nx) int16
_manifest_states: list[dict] = []
_orig_spacing:   tuple[float, float, float] = (1.0, 1.0, 1.0)  # mm (sz, sy, sx)
_orig_origin:    tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm (oz, oy, ox)
_orig_direction: np.ndarray = np.eye(3)                         # (3,3) row-major

# RTSTRUCT contours: list of {"name", "color", "contours": [(N,3) x,y,z mm]}
_orig_rtstruct_rois:     list[dict] = []
_deformed_rtstruct_rois: dict[str, list[dict]] = {}


# ─── Image utilities ──────────────────────────────────────────────────────────

def _window_level(arr: np.ndarray, w: float, l: float, gamma: float) -> np.ndarray:
    """HU → [0, 1] with W/L windowing and optional gamma."""
    lo = l - w / 2.0
    hi = l + w / 2.0
    x = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return np.power(x, gamma) if gamma != 1.0 else x


def _get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract a 2-D slice; idx is clamped to valid range."""
    idx = int(np.clip(idx, 0, vol.shape[axis] - 1))
    if axis == 0:   # axial:   (ny, nx) — flip y so anterior is up
        return vol[idx, ::-1, :]
    elif axis == 1:  # coronal: (nz, nx)  — flip z so superior is up
        return vol[::-1, idx, :]
    else:            # sagittal:(nz, ny)  — flip z so superior is up
        return vol[::-1, :, idx]


def _make_fig(
    z2d: np.ndarray,
    colorscale,
    zmin: float,
    zmax: float,
    title: str,
    row_spacing_mm: float = 1.0,
    col_spacing_mm: float = 1.0,
) -> go.Figure:
    """Build a Heatmap figure for one CT panel.

    row_spacing_mm / col_spacing_mm set the physical pixel size so the
    displayed image has correct anatomical proportions.
    """
    # scaleratio = how many x-units equal one y-unit in physical space
    scaleratio = row_spacing_mm / col_spacing_mm
    fig = go.Figure(go.Heatmap(
        z=z2d,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        showscale=(colorscale != GRAY),
        zsmooth=False,
        hovertemplate="x=%{x}  y=%{y}  val=%{z:.1f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=9)),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=11, color="#aaa"), x=0.5, xanchor="center"),
        margin=dict(l=2, r=2, t=28, b=2),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        xaxis=dict(visible=False, constrain="domain"),
        yaxis=dict(
            visible=False,
            scaleanchor="x",
            scaleratio=scaleratio,   # 1 y-px = scaleratio x-px in physical mm
            constrain="domain",
        ),
    )
    return fig


def _blank_fig(title: str) -> go.Figure:
    return _make_fig(np.zeros((4, 4)), GRAY, 0.0, 1.0, title)


# ─── RTSTRUCT overlay ─────────────────────────────────────────────────────────

def _load_rtstruct(path: str | Path) -> list[dict]:
    """Parse an RTSTRUCT DICOM → list of ROI dicts.

    Each dict: ``name`` (str), ``color`` ("rgb(r,g,b)"),
    ``contours`` (list of (N,3) float64, each row (x,y,z) LPS mm).
    """
    ds = pydicom.dcmread(str(path))
    roi_names: dict[int, str] = {}
    for roi in getattr(ds, "StructureSetROISequence", []):
        roi_names[int(roi.ROINumber)] = getattr(roi, "ROIName", f"ROI-{roi.ROINumber}")

    _palette = [
        "rgb(255,80,80)",  "rgb(80,220,80)",  "rgb(80,140,255)",
        "rgb(255,220,60)", "rgb(220,80,255)", "rgb(60,220,220)",
        "rgb(255,160,60)", "rgb(160,255,60)", "rgb(255,80,180)",
        "rgb(80,200,255)",
    ]
    rois: list[dict] = []
    for i, roi_c in enumerate(getattr(ds, "ROIContourSequence", [])):
        roi_num = int(getattr(roi_c, "ReferencedROINumber", i))
        name = roi_names.get(roi_num, f"ROI-{roi_num}")
        if hasattr(roi_c, "ROIDisplayColor"):
            r, g, b = (int(float(c)) for c in roi_c.ROIDisplayColor)
            color = f"rgb({r},{g},{b})"
        else:
            color = _palette[i % len(_palette)]
        contours: list[np.ndarray] = []
        for contour in getattr(roi_c, "ContourSequence", []):
            flat = list(getattr(contour, "ContourData", []))
            if len(flat) >= 9 and len(flat) % 3 == 0:
                contours.append(np.array(flat, dtype=np.float64).reshape(-1, 3))
        if contours:
            rois.append({"name": name, "color": color, "contours": contours})
    return rois


def _get_contour_traces(
    rois: list[dict],
    axis: int,
    slice_idx: int,
    visible_names: set[str] | None,
) -> list[go.Scatter]:
    """Scatter traces for RTSTRUCT contours at the current slice plane.

    Uses the module-level CT geometry (_orig_origin, _orig_spacing, _orig_direction).
    Contour points are DICOM LPS (x, y, z) mm.

    Heatmap pixel conventions (matching _get_slice flips):
      axis=0  col=ix,  row=ny-1-iy
      axis=1  col=ix,  row=nz-1-iz
      axis=2  col=iy,  row=nz-1-iz
    """
    if not rois or _orig is None:
        return []
    nz, ny, nx = _orig.shape
    spacing_arr = np.array(_orig_spacing,   dtype=np.float64)  # (sz, sy, sx)
    origin_xyz  = np.array(_orig_origin[::-1], dtype=np.float64)  # project(z,y,x)→LPS(x,y,z)
    # Correct inverse: vox(iz,iy,ix) = diag(1/spacing) @ direction @ (p_xyz - origin_xyz)
    # direction rows are unit vectors in LPS(x,y,z); dot with rel gives iz*sz, iy*sy, ix*sx
    M_inv = np.diag(1.0 / spacing_arr) @ _orig_direction    # (3,3)

    traces: list[go.Scatter] = []
    for roi in rois:
        if visible_names is not None and roi["name"] not in visible_names:
            continue
        xs_all: list = []
        ys_all: list = []
        for pts_xyz in roi["contours"]:          # (N,3) LPS (x,y,z)
            rel  = pts_xyz - origin_xyz          # (N,3) in LPS (x,y,z)
            vox  = (M_inv @ rel.T).T             # (N,3) fractional (iz,iy,ix)

            if axis == 0:                        # ── axial ──────────────────
                if abs(float(np.mean(vox[:, 0])) - slice_idx) > 0.6:
                    continue
                cols = vox[:, 2]
                rows = ny - 1 - vox[:, 1]
                xs_all.extend(np.append(cols, cols[0]).tolist() + [None])
                ys_all.extend(np.append(rows, rows[0]).tolist() + [None])

            else:                                # ── coronal / sagittal ─────
                cut_ax = 1 if axis == 1 else 2   # iy or ix
                coord  = vox[:, cut_ax]
                n_pts  = len(vox)
                int_pts: list[np.ndarray] = []
                for k in range(n_pts):
                    l = (k + 1) % n_pts
                    c0, c1 = coord[k], coord[l]
                    if (c0 - slice_idx) * (c1 - slice_idx) <= 0 and c0 != c1:
                        t = (slice_idx - c0) / (c1 - c0)
                        int_pts.append(vox[k] + t * (vox[l] - vox[k]))
                if len(int_pts) < 2:
                    continue
                arr_i  = np.array(int_pts)
                rows_i = nz - 1 - arr_i[:, 0]
                cols_i = arr_i[:, 2] if axis == 1 else arr_i[:, 1]
                for k in range(0, len(int_pts) - 1, 2):
                    xs_all.extend([float(cols_i[k]), float(cols_i[k + 1]), None])
                    ys_all.extend([float(rows_i[k]), float(rows_i[k + 1]), None])

        if xs_all:
            traces.append(go.Scatter(
                x=xs_all, y=ys_all, mode="lines",
                line=dict(color=roi["color"], width=1.5),
                name=roi["name"], showlegend=False, hoverinfo="name",
            ))
    return traces


def _add_contours(
    fig: go.Figure,
    rois: list[dict],
    axis: int,
    slice_idx: int,
    visible_names: set[str] | None,
) -> go.Figure:
    """Add contour scatter traces and lock axis ranges to the image bounds."""
    for tr in _get_contour_traces(rois, axis, slice_idx, visible_names):
        fig.add_trace(tr)
    if _orig is not None:
        nz, ny, nx = _orig.shape
        if axis == 0:
            fig.update_xaxes(range=[-0.5, nx - 0.5])
            fig.update_yaxes(range=[-0.5, ny - 0.5])
        elif axis == 1:
            fig.update_xaxes(range=[-0.5, nx - 0.5])
            fig.update_yaxes(range=[-0.5, nz - 0.5])
        else:
            fig.update_xaxes(range=[-0.5, ny - 0.5])
            fig.update_yaxes(range=[-0.5, nz - 0.5])
    return fig


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_orig_ct(ct_dir: str) -> None:
    global _orig, _orig_spacing, _orig_origin, _orig_direction
    if _orig is not None:
        return
    try:
        from gendosecalc.deform.dicom_export import load_ct_series
        arr, spacing_mm, origin_mm, direction, *_ = load_ct_series(ct_dir)
        _orig          = arr.astype(np.int16)
        _orig_spacing  = tuple(float(s) for s in spacing_mm)   # (sz, sy, sx)
        _orig_origin   = tuple(float(o) for o in origin_mm)    # (oz, oy, ox)
        _orig_direction = np.array(direction, dtype=np.float64)
        logger.info("Loaded original CT: %s  shape=%s", ct_dir, _orig.shape)
    except Exception as exc:
        logger.error("Failed to load original CT from %s: %s", ct_dir, exc)
        raise


def _load_ensemble(ensemble_dir: str) -> None:
    p = Path(ensemble_dir)
    # Accept either a directory or a direct path to manifest.json
    if p.is_file() and p.suffix == ".json":
        manifest_p = p
        ensemble_dir = str(p.parent)
    else:
        manifest_p = p / "manifest.json"
    if not manifest_p.exists():
        logger.warning("No manifest.json found in %s", ensemble_dir)
        return
    with open(manifest_p) as f:
        data = json.load(f)
    states = data.get("states", [])

    try:
        from gendosecalc.deform.dicom_export import load_ct_series
    except ImportError as exc:
        logger.error("Cannot import load_ct_series: %s", exc)
        return

    for s in states:
        key = s.get("ct_dir", "")
        ct_path = Path(ensemble_dir) / key
        if not ct_path.exists():
            logger.warning("State CT dir missing: %s — skipping", ct_path)
            continue
        try:
            arr, *_ = load_ct_series(str(ct_path))
            _deformed[key] = arr.astype(np.int16)
            _manifest_states.append(s)
            logger.info("Loaded state %s: %s", key, arr.shape)
        except Exception as exc:
            logger.warning("Could not load state %s: %s", key, exc)
            continue

        # Load deformed RTSTRUCT if recorded in manifest
        rs_rel = s.get("rtstruct_path", "")
        if rs_rel:
            rs_path = Path(ensemble_dir) / rs_rel
            if rs_path.exists():
                try:
                    _deformed_rtstruct_rois[key] = _load_rtstruct(rs_path)
                    logger.info("Loaded deformed RTSTRUCT for state %s", key)
                except Exception as exc:
                    logger.warning("Could not load deformed RS %s: %s", rs_path, exc)

    logger.info(
        "Ensemble: %d states loaded  (%d with deformed RTSTRUCT)",
        len(_manifest_states), len(_deformed_rtstruct_rois),
    )


# ─── App factory ──────────────────────────────────────────────────────────────

def build_app(
    orig_ct: str,
    ensemble_dir: str | None,
    orig_rtstruct: str | None = None,
) -> dash.Dash:
    global _orig_rtstruct_rois
    # Load data up front so the app starts ready
    _load_orig_ct(orig_ct)
    if ensemble_dir:
        _load_ensemble(ensemble_dir)
    if orig_rtstruct:
        try:
            _orig_rtstruct_rois = _load_rtstruct(orig_rtstruct)
            logger.info("Loaded original RTSTRUCT: %d ROIs", len(_orig_rtstruct_rois))
        except Exception as exc:
            logger.error("Failed to load RTSTRUCT %s: %s", orig_rtstruct, exc)

    nz, ny, nx = _orig.shape

    # ── State dropdown options ────────────────────────────────────────────────
    state_opts = []
    for s in _manifest_states:
        tx = s.get("tx", 0.0); ty = s.get("ty", 0.0); tz = s.get("tz", 0.0)
        label = (
            f"State {s['state_index']:03d} · "
            f"({tx:+.1f}, {ty:+.1f}, {tz:+.1f}) mm"
            f"  ×{s.get('cluster_weight', '?')}"
        )
        state_opts.append({"label": label, "value": s["ct_dir"]})

    default_state = state_opts[0]["value"] if state_opts else "__none__"

    # ── Collect all unique ROI names (orig + all deformed) ─────────────────────
    _seen_rois: set[str] = set()
    _all_rois: list[dict] = []          # unique ROI dicts (name + color)
    for _r in _orig_rtstruct_rois:
        if _r["name"] not in _seen_rois:
            _all_rois.append(_r); _seen_rois.add(_r["name"])
    for _rois in _deformed_rtstruct_rois.values():
        for _r in _rois:
            if _r["name"] not in _seen_rois:
                _all_rois.append(_r); _seen_rois.add(_r["name"])
    has_rtstruct = bool(_all_rois)
    roi_opts = [{"label": r["name"], "value": r["name"]} for r in _all_rois]

    # ── Preset W/L combinations ───────────────────────────────────────────────
    wl_presets = [
        {"label": "Soft tissue  W400/L40",  "value": "400,40"},
        {"label": "Lung         W1500/L-600", "value": "1500,-600"},
        {"label": "Bone         W1500/L400", "value": "1500,400"},
        {"label": "Abdomen      W350/L40",   "value": "350,40"},
        {"label": "Brain        W80/L40",    "value": "80,40"},
    ]

    # ── Controls panel ───────────────────────────────────────────────────────
    _sl = {"marginBottom": "6px"}

    controls = dbc.Card([dbc.CardBody([
        # State selector
        html.Label("Deformed state", className="text-info small mb-1"),
        dcc.Dropdown(
            id="state-dd",
            options=state_opts,
            value=default_state,
            clearable=False,
            placeholder="No ensemble loaded" if not state_opts else None,
            style={"fontSize": "12px", "color": "#000"},
        ),
        html.Hr(className="my-2"),

        # Orientation
        html.Label("Orientation", className="text-info small mb-1"),
        dbc.RadioItems(
            id="orient",
            options=[
                {"label": "Axial",    "value": 0},
                {"label": "Coronal",  "value": 1},
                {"label": "Sagittal", "value": 2},
            ],
            value=0,
            inline=True,
            className="small",
        ),
        html.Hr(className="my-2"),

        # W/L presets
        html.Label("W/L preset", className="text-info small mb-1"),
        dcc.Dropdown(
            id="wl-preset",
            options=wl_presets,
            value=None,
            placeholder="Custom…",
            clearable=True,
            style={"fontSize": "12px", "color": "#000"},
        ),
        html.Div(style={"height": "6px"}),

        html.Label("Window", className="text-muted small"),
        dcc.Slider(id="w-sl", min=1, max=4000, step=1, value=_DEF_WINDOW,
                   marks={1: "1", 400: "400", 2000: "2k", 4000: "4k"},
                   tooltip={"always_visible": False, "placement": "right"},
                   className="mb-1"),
        html.Label("Level", className="text-muted small"),
        dcc.Slider(id="l-sl", min=-1024, max=3071, step=1, value=_DEF_LEVEL,
                   marks={-1024: "-1k", 0: "0", 1000: "1k", 3071: "3k"},
                   tooltip={"always_visible": False, "placement": "right"},
                   className="mb-1"),
        html.Label("Gamma", className="text-muted small"),
        dcc.Slider(id="g-sl", min=0.1, max=3.0, step=0.05, value=_DEF_GAMMA,
                   marks={0.1: "0.1", 1.0: "1.0", 2.0: "2.0", 3.0: "3.0"},
                   tooltip={"always_visible": False, "placement": "right"},
                   className="mb-1"),

        html.Hr(className="my-2"),
        html.Label("Diff range ±HU", className="text-info small mb-1"),
        dcc.Slider(id="dr-sl", min=10, max=1000, step=10, value=_DEF_DIFF,
                   marks={10: "10", 100: "100", 500: "500", 1000: "1k"},
                   tooltip={"always_visible": False, "placement": "right"},
                   className="mb-1"),

        html.Hr(className="my-2"),
        html.Label("Show overlay", className="text-info small mb-1"),
        dbc.Checklist(
            id="overlay-toggle",
            options=[{"label": "Diff overlay on Deformed", "value": "on"}],
            value=[],
            className="small",
        ),

        html.Hr(className="my-2"),
        html.Label("ROI contours", className="text-info small mb-1"),
        dbc.Checklist(
            id="roi-show",
            options=[{"label": "Show contours", "value": "on"}],
            value=["on"] if has_rtstruct else [],
            className="small mb-1",
            style={} if has_rtstruct else {"opacity": "0.4"},
        ),
        dcc.Dropdown(
            id="roi-names",
            options=roi_opts,
            value=None,
            multi=True,
            placeholder="All ROIs…" if has_rtstruct else "No RTSTRUCT loaded",
            disabled=not has_rtstruct,
            style={"fontSize": "11px", "color": "#000"},
        ),
        # colour swatches
        html.Div([
            html.Div([
                html.Span("■ ", style={"color": r["color"],
                                        "fontSize": "13px",
                                        "fontFamily": "monospace"}),
                html.Span(r["name"], style={"fontSize": "10px"}),
            ], className="mt-1")
            for r in _all_rois
        ], className="mt-1"),

        html.Hr(className="my-2"),
        html.Div(id="info-box",
                 className="small text-muted",
                 style={"whiteSpace": "pre-line", "fontFamily": "monospace",
                        "fontSize": "11px"}),
    ])], style={"height": "100%", "overflowY": "auto"})

    # ── App ──────────────────────────────────────────────────────────────────
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="DeformCT Viewer",
    )

    header = dbc.Row([
        dbc.Col(html.H5("DeformCT Viewer", className="text-primary mb-0"), width="auto"),
        dbc.Col(
            html.Small(
                f"{Path(orig_ct).name}  ·  {nz}×{ny}×{nx} vox"
                + (f"  ·  {_orig_spacing[2]:.2f}×{_orig_spacing[1]:.2f}×{_orig_spacing[0]:.2f} mm"
                   if _orig_spacing else ""),
                className="text-muted align-self-center",
            ),
            width="auto",
        ),
        dbc.Col(
            html.Small(
                f"{len(_manifest_states)} deformed states loaded" if _manifest_states
                else "No ensemble — showing original only",
                className="text-warning align-self-center",
            ),
            width="auto",
        ),
        dbc.Col(
            html.Small(
                f"{len(_all_rois)} ROIs loaded" if has_rtstruct else "",
                className="text-success align-self-center",
            ),
            width="auto",
        ),
    ], className="mt-2 mb-1 align-items-center g-3")

    image_row = dbc.Row([
        dbc.Col(dcc.Graph(id="g-orig",  config={"scrollZoom": True, "displayModeBar": False},
                          style={"height": "440px"}), width=4),
        dbc.Col(dcc.Graph(id="g-def",   config={"scrollZoom": True, "displayModeBar": False},
                          style={"height": "440px"}), width=4),
        dbc.Col(dcc.Graph(id="g-diff",  config={"scrollZoom": True, "displayModeBar": False},
                          style={"height": "440px"}), width=4),
    ], className="g-1")

    slice_row = dbc.Row([
        dbc.Col([
            html.Small("Slice index", className="text-muted me-2"),
            dcc.Slider(
                id="slice-sl",
                min=0, max=nz - 1, step=1, value=nz // 2,
                tooltip={"always_visible": True, "placement": "top"},
                marks={0: "0", nz // 2: str(nz // 2), nz - 1: str(nz - 1)},
            ),
        ], width=12),
    ], className="mt-1")

    app.layout = dbc.Container([
        header,
        dbc.Row([
            dbc.Col(controls, width=2, style={"paddingRight": "4px"}),
            dbc.Col([image_row, slice_row], width=10),
        ]),
    ], fluid=True, style={"backgroundColor": "#0d1117", "minHeight": "100vh"})

    # ── Callback: W/L preset → update sliders ────────────────────────────────
    @app.callback(
        Output("w-sl", "value"),
        Output("l-sl", "value"),
        Input("wl-preset", "value"),
        prevent_initial_call=True,
    )
    def apply_wl_preset(preset: str | None):
        if not preset:
            return dash.no_update, dash.no_update
        w, l = preset.split(",")
        return int(w), int(l)

    # ── Callback: orientation change → update slice range ────────────────────
    @app.callback(
        Output("slice-sl", "max"),
        Output("slice-sl", "marks"),
        Output("slice-sl", "value"),
        Input("orient", "value"),
        State("slice-sl", "value"),
    )
    def update_slice_range(axis: int, current_val: int):
        n = _orig.shape[axis]
        mid = n // 2
        new_val = min(int(current_val or mid), n - 1)
        return n - 1, {0: "0", mid: str(mid), n - 1: str(n - 1)}, new_val

    # ── Callback: update all three image panels ───────────────────────────────
    @app.callback(
        Output("g-orig",  "figure"),
        Output("g-def",   "figure"),
        Output("g-diff",  "figure"),
        Output("info-box", "children"),
        Input("slice-sl",       "value"),
        Input("orient",         "value"),
        Input("state-dd",       "value"),
        Input("w-sl",           "value"),
        Input("l-sl",           "value"),
        Input("g-sl",           "value"),
        Input("dr-sl",          "value"),
        Input("overlay-toggle", "value"),
        Input("roi-show",       "value"),
        Input("roi-names",      "value"),
    )
    def update_images(
        slice_idx, axis, state_key,
        window, level, gamma, diff_range, overlay,
        roi_show, roi_names_sel,
    ):
        window     = float(window  or _DEF_WINDOW)
        level      = float(level   or _DEF_LEVEL)
        gamma      = float(gamma   or _DEF_GAMMA)
        diff_range = float(diff_range or _DEF_DIFF)
        axis       = int(axis or 0)
        show_roi   = bool(roi_show and "on" in roi_show)
        vis_names  = set(roi_names_sel) if roi_names_sel else None

        # Physical spacing per orientation — _orig_spacing is (sz, sy, sx) in mm
        sz, sy, sx = _orig_spacing
        if axis == 0:    # axial:    rows=y, cols=x
            row_mm, col_mm = sy, sx
        elif axis == 1:  # coronal:  rows=z, cols=x
            row_mm, col_mm = sz, sx
        else:            # sagittal: rows=z, cols=y
            row_mm, col_mm = sz, sy

        orig_s    = _get_slice(_orig, axis, slice_idx)
        orig_img  = _window_level(orig_s, window, level, gamma)
        fig_orig  = _make_fig(orig_img, GRAY, 0.0, 1.0, "Original",
                              row_spacing_mm=row_mm, col_spacing_mm=col_mm)
        if show_roi and _orig_rtstruct_rois:
            _add_contours(fig_orig, _orig_rtstruct_rois, axis, slice_idx, vis_names)

        # Axis labels for the title
        _ax_name = {0: "Axial", 1: "Coronal", 2: "Sagittal"}
        ax_label = _ax_name[axis]

        if state_key == "__none__" or state_key not in _deformed:
            fig_def  = _blank_fig("Deformed — no state selected")
            fig_diff = _blank_fig("Difference ΔHU — no state")
            info     = "No deformed state selected."
            return fig_orig, fig_def, fig_diff, info

        def_vol = _deformed[state_key]
        def_s   = _get_slice(def_vol, axis, slice_idx)
        def_img = _window_level(def_s, window, level, gamma)

        # Difference in raw HU
        diff_hu = def_s.astype(np.float32) - orig_s.astype(np.float32)

        # Deformed RTSTRUCT: use per-state deformed RS, fall back to original
        def_rois = _deformed_rtstruct_rois.get(state_key) or _orig_rtstruct_rois

        # Optional: blend diff as colour overlay on the deformed image
        # (contours NOT added when in overlay/gamma mode — underlying image is not CT HU)
        if overlay and "on" in overlay:
            fig_def = _make_fig(
                diff_hu, DIFF_CSCALE, -diff_range, diff_range,
                f"Deformed + ΔHU overlay  [{ax_label}]",
                row_spacing_mm=row_mm, col_spacing_mm=col_mm,
            )
        else:
            fig_def = _make_fig(def_img, GRAY, 0.0, 1.0, f"Deformed  [{ax_label}]",
                                row_spacing_mm=row_mm, col_spacing_mm=col_mm)
            if show_roi and def_rois:
                _add_contours(fig_def, def_rois, axis, slice_idx, vis_names)

        fig_diff = _make_fig(
            diff_hu, DIFF_CSCALE, -diff_range, diff_range,
            f"Difference ±{diff_range:.0f} HU  [{ax_label}]",
            row_spacing_mm=row_mm, col_spacing_mm=col_mm,
        )
        if show_roi and def_rois:
            _add_contours(fig_diff, def_rois, axis, slice_idx, vis_names)

        # ── Provenance info box ────────────────────────────────────────────
        entry  = next((s for s in _manifest_states if s.get("ct_dir") == state_key), {})
        tx     = entry.get("tx", 0.0);  ty = entry.get("ty", 0.0); tz = entry.get("tz", 0.0)
        rx     = entry.get("rx", 0.0);  ry = entry.get("ry", 0.0); rz = entry.get("rz", 0.0)
        wt     = entry.get("cluster_weight", "—")
        ts     = entry.get("iso_timestamp", "")[:19]
        abs_dh = np.abs(diff_hu)

        info = (
            f"State {entry.get('state_index', '?')}  [{ts}]\n"
            f"\n"
            f"Translation (mm)\n"
            f"  x: {tx:+.2f}  y: {ty:+.2f}  z: {tz:+.2f}\n"
            f"Rotation (°)\n"
            f"  rx:{rx:+.2f}  ry:{ry:+.2f}  rz:{rz:+.2f}\n"
            f"Cluster weight: {wt}\n"
            f"\n"
            f"This slice ΔHU stats\n"
            f"  mean |Δ| : {abs_dh.mean():.1f}\n"
            f"  max  Δ  : {diff_hu.max():.1f}\n"
            f"  min  Δ  : {diff_hu.min():.1f}\n"
            f"  >10 HU  : {(abs_dh > 10).mean() * 100:.1f}%"
        )

        return fig_orig, fig_def, fig_diff, info

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="DeformCT Viewer — compare original and deformed CT volumes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--orig-ct",  required=True,
                        help="Path to original CT DICOM directory")
    parser.add_argument("--ensemble", default=None,
                        help="Path to generate_ensemble output directory "
                             "(containing manifest.json and state subdirs)")
    parser.add_argument("--orig-rtstruct", default=None, metavar="FILE",
                        dest="orig_rtstruct",
                        help="Original RTSTRUCT DICOM for contour overlay on "
                             "the Original CT panel.  Deformed RTSTRUCTs are "
                             "auto-loaded from the ensemble manifest.")
    parser.add_argument("--host",  default="127.0.0.1",
                        help="Host to bind (use 0.0.0.0 to expose on LAN)")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--debug", action="store_true",
                        help="Enable Dash hot-reload and debug overlay")
    args = parser.parse_args()

    app = build_app(args.orig_ct, args.ensemble, args.orig_rtstruct)
    print(
        f"\n  DeformCT Viewer is running.\n"
        f"  Open  http://{args.host}:{args.port}/  in your browser.\n"
        f"  Ctrl-C to quit.\n"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
