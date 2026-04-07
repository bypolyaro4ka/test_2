"""
Concrete Strength Prediction — Dashboard.
Dark theme, glassmorphism, Dash + Plotly.
Same visual style as alpha_model dashboards.

Sections:
  1. Hero header with dataset chips
  2. Model selector (Classic / Neural Network)
  3. Metrics overview (cards)
  4. Predictions scatter (selected pair)
  5. All models comparison (R2, RMSE, MAE bar charts)
  6. NN training curves
  7. Predictions table

Запуск: .venv/bin/python3 dashboard.py
"""
from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input

# -- Theme (from alpha_model) ------------------------------------------------
BG = "#000000"
SURFACE = "rgba(8, 8, 8, 0.92)"
CARD = "#0B0B0B"
CARD_ALT = "#020202"
BORDER = "rgba(255, 255, 255, 0.10)"
BORDER_STRONG = "rgba(255, 255, 255, 0.18)"
TXT = "#F6F7FB"
TXT2 = "#A1A1A6"
TXT3 = "#6E6E73"
GREEN = "#55E08B"
RED = "#FF6B57"
BLUE = "#43B0FF"
PURPLE = "#A184FF"
ORANGE = "#FFB34D"
CYAN = "#63E6FF"
FONT = (
    "-apple-system, BlinkMacSystemFont, 'SF Pro Display', "
    "'SF Pro Text', 'Helvetica Neue', Arial, sans-serif"
)
SHADOW = "0 28px 80px rgba(0, 0, 0, 0.40)"
GRID = "rgba(255, 255, 255, 0.12)"
GRAPH_CFG = {
    "displayModeBar": True, "responsive": True, "scrollZoom": False,
    "modeBarButtonsToRemove": [
        "toImage", "sendDataToCloud", "select2d", "lasso2d",
    ],
}


# -- UI helpers ---------------------------------------------------------------
def _layout(fig, h=400, showlegend=True):
    top = 92 if showlegend else 54
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=CARD_ALT,
        font=dict(family=FONT, color=TXT, size=12),
        colorway=[BLUE, GREEN, RED, CYAN, PURPLE, ORANGE],
        xaxis=dict(
            gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
            tickfont=dict(color=TXT2, size=11),
            title_font=dict(color=TXT2, size=12),
        ),
        yaxis=dict(
            gridcolor=GRID, gridwidth=1, showline=False, zeroline=False,
            tickfont=dict(color=TXT2, size=11),
            title_font=dict(color=TXT2, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0, orientation="h",
            yanchor="bottom", y=1.12, xanchor="left", x=0,
            font=dict(size=10, color=TXT), itemsizing="constant",
        ),
        margin=dict(l=58, r=28, t=top, b=46),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0B1420", bordercolor=BORDER_STRONG,
            font=dict(family=FONT, color=TXT, size=12),
        ),
        height=h, showlegend=showlegend,
    )
    return fig


def _accent_bar():
    return html.Div(style={
        "width": "52px", "height": "4px", "borderRadius": "999px",
        "background": TXT,
        "boxShadow": "0 0 28px rgba(246,247,251,0.2)",
        "marginBottom": "18px",
    })


def _panel(children, padding="22px 24px"):
    if not isinstance(children, list):
        children = [children]
    return html.Div(
        [_accent_bar()] + children,
        style={
            "background": (
                "linear-gradient(180deg, rgba(14,14,14,0.98) 0%, "
                "rgba(4,4,4,0.98) 100%)"
            ),
            "border": "1px solid {}".format(BORDER),
            "borderRadius": "26px",
            "padding": padding,
            "boxShadow": SHADOW,
            "backdropFilter": "blur(16px)",
        },
    )


def _section_header(title, subtitle=None):
    ch = [html.Div(title, style={
        "fontSize": "24px", "fontWeight": "700", "color": TXT,
        "letterSpacing": "-0.5px", "lineHeight": "1.1",
    })]
    if subtitle:
        ch.append(html.Div(subtitle, style={
            "fontSize": "13px", "color": TXT2, "marginTop": "8px",
            "maxWidth": "760px", "lineHeight": "1.5",
        }))
    return html.Div(ch, style={"marginBottom": "16px"})


def _hero_chip(label, value):
    return html.Div([
        html.Div(label, style={
            "fontSize": "10px", "fontWeight": "700",
            "letterSpacing": "0.9px", "textTransform": "uppercase",
            "color": TXT3, "marginBottom": "7px",
        }),
        html.Div(value, style={
            "fontSize": "17px", "fontWeight": "700",
            "color": TXT, "lineHeight": "1.1",
        }),
    ], style={
        "padding": "14px 16px", "minWidth": "148px",
        "background": "rgba(255,255,255,0.03)",
        "border": "1px solid {}".format(BORDER),
        "borderRadius": "18px",
    })


def _card(label, value, sub=None, color=TXT):
    ch = [
        html.Div(label, style={
            "fontSize": "11px", "fontWeight": "700", "color": TXT3,
            "letterSpacing": "0.8px", "textTransform": "uppercase",
            "marginBottom": "10px",
        }),
        html.Div(str(value), style={
            "fontSize": "29px", "fontWeight": "750", "color": color,
            "lineHeight": "1.05", "letterSpacing": "-0.6px",
        }),
    ]
    if sub:
        ch.append(html.Div(sub, style={
            "fontSize": "12px", "color": TXT2,
            "marginTop": "9px", "lineHeight": "1.45",
        }))
    return html.Div(ch, style={
        "background": (
            "linear-gradient(180deg, rgba(255,255,255,0.025) 0%, "
            "rgba(255,255,255,0.008) 100%)"
        ),
        "border": "1px solid {}".format(BORDER),
        "borderRadius": "20px",
        "padding": "18px 20px", "minWidth": "180px", "minHeight": "132px",
        "flex": "1 1 220px",
        "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.04)",
    })


def _btn_style(active):
    return {
        "padding": "10px 18px", "borderRadius": "999px",
        "fontSize": "13px", "fontWeight": "700", "cursor": "pointer",
        "border": "1px solid {}".format(
            BORDER_STRONG if active else BORDER
        ),
        "background": (
            "rgba(255,255,255,0.08)" if active
            else "rgba(255,255,255,0.02)"
        ),
        "color": TXT if active else TXT2,
        "fontFamily": FONT, "outline": "none",
        "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.05)",
    }


# -- Data loading -------------------------------------------------------------
def _load_results():
    metrics_files = sorted(glob.glob('results/metrics/*.json'))
    all_results = {}
    nn_histories = {}

    for f in metrics_files:
        if '_history' in f:
            name = os.path.basename(f).replace('_history.json', '')
            with open(f, 'r') as fp:
                nn_histories[name] = json.load(fp)
            continue
        with open(f, 'r') as fp:
            data = json.load(fp)
        display = data.get('display_name', data['model'])
        all_results[display] = data

    all_preds = {}
    for name, r in all_results.items():
        pred_file = 'results/predictions/{}.npy'.format(r['model'])
        if os.path.exists(pred_file):
            all_preds[name] = np.load(pred_file)

    y_test = None
    pkl_path = 'results/data/prepared_data.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as fp:
            y_test = pickle.load(fp)['y_test']

    dataset = None
    csv_path = 'results/data/dataset.csv'
    if os.path.exists(csv_path):
        dataset = pd.read_csv(csv_path)

    return all_results, all_preds, y_test, nn_histories, dataset


# -- Charts -------------------------------------------------------------------
def _build_scatter(y_test, y_pred, name, color, r2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode='markers',
        marker=dict(color=color, size=7, opacity=0.6,
                    line=dict(width=0.5, color='rgba(0,0,0,0.4)')),
        name='Прогнозы',
        hovertemplate=(
            'Реальная: %{x:.1f} МПа<br>'
            'Предсказанная: %{y:.1f} МПа<extra></extra>'
        ),
    ))
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color=RED, width=2, dash='dash'),
        name='Идеал', hoverinfo='skip',
    ))
    fig.update_xaxes(title_text="Реальные (МПа)")
    fig.update_yaxes(title_text="Предсказанные (МПа)")
    _layout(fig, 420)
    return fig


def _build_comparison_bar(all_results, metric, title, reverse=False):
    classic = {n: r for n, r in all_results.items()
               if r.get('type') == 'classic'}
    nn = {n: r for n, r in all_results.items()
          if r.get('type') == 'neural_network'}

    names_c = list(classic.keys())
    names_n = list(nn.keys())
    vals_c = [classic[n][metric] for n in names_c]
    vals_n = [nn[n][metric] for n in names_n]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names_c, y=vals_c, name='Классика',
        marker_color=TXT, marker_line_width=0,
        text=['{:.4f}'.format(v) if metric == 'R2'
              else '{:.2f}'.format(v) for v in vals_c],
        textposition='outside',
        textfont=dict(color=TXT, size=12, family=FONT),
    ))
    fig.add_trace(go.Bar(
        x=names_n, y=vals_n, name='Нейронная сеть',
        marker_color=ORANGE, marker_line_width=0,
        text=['{:.4f}'.format(v) if metric == 'R2'
              else '{:.2f}'.format(v) for v in vals_n],
        textposition='outside',
        textfont=dict(color=TXT, size=12, family=FONT),
    ))
    fig.update_yaxes(title_text=title)
    _layout(fig, 400)
    return fig


def _build_training(nn_histories, all_results):
    nn_names = [n for n, r in all_results.items()
                if r.get('type') == 'neural_network']
    nn_models = [all_results[n]['model'] for n in nn_names]

    fig = make_subplots(
        rows=1, cols=len(nn_models),
        subplot_titles=nn_names,
    )
    colors = [ORANGE, PURPLE, GREEN]
    for i, (model_key, name) in enumerate(zip(nn_models, nn_names)):
        if model_key not in nn_histories:
            continue
        h = nn_histories[model_key]
        c = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            y=h['loss'], mode='lines', name='Обучение',
            line=dict(color=c, width=2),
            showlegend=(i == 0),
        ), row=1, col=i + 1)
        fig.add_trace(go.Scatter(
            y=h['val_loss'], mode='lines', name='Валидация',
            line=dict(color=c, width=2, dash='dash'),
            showlegend=(i == 0),
        ), row=1, col=i + 1)

    fig.update_xaxes(title_text="Эпоха")
    fig.update_yaxes(title_text="Потери (MSE)")
    _layout(fig, 380)
    return fig


def _build_radar(all_results, selected):
    if not selected:
        fig = go.Figure()
        _layout(fig, 400, showlegend=False)
        return fig

    max_rmse = max(r['RMSE'] for r in all_results.values())
    max_mae = max(r['MAE'] for r in all_results.values())
    categories = ['R2', '1 - RMSE/max', '1 - MAE/max']
    colors_map = [BLUE, GREEN, RED, CYAN, PURPLE, ORANGE]

    fig = go.Figure()
    for i, name in enumerate(selected):
        r = all_results[name]
        vals = [
            r['R2'],
            1 - r['RMSE'] / max_rmse,
            1 - r['MAE'] / max_mae,
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill='toself', name=name, opacity=0.6,
            line=dict(color=colors_map[i % len(colors_map)]),
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_ALT,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=GRID, color=TXT2,
            ),
            angularaxis=dict(
                gridcolor=GRID, color=TXT2,
            ),
        ),
    )
    _layout(fig, 440)
    return fig


def _build_pred_table(y_test, y_pred, n=20):
    if y_test is None or y_pred is None:
        return html.Div("No data", style={"color": TXT2, "padding": "20px"})

    hs = {
        "fontSize": "11px", "fontWeight": "700", "color": TXT2,
        "textTransform": "uppercase", "letterSpacing": "0.6px",
        "padding": "13px 14px",
        "borderBottom": "1px solid {}".format(BORDER),
        "textAlign": "left", "backgroundColor": "#101927",
        "position": "sticky", "top": 0, "zIndex": 1,
    }
    cs = {
        "fontSize": "13px", "color": TXT, "padding": "11px 14px",
        "borderBottom": "1px solid {}".format(BORDER),
    }

    columns = ["#", "Реальная (МПа)", "Предсказанная (МПа)", "Ошибка (МПа)"]
    header = html.Tr([html.Th(c, style=hs) for c in columns])

    rows = []
    for i in range(min(n, len(y_test))):
        real = y_test[i]
        pred = y_pred[i]
        err = abs(real - pred)
        err_color = GREEN if err < 3 else (ORANGE if err < 8 else RED)
        bg = ("rgba(255,255,255,0.018)" if i % 2 == 0
              else "transparent")
        rows.append(html.Tr([
            html.Td(str(i + 1), style=cs),
            html.Td("{:.2f}".format(real), style=cs),
            html.Td("{:.2f}".format(pred), style=cs),
            html.Td(
                "{:.2f}".format(err),
                style={**cs, "color": err_color, "fontWeight": "700"},
            ),
        ], style={"backgroundColor": bg}))

    return html.Div(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={
            "backgroundColor": CARD_ALT, "borderRadius": "20px",
            "border": "1px solid {}".format(BORDER),
            "maxHeight": "520px", "overflow": "auto",
        },
    )


def _build_leaderboard(all_results):
    """Leaderboard table sorted by R2 descending."""
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]['R2'],
        reverse=True,
    )

    hs = {
        "fontSize": "11px", "fontWeight": "700", "color": TXT2,
        "textTransform": "uppercase", "letterSpacing": "0.6px",
        "padding": "14px 16px",
        "borderBottom": "1px solid {}".format(BORDER),
        "textAlign": "left", "backgroundColor": "#101927",
        "position": "sticky", "top": 0, "zIndex": 1,
    }
    cs = {
        "fontSize": "14px", "color": TXT, "padding": "14px 16px",
        "borderBottom": "1px solid {}".format(BORDER),
    }

    columns = [
        "Место", "Модель", "Тип", "R2", "RMSE", "MAE", "MSE",
    ]
    header = html.Tr([html.Th(c, style=hs) for c in columns])

    rows = []
    for i, (name, r) in enumerate(sorted_models):
        is_nn = r.get('type') == 'neural_network'
        model_type = "Нейронная сеть" if is_nn else "Классика"
        type_color = ORANGE if is_nn else TXT

        # Rank styling
        rank_text = str(i + 1)
        if i == 0:
            row_bg = "rgba(85, 224, 139, 0.06)"
            r2_color = GREEN
        else:
            row_bg = (
                "rgba(255,255,255,0.018)" if i % 2 == 0
                else "transparent"
            )
            r2_color = TXT if i < 3 else TXT2

        rank_style = {
            **cs,
            "fontSize": "16px", "fontWeight": "700",
            "textAlign": "center", "width": "60px",
        }

        rows.append(html.Tr([
            html.Td(rank_text, style=rank_style),
            html.Td(name, style={
                **cs,
                "fontWeight": "700",
                "fontSize": "15px",
            }),
            html.Td(model_type, style={
                **cs, "color": type_color, "fontWeight": "600",
            }),
            html.Td("{:.4f}".format(r['R2']), style={
                **cs, "color": r2_color, "fontWeight": "750",
                "fontSize": "17px",
            }),
            html.Td("{:.2f}".format(r['RMSE']), style=cs),
            html.Td("{:.2f}".format(r['MAE']), style=cs),
            html.Td("{:.2f}".format(r['MSE']), style=cs),
        ], style={"backgroundColor": row_bg}))

    return html.Div(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        ),
        style={
            "backgroundColor": CARD_ALT, "borderRadius": "20px",
            "border": "1px solid {}".format(BORDER),
            "overflow": "auto",
        },
    )


# -- EDA charts ---------------------------------------------------------------
def _build_correlation(dataset):
    """Correlation heatmap."""
    corr = dataset.corr()
    labels = [c.split(' (')[0] for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale='RdBu_r', zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        textfont=dict(size=10, color=TXT),
        hovertemplate=(
            "%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>"
        ),
    ))
    _layout(fig, 500, showlegend=False)
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        margin=dict(l=120, b=120, t=40, r=28),
    )
    return fig


def _build_eda_scatter(dataset, x_col, y_col, color=CYAN):
    """Scatter plot for two columns."""
    fig = go.Figure(go.Scatter(
        x=dataset[x_col], y=dataset[y_col],
        mode='markers',
        marker=dict(
            color=color, size=6, opacity=0.5,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
        ),
        hovertemplate=(
            "{}: %{{x:.1f}}<br>{}: %{{y:.1f}}"
            "<extra></extra>".format(
                x_col.split(' (')[0], y_col.split(' (')[0],
            )
        ),
    ))
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)
    _layout(fig, 380, showlegend=False)
    return fig


def _build_distribution(dataset, col, color=CYAN):
    """Histogram of a column."""
    fig = go.Figure(go.Histogram(
        x=dataset[col], nbinsx=30,
        marker_color=color, marker_line_width=0,
        opacity=0.8,
        hovertemplate="Диапазон: %{x}<br>Кол-во: %{y}<extra></extra>",
    ))
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Количество")
    _layout(fig, 380, showlegend=False)
    return fig


def _build_pair_single(dataset, col_x, col_y, color=CYAN):
    """Single pair chart: scatter if x!=y, histogram if x==y."""
    short_x = col_x.split(' (')[0]
    short_y = col_y.split(' (')[0]
    if col_x == col_y:
        fig = go.Figure(go.Histogram(
            x=dataset[col_x], nbinsx=25,
            marker_color=color, marker_line_width=0,
            opacity=0.8,
        ))
        fig.update_xaxes(title_text=short_x)
        fig.update_yaxes(title_text="Кол-во")
    else:
        fig = go.Figure(go.Scatter(
            x=dataset[col_x], y=dataset[col_y],
            mode='markers',
            marker=dict(color=color, size=5, opacity=0.5),
        ))
        fig.update_xaxes(title_text=short_x)
        fig.update_yaxes(title_text=short_y)
    _layout(fig, 320, showlegend=False)
    fig.update_layout(margin=dict(l=50, r=16, t=30, b=46))
    return fig


# -- App factory --------------------------------------------------------------
ALL_RESULTS, ALL_PREDS, Y_TEST, NN_HIST, DATASET = _load_results()

CLASSIC_NAMES = [
    n for n, r in ALL_RESULTS.items()
    if r.get('type') == 'classic'
]
NN_NAMES = [
    n for n, r in ALL_RESULTS.items()
    if r.get('type') == 'neural_network'
]
ALL_NAMES = CLASSIC_NAMES + NN_NAMES

app = Dash(__name__)

app.index_string = """<!DOCTYPE html>
<html>
<head>{%metas%}
<title>Прогнозирование прочности бетона</title>
{%css%}
<style>
html, body, #react-entry-point, #_dash-app-content {
    min-height: 100%;
    background:
        radial-gradient(circle at top center,
            rgba(255,255,255,0.07), transparent 22%),
        linear-gradient(180deg, #050505 0%, #000000 100%) !important;
    margin: 0; padding: 0;
}
body { color: #F6F7FB; }
* { box-sizing: border-box; }
.js-plotly-plot .plotly .modebar {
    opacity: 0.5;
    background: rgba(14,14,14,0.9) !important;
    border-radius: 12px;
    padding: 4px 8px;
    border: 1px solid rgba(255,255,255,0.08);
}
.js-plotly-plot .plotly .modebar:hover { opacity: 1; }
.js-plotly-plot .plotly .modebar-btn {
    color: #A1A1A6 !important;
}
.js-plotly-plot .plotly .modebar-btn:hover {
    color: #43B0FF !important;
}
@media (max-width: 980px) {
    .dashboard-shell { padding: 18px !important; }
    .dashboard-hero { padding: 24px !important; }
}
</style>
</head>
<body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body>
</html>"""

# Dataset info
n_rows = len(DATASET) if DATASET is not None else 0
n_features = DATASET.shape[1] - 1 if DATASET is not None else 0
target_col = DATASET.columns[-1] if DATASET is not None else ''
best_name = max(ALL_RESULTS, key=lambda n: ALL_RESULTS[n]['R2'])
best_r2 = ALL_RESULTS[best_name]['R2']

app.layout = html.Div([
    # ── Hero header ──
    html.Div([
        html.Div([
            html.Div(
                "Прогнозирование прочности бетона",
                style={
                    "display": "inline-flex", "padding": "7px 12px",
                    "borderRadius": "999px",
                    "border": "1px solid {}".format(BORDER),
                    "background": "rgba(255,255,255,0.03)",
                    "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "0.9px",
                    "textTransform": "uppercase", "color": TXT2,
                    "marginBottom": "14px",
                },
            ),
            html.Div([
                html.Span("Классические методы", style={
                    "fontSize": "46px", "fontWeight": "780",
                    "color": TXT, "letterSpacing": "-1.2px",
                }),
                html.Span(" vs ", style={
                    "fontSize": "46px", "fontWeight": "320",
                    "color": TXT3, "letterSpacing": "-1.1px",
                }),
                html.Span("Нейронные сети", style={
                    "fontSize": "46px", "fontWeight": "780",
                    "color": ORANGE, "letterSpacing": "-1.2px",
                }),
            ], style={
                "display": "flex", "gap": "8px", "flexWrap": "wrap",
                "alignItems": "baseline", "lineHeight": "1.0",
            }),
            html.Div(
                "Прогнозирование прочности бетона на сжатие "
                "с помощью классических методов ML и нейронных "
                "сетей Keras. Датасет UCI, 1030 образцов, "
                "8 признаков.",
                style={
                    "fontSize": "15px", "color": TXT2,
                    "lineHeight": "1.55", "maxWidth": "760px",
                    "marginTop": "14px",
                },
            ),
        ], style={"flex": "1 1 100%"}),
    ], className="dashboard-hero", style={
        "display": "flex", "gap": "20px", "flexWrap": "wrap",
        "alignItems": "flex-end", "padding": "30px 32px",
        "background": (
            "linear-gradient(135deg, rgba(18,18,18,0.98) 0%, "
            "rgba(6,6,6,0.96) 58%, rgba(0,0,0,0.99) 100%)"
        ),
        "border": "1px solid {}".format(BORDER),
        "borderRadius": "30px",
        "boxShadow": SHADOW, "marginBottom": "16px",
    }),

    # ── Model selector: Classic ──
    html.Div([
        html.Div("Классические модели", style={
            "fontSize": "12px", "fontWeight": "700",
            "letterSpacing": "0.9px", "textTransform": "uppercase",
            "color": TXT3, "marginBottom": "10px",
        }),
        html.Div([
            html.Button(
                n, id="btn-c-{}".format(i), n_clicks=0,
                style=_btn_style(i == 0),
            ) for i, n in enumerate(CLASSIC_NAMES)
        ], style={
            "display": "flex", "gap": "10px", "flexWrap": "wrap",
        }),
    ], style={
        "padding": "18px 22px", "background": SURFACE,
        "border": "1px solid {}".format(BORDER),
        "borderRadius": "22px",
        "boxShadow": SHADOW, "marginBottom": "12px",
    }),

    # ── Model selector: Neural Network ──
    html.Div([
        html.Div("Нейронные сети", style={
            "fontSize": "12px", "fontWeight": "700",
            "letterSpacing": "0.9px", "textTransform": "uppercase",
            "color": TXT3, "marginBottom": "10px",
        }),
        html.Div([
            html.Button(
                n, id="btn-n-{}".format(i), n_clicks=0,
                style=_btn_style(i == 0),
            ) for i, n in enumerate(NN_NAMES)
        ], style={
            "display": "flex", "gap": "10px", "flexWrap": "wrap",
        }),
    ], style={
        "padding": "18px 22px", "background": SURFACE,
        "border": "1px solid {}".format(BORDER),
        "borderRadius": "22px",
        "boxShadow": SHADOW, "marginBottom": "16px",
    }),

    dcc.Store(id="sel-classic", data=CLASSIC_NAMES[0] if CLASSIC_NAMES else ""),
    dcc.Store(id="sel-nn", data=NN_NAMES[0] if NN_NAMES else ""),

    # ── EDA section (static) ──
    html.Div(id="eda-section", style={"marginBottom": "16px"}),

    # ── Dynamic sections ──
    html.Div(id="metrics-section", style={"marginBottom": "16px"}),
    html.Div(id="leaderboard-section", style={"marginBottom": "16px"}),
    html.Div(id="scatter-section", style={
        "display": "flex", "gap": "16px",
        "flexWrap": "wrap", "marginBottom": "16px",
    }),
    html.Div(id="comparison-section", style={"marginBottom": "16px"}),
    html.Div(id="training-section", style={"marginBottom": "16px"}),
    html.Div(id="radar-section", style={"marginBottom": "16px"}),
    html.Div(id="table-section", style={"paddingBottom": "32px"}),

], className="dashboard-shell", style={
    "minHeight": "100vh", "fontFamily": FONT,
    "maxWidth": "1520px", "margin": "0 auto",
    "padding": "24px 24px 32px",
})


# -- Callbacks ----------------------------------------------------------------
@app.callback(
    [Output("sel-classic", "data")]
    + [Output("btn-c-{}".format(i), "style")
       for i in range(len(CLASSIC_NAMES))],
    [Input("btn-c-{}".format(i), "n_clicks")
     for i in range(len(CLASSIC_NAMES))],
)
def toggle_classic(*clicks):
    from dash import ctx
    triggered = ctx.triggered_id
    idx = 0
    if triggered:
        idx = int(triggered.replace("btn-c-", ""))
    selected = CLASSIC_NAMES[idx]
    styles = [_btn_style(i == idx) for i in range(len(CLASSIC_NAMES))]
    return [selected] + styles


@app.callback(
    [Output("sel-nn", "data")]
    + [Output("btn-n-{}".format(i), "style")
       for i in range(len(NN_NAMES))],
    [Input("btn-n-{}".format(i), "n_clicks")
     for i in range(len(NN_NAMES))],
)
def toggle_nn(*clicks):
    from dash import ctx
    triggered = ctx.triggered_id
    idx = 0
    if triggered:
        idx = int(triggered.replace("btn-n-", ""))
    selected = NN_NAMES[idx]
    styles = [_btn_style(i == idx) for i in range(len(NN_NAMES))]
    return [selected] + styles


@app.callback(
    [Output("eda-section", "children"),
     Output("metrics-section", "children"),
     Output("leaderboard-section", "children"),
     Output("scatter-section", "children"),
     Output("comparison-section", "children"),
     Output("training-section", "children"),
     Output("radar-section", "children"),
     Output("table-section", "children")],
    [Input("sel-classic", "data"),
     Input("sel-nn", "data")],
)
def update_all(sel_classic, sel_nn):
    rc = ALL_RESULTS.get(sel_classic, {})
    rn = ALL_RESULTS.get(sel_nn, {})

    # -- Metrics cards --
    def _model_cards(r, model_type):
        color = TXT if model_type == 'classic' else ORANGE
        type_label = "Классика" if model_type == 'classic' else "НС"
        return [
            _card("R2", "{:.4f}".format(r.get('R2', 0)),
                  type_label, color),
            _card("RMSE", "{:.2f}".format(r.get('RMSE', 0)),
                  "MPa", color),
            _card("MAE", "{:.2f}".format(r.get('MAE', 0)),
                  "MPa", color),
            _card("MSE", "{:.2f}".format(r.get('MSE', 0)),
                  None, color),
        ]

    # Delta
    delta_r2 = rn.get('R2', 0) - rc.get('R2', 0)
    delta_rmse = rn.get('RMSE', 0) - rc.get('RMSE', 0)
    delta_color = GREEN if delta_r2 > 0 else RED

    metrics = _panel([
        _section_header(
            "{} vs {}".format(sel_classic, sel_nn),
            "Сравнение выбранных моделей.",
        ),
        html.Div([
            html.Div(
                _model_cards(rc, 'classic'),
                style={
                    "display": "flex", "gap": "12px",
                    "flexWrap": "wrap", "flex": "1",
                },
            ),
            html.Div(style={
                "width": "1px", "background": BORDER,
                "margin": "0 8px",
            }),
            html.Div(
                _model_cards(rn, 'neural_network'),
                style={
                    "display": "flex", "gap": "12px",
                    "flexWrap": "wrap", "flex": "1",
                },
            ),
        ], style={
            "display": "flex", "gap": "16px",
            "flexWrap": "wrap",
        }),
        html.Div([
            _card(
                "Delta R2",
                "{:+.4f}".format(delta_r2),
                "НС vs Классика", delta_color,
            ),
            _card(
                "Delta RMSE",
                "{:+.2f}".format(delta_rmse),
                "MPa",
                GREEN if delta_rmse < 0 else RED,
            ),
            _card(
                "Архитектура НС",
                rn.get('architecture', '-'),
                "{} параметров".format(
                    "{:,}".format(rn['params']) if 'params' in rn
                    else '-'
                ),
                PURPLE,
            ),
        ], style={
            "display": "flex", "gap": "12px",
            "flexWrap": "wrap", "marginTop": "16px",
        }),
    ])

    # -- Scatter plots --
    scatter_children = []
    for name, color, label in [
        (sel_classic, TXT, 'Classic'),
        (sel_nn, ORANGE, 'Neural Network'),
    ]:
        if name in ALL_PREDS and Y_TEST is not None:
            r2 = ALL_RESULTS[name]['R2']
            fig = _build_scatter(
                Y_TEST, ALL_PREDS[name], name, color, r2,
            )
            scatter_children.append(html.Div(
                _panel([
                    _section_header(
                        name,
                        "R2 = {:.4f} | RMSE = {:.2f} MPa".format(
                            ALL_RESULTS[name]['R2'],
                            ALL_RESULTS[name]['RMSE'],
                        ),
                    ),
                    dcc.Graph(
                        figure=fig, config=GRAPH_CFG,
                        style={"height": "100%"},
                    ),
                ], padding="22px 22px 16px"),
                style={
                    "flex": "1 1 520px", "minWidth": "320px",
                },
            ))

    # -- Comparison bars --
    comparison = _panel([
        _section_header(
            "Сравнение всех моделей",
            "Метрики по всем 7 моделям.",
        ),
        dcc.Graph(
            figure=_build_comparison_bar(
                ALL_RESULTS, 'R2', 'R2 Score',
            ),
            config=GRAPH_CFG,
        ),
        dcc.Graph(
            figure=_build_comparison_bar(
                ALL_RESULTS, 'RMSE', 'RMSE (MPa)',
            ),
            config=GRAPH_CFG,
        ),
    ], padding="22px 22px 16px")

    # -- Training curves --
    training = _panel([
        _section_header(
            "Обучение нейронных сетей",
            "Потери (MSE) по эпохам для всех НС.",
        ),
        dcc.Graph(
            figure=_build_training(NN_HIST, ALL_RESULTS),
            config=GRAPH_CFG,
        ),
    ], padding="22px 22px 16px")

    # -- Radar --
    radar = _panel([
        _section_header(
            "Радар сравнения",
            "Выбранные модели рядом. "
            "Чем больше площадь — тем лучше модель.",
        ),
        dcc.Graph(
            figure=_build_radar(
                ALL_RESULTS, [sel_classic, sel_nn],
            ),
            config=GRAPH_CFG,
        ),
    ], padding="22px 22px 16px")

    # -- Table --
    pred_for_table = ALL_PREDS.get(sel_nn)
    table = _panel([
        _section_header(
            "Предсказания — {}".format(sel_nn),
            "Первые 20 тестовых образцов: реальные vs предсказанные.",
        ),
        _build_pred_table(Y_TEST, pred_for_table),
    ])

    # -- Leaderboard --
    leaderboard = _panel([
        _section_header(
            "Лидерборд",
            "Все модели отсортированы по R2. "
            "Лучшая модель сверху.",
        ),
        _build_leaderboard(ALL_RESULTS),
    ])

    # -- EDA --
    eda_children = []
    if DATASET is not None:
        target = DATASET.columns[-1]
        cement_col = DATASET.columns[0]
        age_col = DATASET.columns[7]

        # Correlation + Distribution row
        eda_children.append(html.Div([
            html.Div(
                _panel([
                    _section_header(
                        "Корреляционная матрица",
                        "Взаимосвязи между всеми признаками.",
                    ),
                    dcc.Graph(
                        figure=_build_correlation(DATASET),
                        config=GRAPH_CFG,
                    ),
                ], padding="22px 22px 16px"),
                style={"flex": "1 1 520px", "minWidth": "320px"},
            ),
            html.Div(
                _panel([
                    _section_header(
                        "Распределение прочности",
                        "Гистограмма целевой переменной — "
                        "прочность бетона на сжатие.",
                    ),
                    dcc.Graph(
                        figure=_build_distribution(
                            DATASET, target, CYAN,
                        ),
                        config=GRAPH_CFG,
                    ),
                ], padding="22px 22px 16px"),
                style={"flex": "1 1 520px", "minWidth": "320px"},
            ),
        ], style={
            "display": "flex", "gap": "16px",
            "flexWrap": "wrap", "marginBottom": "16px",
        }))

        # Scatter: cement + age row
        eda_children.append(html.Div([
            html.Div(
                _panel([
                    _section_header(
                        "Цемент vs Прочность",
                        "Чем больше цемента — тем выше "
                        "прочность бетона.",
                    ),
                    dcc.Graph(
                        figure=_build_eda_scatter(
                            DATASET, cement_col, target, CYAN,
                        ),
                        config=GRAPH_CFG,
                    ),
                ], padding="22px 22px 16px"),
                style={"flex": "1 1 520px", "minWidth": "320px"},
            ),
            html.Div(
                _panel([
                    _section_header(
                        "Возраст vs Прочность",
                        "Прочность бетона растёт с возрастом.",
                    ),
                    dcc.Graph(
                        figure=_build_eda_scatter(
                            DATASET, age_col, target, GREEN,
                        ),
                        config=GRAPH_CFG,
                    ),
                ], padding="22px 22px 16px"),
                style={"flex": "1 1 520px", "minWidth": "320px"},
            ),
        ], style={
            "display": "flex", "gap": "16px",
            "flexWrap": "wrap", "marginBottom": "16px",
        }))

        # Парные графики — сетка отдельных графиков
        cols = DATASET.columns.tolist()
        target = cols[-1]
        pair_colors = [
            CYAN, GREEN, ORANGE, PURPLE,
            BLUE, RED, TXT2, CYAN, GREEN,
        ]
        pair_graphs = []
        for idx, col in enumerate(cols):
            c = pair_colors[idx % len(pair_colors)]
            fig = _build_pair_single(DATASET, col, target, c)
            short = col.split(' (')[0]
            pair_graphs.append(html.Div(
                _panel([
                    _section_header(
                        "{} vs Прочность".format(short)
                        if col != target
                        else "Распределение: {}".format(short),
                    ),
                    dcc.Graph(
                        figure=fig, config=GRAPH_CFG,
                    ),
                ], padding="16px 18px 12px"),
                style={
                    "flex": "1 1 340px", "minWidth": "300px",
                },
            ))
        eda_children.append(html.Div(
            pair_graphs,
            style={
                "display": "flex", "gap": "12px",
                "flexWrap": "wrap",
            },
        ))

    eda_section = html.Div(eda_children) if eda_children else html.Div()

    return (
        eda_section,
        metrics,
        leaderboard,
        scatter_children,
        comparison,
        training,
        radar,
        table,
    )


# -- Run ----------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=False, port=8050)
