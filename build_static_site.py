"""
Build a static GitHub Pages version of the Dash dashboard.

The generated page does not need a Python server. It embeds current metrics,
predictions and Plotly charts into docs/index.html.

Run:
  .venv/bin/python3 build_static_site.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

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


def _layout(fig: go.Figure, height: int = 400, showlegend: bool = True) -> go.Figure:
    top = 92 if showlegend else 54
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CARD_ALT,
        font=dict(family=FONT, color=TXT, size=12),
        colorway=[BLUE, GREEN, RED, CYAN, PURPLE, ORANGE],
        height=height,
        margin=dict(l=58, r=28, t=top, b=46),
        showlegend=showlegend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="left",
            x=0,
            font=dict(size=10, color=TXT),
            itemsizing="constant",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0B1420",
            bordercolor=BORDER_STRONG,
            font=dict(family=FONT, color=TXT, size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        gridwidth=1,
        showline=False,
        zeroline=False,
        tickfont=dict(color=TXT2, size=11),
        title_font=dict(color=TXT2, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        gridwidth=1,
        showline=False,
        zeroline=False,
        tickfont=dict(color=TXT2, size=11),
        title_font=dict(color=TXT2, size=12),
    )
    return fig


def _plot_html(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "displayModeBar": True,
            "responsive": True,
            "scrollZoom": False,
            "modeBarButtonsToRemove": [
                "toImage",
                "sendDataToCloud",
                "select2d",
                "lasso2d",
            ],
        },
    )


def _load_results() -> tuple[dict[str, dict], dict[str, list[float]], np.ndarray, dict, pd.DataFrame]:
    results = {}
    histories = {}

    for path in sorted((RESULTS_DIR / "metrics").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if path.name.endswith("_history.json"):
            histories[path.name.replace("_history.json", "")] = data
            continue
        display_name = data.get("display_name", data["model"])
        results[display_name] = data

    predictions = {}
    for display_name, data in results.items():
        pred_path = RESULTS_DIR / "predictions" / f"{data['model']}.npy"
        if pred_path.exists():
            predictions[display_name] = np.load(pred_path).astype(float).round(4).tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (RESULTS_DIR / "data" / "prepared_data.pkl").open("rb") as fp:
            y_test = pickle.load(fp)["y_test"]

    dataset = pd.read_csv(RESULTS_DIR / "data" / "dataset.csv")
    return results, predictions, np.asarray(y_test, dtype=float), histories, dataset


def _build_leaderboard(results: dict[str, dict]) -> str:
    rows = []
    for idx, (name, row) in enumerate(
        sorted(results.items(), key=lambda item: item[1]["R2"], reverse=True),
        start=1,
    ):
        model_type = "Нейронная сеть" if row.get("type") == "neural_network" else "Классика"
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{model_type}</td>"
            f"<td><strong>{row['R2']:.4f}</strong></td>"
            f"<td>{row['RMSE']:.2f}</td>"
            f"<td>{row['MAE']:.2f}</td>"
            f"<td>{row['MSE']:.2f}</td>"
            "</tr>"
        )

    return """
    <div class="panel">
      <div class="section-title">Лидерборд</div>
      <div class="section-subtitle">Все модели отсортированы по R2. Лучшая модель сверху.</div>
      <div class="table-wrap leaderboard-table">
        <table>
          <thead>
            <tr><th>Место</th><th>Модель</th><th>Тип</th><th>R2</th><th>RMSE</th><th>MAE</th><th>MSE</th></tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </div>
    """.format(rows="\n".join(rows))


def _build_comparison(results: dict[str, dict], metric: str, title: str) -> go.Figure:
    classic = {name: row for name, row in results.items() if row.get("type") == "classic"}
    neural = {name: row for name, row in results.items() if row.get("type") == "neural_network"}

    fig = go.Figure()
    for name, rows, color in [
        ("Классика", classic, TXT),
        ("Нейронная сеть", neural, ORANGE),
    ]:
        values = [row[metric] for row in rows.values()]
        fig.add_trace(
            go.Bar(
                x=list(rows.keys()),
                y=values,
                name=name,
                marker_color=color,
                text=[
                    f"{value:.4f}" if metric == "R2" else f"{value:.2f}"
                    for value in values
                ],
                textposition="outside",
                textfont=dict(color=TXT, size=12, family=FONT),
            )
        )

    fig.update_yaxes(title_text=title)
    return _layout(fig, 410)


def _build_training(histories: dict, results: dict[str, dict]) -> go.Figure:
    nn_names = [name for name, row in results.items() if row.get("type") == "neural_network"]
    fig = make_subplots(rows=1, cols=len(nn_names), subplot_titles=nn_names)
    colors = [ORANGE, PURPLE, GREEN]

    for idx, name in enumerate(nn_names):
        model_key = results[name]["model"]
        history = histories.get(model_key)
        if not history:
            continue
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                y=history["loss"],
                mode="lines",
                name="Обучение",
                line=dict(color=color, width=2),
                showlegend=idx == 0,
            ),
            row=1,
            col=idx + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=history["val_loss"],
                mode="lines",
                name="Валидация",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=idx == 0,
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_xaxes(title_text="Эпоха")
    fig.update_yaxes(title_text="Потери (MSE)")
    return _layout(fig, 390)


def _build_correlation(dataset: pd.DataFrame) -> go.Figure:
    corr = dataset.corr(numeric_only=True)
    labels = [column.split(" (")[0] for column in corr.columns]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, color=TXT),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
            colorbar=dict(title="corr"),
        )
    )
    _layout(fig, 500, showlegend=False)
    fig.update_layout(
        xaxis=dict(tickangle=-45),
        margin=dict(l=120, b=120, t=40, r=28),
    )
    return fig


def _build_distribution(dataset: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=dataset[column],
            nbinsx=30,
            marker_color=CYAN,
            opacity=0.78,
        )
    )
    fig.update_xaxes(title_text=column)
    fig.update_yaxes(title_text="Частота")
    return _layout(fig, 380, showlegend=False)


def _build_dataset_scatter(dataset: pd.DataFrame, x_col: str, y_col: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=dataset[x_col],
            y=dataset[y_col],
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.55),
            hovertemplate=f"{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)
    return _layout(fig, 380, showlegend=False)


def _build_pair_single(dataset: pd.DataFrame, col_x: str, col_y: str, color: str) -> go.Figure:
    if col_x == col_y:
        fig = go.Figure(
            go.Histogram(
                x=dataset[col_x],
                nbinsx=25,
                marker_color=color,
                opacity=0.78,
            )
        )
        fig.update_yaxes(title_text="Кол-во")
    else:
        fig = go.Figure(
            go.Scatter(
                x=dataset[col_x],
                y=dataset[col_y],
                mode="markers",
                marker=dict(color=color, size=5, opacity=0.48),
            )
        )
        fig.update_yaxes(title_text=col_y.split(" (")[0])

    fig.update_xaxes(title_text=col_x.split(" (")[0])
    return _layout(fig, 320, showlegend=False)


def _build_static_charts(results: dict[str, dict], histories: dict, dataset: pd.DataFrame) -> str:
    target = dataset.columns[-1]
    cement_col = dataset.columns[0]
    age_col = dataset.columns[7]
    colors = [CYAN, GREEN, ORANGE, PURPLE, BLUE, RED, TXT2, CYAN, GREEN]

    pair_cards = []
    for idx, column in enumerate(dataset.columns):
        title = (
            f"{column.split(' (')[0]} vs Прочность"
            if column != target
            else f"Распределение: {column.split(' (')[0]}"
        )
        pair_cards.append(
            f"""
            <div class="panel compact">
              <div class="section-title small">{title}</div>
              {_plot_html(_build_pair_single(dataset, column, target, colors[idx % len(colors)]))}
            </div>
            """
        )

    return f"""
    <div class="grid two">
      <div class="panel">
        <div class="section-title">Корреляционная матрица</div>
        <div class="section-subtitle">Взаимосвязи между всеми признаками.</div>
        {_plot_html(_build_correlation(dataset))}
      </div>
      <div class="panel">
        <div class="section-title">Распределение прочности</div>
        <div class="section-subtitle">Гистограмма целевой переменной.</div>
        {_plot_html(_build_distribution(dataset, target))}
      </div>
    </div>

    <div class="grid two">
      <div class="panel">
        <div class="section-title">Цемент vs Прочность</div>
        <div class="section-subtitle">Связь количества цемента с прочностью бетона.</div>
        {_plot_html(_build_dataset_scatter(dataset, cement_col, target, CYAN))}
      </div>
      <div class="panel">
        <div class="section-title">Возраст vs Прочность</div>
        <div class="section-subtitle">Связь возраста образца с прочностью бетона.</div>
        {_plot_html(_build_dataset_scatter(dataset, age_col, target, GREEN))}
      </div>
    </div>

    <div class="panel">
      <div class="section-title">Сравнение всех моделей</div>
      <div class="section-subtitle">Метрики по всем сохраненным моделям.</div>
      {_plot_html(_build_comparison(results, "R2", "R2 Score"))}
      {_plot_html(_build_comparison(results, "RMSE", "RMSE (MPa)"))}
    </div>

    <div class="panel">
      <div class="section-title">Обучение нейронных сетей</div>
      <div class="section-subtitle">Потери MSE по эпохам для всех нейросетевых моделей.</div>
      {_plot_html(_build_training(histories, results))}
    </div>

    <div class="pair-grid">
      {"".join(pair_cards)}
    </div>
    """


def _json_script(name: str, value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False)
    return f'<script id="{name}" type="application/json">{payload}</script>'


def _build_html(
    results: dict[str, dict],
    predictions: dict[str, list[float]],
    y_test: np.ndarray,
    histories: dict,
    dataset: pd.DataFrame,
) -> str:
    classic_names = [name for name, row in results.items() if row.get("type") == "classic"]
    nn_names = [name for name, row in results.items() if row.get("type") == "neural_network"]
    best_name, best = max(results.items(), key=lambda item: item[1]["R2"])

    data_scripts = "\n".join(
        [
            _json_script("results-data", results),
            _json_script("predictions-data", predictions),
            _json_script("y-test-data", y_test.astype(float).round(4).tolist()),
            _json_script("classic-names-data", classic_names),
            _json_script("nn-names-data", nn_names),
        ]
    )

    static_charts = _build_static_charts(results, histories, dataset)
    leaderboard = _build_leaderboard(results)

    classic_buttons = "".join(
        f'<button class="model-btn classic-btn" data-model="{name}">{name}</button>'
        for name in classic_names
    )
    nn_buttons = "".join(
        f'<button class="model-btn nn-btn" data-model="{name}">{name}</button>'
        for name in nn_names
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Прогнозирование прочности бетона</title>
  <script src="https://cdn.plot.ly/plotly-3.6.0.min.js"></script>
  <style>
    :root {{
      color-scheme: dark;
      --bg: {BG};
      --surface: {SURFACE};
      --card: {CARD};
      --card-alt: {CARD_ALT};
      --border: {BORDER};
      --border-strong: {BORDER_STRONG};
      --txt: {TXT};
      --txt2: {TXT2};
      --txt3: {TXT3};
      --green: {GREEN};
      --red: {RED};
      --blue: {BLUE};
      --orange: {ORANGE};
      --purple: {PURPLE};
      --cyan: {CYAN};
      --font: {FONT};
      --shadow: {SHADOW};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      min-height: 100%;
      margin: 0;
      background:
        radial-gradient(circle at top center, rgba(255,255,255,0.07), transparent 22%),
        linear-gradient(180deg, #050505 0%, #000000 100%);
      color: var(--txt);
      font-family: var(--font);
    }}
    body {{ padding: 24px; }}
    .shell {{
      max-width: 1520px;
      margin: 0 auto;
    }}
    .hero {{
      padding: 30px 32px;
      border: 1px solid var(--border);
      border-radius: 30px;
      background: linear-gradient(135deg, rgba(18,18,18,0.98) 0%, rgba(6,6,6,0.96) 58%, rgba(0,0,0,0.99) 100%);
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }}
    .eyebrow {{
      display: inline-flex;
      padding: 7px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      color: var(--txt2);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .9px;
      text-transform: uppercase;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      line-height: 1;
      font-size: 46px;
      letter-spacing: 0;
      font-weight: 780;
    }}
    h1 span {{ color: var(--orange); }}
    .lead {{
      max-width: 780px;
      color: var(--txt2);
      line-height: 1.55;
      font-size: 15px;
      margin: 14px 0 0;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .chip {{
      min-width: 148px;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.03);
      color: var(--txt);
      font-size: 13px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .selector {{
      padding: 18px 22px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      margin-bottom: 12px;
    }}
    .label {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .9px;
      text-transform: uppercase;
      color: var(--txt3);
      margin-bottom: 10px;
    }}
    .btn-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .model-btn {{
      padding: 10px 18px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
      color: var(--txt2);
      font-family: var(--font);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      outline: none;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }}
    .model-btn.active {{
      border-color: var(--border-strong);
      background: rgba(255,255,255,0.08);
      color: var(--txt);
    }}
    .panel {{
      position: relative;
      padding: 22px 24px;
      border: 1px solid var(--border);
      border-radius: 26px;
      background: linear-gradient(180deg, rgba(14,14,14,0.98) 0%, rgba(4,4,4,0.98) 100%);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      margin-bottom: 16px;
    }}
    .panel::before {{
      content: "";
      display: block;
      width: 52px;
      height: 4px;
      border-radius: 999px;
      background: var(--txt);
      box-shadow: 0 0 28px rgba(246,247,251,0.2);
      margin-bottom: 18px;
    }}
    .panel.compact {{ padding: 16px 18px 12px; }}
    .section-title {{
      font-size: 24px;
      font-weight: 700;
      line-height: 1.1;
      margin-bottom: 0;
      letter-spacing: 0;
    }}
    .section-title.small {{ font-size: 18px; }}
    .section-subtitle {{
      color: var(--txt2);
      font-size: 13px;
      line-height: 1.5;
      max-width: 760px;
      margin-top: 8px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      margin-bottom: 0;
    }}
    .grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 12px;
    }}
    .metric-card {{
      min-width: 180px;
      min-height: 132px;
      padding: 18px 20px;
      border-radius: 20px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.008) 100%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }}
    .metric-label {{
      color: var(--txt3);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .8px;
      margin-bottom: 10px;
    }}
    .metric-value {{
      font-size: 29px;
      font-weight: 750;
      line-height: 1.05;
      letter-spacing: 0;
    }}
    .metric-sub {{
      margin-top: 9px;
      color: var(--txt2);
      font-size: 12px;
      line-height: 1.45;
    }}
    .delta-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--card-alt);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--txt2);
      background: #101927;
      font-size: 11px;
      letter-spacing: .6px;
      text-transform: uppercase;
    }}
    td {{ color: var(--txt); font-size: 14px; }}
    tbody tr:nth-child(odd) {{ background: rgba(255,255,255,0.018); }}
    .leaderboard-table tbody tr:first-child {{
      background: rgba(85, 224, 139, 0.06);
    }}
    .leaderboard-table tbody tr:first-child td:nth-child(4) {{
      color: var(--green);
      font-weight: 750;
      font-size: 17px;
    }}
    .js-plotly-plot .plotly .modebar {{
      opacity: 0.5;
      background: rgba(14,14,14,0.9) !important;
      border-radius: 12px;
      padding: 4px 8px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .js-plotly-plot .plotly .modebar:hover {{ opacity: 1; }}
    .js-plotly-plot .plotly .modebar-btn {{
      color: var(--txt2) !important;
    }}
    .js-plotly-plot .plotly .modebar-btn:hover {{
      color: var(--blue) !important;
    }}
    .pair-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .footer {{
      color: var(--txt3);
      font-size: 12px;
      padding: 12px 4px 28px;
    }}
    @media (max-width: 980px) {{
      body {{ padding: 18px; }}
      .hero {{ padding: 24px; }}
      h1 {{ font-size: 34px; }}
      .grid.two, .cards, .delta-grid, .pair-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  {data_scripts}
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">Прогнозирование прочности бетона</div>
      <h1>Классические методы <span>vs Нейронные сети</span></h1>
      <p class="lead">
        Статическая версия Dash-дэшборда для GitHub Pages. Данные, метрики,
        предсказания и графики встроены в HTML и работают без Python-сервера.
      </p>
      <div class="chips">
        <div class="chip">{len(dataset)} образцов</div>
        <div class="chip">{dataset.shape[1] - 1} признаков</div>
        <div class="chip">Лучшая модель: {best_name}</div>
        <div class="chip">R2 = {best["R2"]:.4f}</div>
      </div>
    </section>

    <section class="selector">
      <div class="label">Классические модели</div>
      <div class="btn-row">{classic_buttons}</div>
    </section>

    <section class="selector">
      <div class="label">Нейронные сети</div>
      <div class="btn-row">{nn_buttons}</div>
    </section>

    <section class="panel">
      <div class="section-title" id="selected-title"></div>
      <div class="section-subtitle">Сравнение выбранных моделей.</div>
      <div class="grid two">
        <div>
          <div class="section-subtitle" id="classic-title"></div>
          <div class="cards" id="classic-cards"></div>
        </div>
        <div>
          <div class="section-subtitle" id="nn-title"></div>
          <div class="cards" id="nn-cards"></div>
        </div>
      </div>
      <div class="delta-grid" id="delta-cards"></div>
    </section>

    <section class="grid two">
      <div class="panel">
        <div class="section-title" id="classic-scatter-title"></div>
        <div id="classic-scatter"></div>
      </div>
      <div class="panel">
        <div class="section-title" id="nn-scatter-title"></div>
        <div id="nn-scatter"></div>
      </div>
    </section>

    {leaderboard}

    <section class="panel">
      <div class="section-title" id="prediction-title"></div>
      <div class="section-subtitle">Первые 20 тестовых образцов: реальные vs предсказанные.</div>
      <div class="table-wrap" id="prediction-table"></div>
    </section>

    {static_charts}

    <div class="footer">
      Сгенерировано из локальных артефактов проекта: results/metrics,
      results/predictions и results/data.
    </div>
  </main>

  <script>
    const RESULTS = JSON.parse(document.getElementById("results-data").textContent);
    const PREDS = JSON.parse(document.getElementById("predictions-data").textContent);
    const Y_TEST = JSON.parse(document.getElementById("y-test-data").textContent);
    const CLASSIC_NAMES = JSON.parse(document.getElementById("classic-names-data").textContent);
    const NN_NAMES = JSON.parse(document.getElementById("nn-names-data").textContent);
    const COLORS = {{
      txt: "{TXT}",
      txt2: "{TXT2}",
      txt3: "{TXT3}",
      borderStrong: "{BORDER_STRONG}",
      cardAlt: "{CARD_ALT}",
      grid: "{GRID}",
      green: "{GREEN}",
      red: "{RED}",
      orange: "{ORANGE}",
      blue: "{BLUE}",
      purple: "{PURPLE}"
    }};
    let selectedClassic = CLASSIC_NAMES[0];
    let selectedNn = NN_NAMES[0];

    function metricCard(label, value, sub, color) {{
      return `
        <div class="metric-card">
          <div class="metric-label">${{label}}</div>
          <div class="metric-value" style="color:${{color}}">${{value}}</div>
          <div class="metric-sub">${{sub || ""}}</div>
        </div>
      `;
    }}

    function modelCards(row, typeLabel, color) {{
      return [
        metricCard("R2", row.R2.toFixed(4), typeLabel, color),
        metricCard("RMSE", row.RMSE.toFixed(2), "MPa", color),
        metricCard("MAE", row.MAE.toFixed(2), "MPa", color),
        metricCard("MSE", row.MSE.toFixed(2), "", color),
      ].join("");
    }}

    function baseLayout(height) {{
      return {{
        template: "plotly_dark",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: COLORS.cardAlt,
        font: {{family: "{FONT}", color: COLORS.txt, size: 12}},
        height,
        margin: {{l: 58, r: 28, t: 92, b: 46}},
        hovermode: "x unified",
        hoverlabel: {{
          bgcolor: "#0B1420",
          bordercolor: COLORS.borderStrong,
          font: {{family: "{FONT}", color: COLORS.txt, size: 12}}
        }},
        xaxis: {{
          gridcolor: COLORS.grid,
          gridwidth: 1,
          showline: false,
          zeroline: false,
          tickfont: {{color: COLORS.txt2, size: 11}},
          titlefont: {{color: COLORS.txt2, size: 12}}
        }},
        yaxis: {{
          gridcolor: COLORS.grid,
          gridwidth: 1,
          showline: false,
          zeroline: false,
          tickfont: {{color: COLORS.txt2, size: 11}},
          titlefont: {{color: COLORS.txt2, size: 12}}
        }},
        legend: {{
          bgcolor: "rgba(0,0,0,0)",
          borderwidth: 0,
          orientation: "h",
          yanchor: "bottom",
          y: 1.12,
          xanchor: "left",
          x: 0,
          font: {{size: 10, color: COLORS.txt}},
          itemsizing: "constant"
        }}
      }};
    }}

    function scatterPlot(divId, modelName, color) {{
      const pred = PREDS[modelName] || [];
      const row = RESULTS[modelName];
      const minVal = Math.min(...Y_TEST, ...pred);
      const maxVal = Math.max(...Y_TEST, ...pred);
      const data = [
        {{
          x: Y_TEST,
          y: pred,
          mode: "markers",
          type: "scatter",
          name: "Прогнозы",
          marker: {{color, size: 7, opacity: 0.6, line: {{width: 0.5, color: "rgba(0,0,0,0.4)"}}}},
          hovertemplate: "Реальная: %{{x:.1f}} МПа<br>Предсказанная: %{{y:.1f}} МПа<extra></extra>"
        }},
        {{
          x: [minVal, maxVal],
          y: [minVal, maxVal],
          mode: "lines",
          type: "scatter",
          name: "Идеал",
          line: {{color: COLORS.red, width: 2, dash: "dash"}},
          hoverinfo: "skip"
        }}
      ];
      const layout = baseLayout(420);
      layout.xaxis.title = "Реальные (МПа)";
      layout.yaxis.title = "Предсказанные (МПа)";
      Plotly.react(divId, data, layout, {{
        displayModeBar: true,
        responsive: true,
        scrollZoom: false,
        modeBarButtonsToRemove: ["toImage", "sendDataToCloud", "select2d", "lasso2d"]
      }});
      document.getElementById(`${{divId}}-title`);
      return `R2 = ${{row.R2.toFixed(4)}} | RMSE = ${{row.RMSE.toFixed(2)}} MPa`;
    }}

    function predictionTable() {{
      const pred = PREDS[selectedNn] || [];
      let rows = "";
      for (let i = 0; i < Math.min(20, Y_TEST.length, pred.length); i++) {{
        const err = Math.abs(Y_TEST[i] - pred[i]);
        const color = err < 3 ? COLORS.green : (err < 8 ? COLORS.orange : COLORS.red);
        rows += `
          <tr>
            <td>${{i + 1}}</td>
            <td>${{Y_TEST[i].toFixed(2)}}</td>
            <td>${{pred[i].toFixed(2)}}</td>
            <td style="color:${{color}}"><strong>${{err.toFixed(2)}}</strong></td>
          </tr>
        `;
      }}
      document.getElementById("prediction-title").textContent = `Предсказания — ${{selectedNn}}`;
      document.getElementById("prediction-table").innerHTML = `
        <table>
          <thead><tr><th>#</th><th>Реальная (МПа)</th><th>Предсказанная (МПа)</th><th>Ошибка (МПа)</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    function updateButtons() {{
      document.querySelectorAll(".classic-btn").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.model === selectedClassic);
      }});
      document.querySelectorAll(".nn-btn").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.model === selectedNn);
      }});
    }}

    function updateDashboard() {{
      const rc = RESULTS[selectedClassic];
      const rn = RESULTS[selectedNn];
      const deltaR2 = rn.R2 - rc.R2;
      const deltaRmse = rn.RMSE - rc.RMSE;
      document.getElementById("selected-title").textContent = `${{selectedClassic}} vs ${{selectedNn}}`;
      document.getElementById("classic-title").textContent = selectedClassic;
      document.getElementById("nn-title").textContent = selectedNn;
      document.getElementById("classic-cards").innerHTML = modelCards(rc, "Классика", COLORS.txt);
      document.getElementById("nn-cards").innerHTML = modelCards(rn, "НС", COLORS.orange);
      document.getElementById("delta-cards").innerHTML = [
        metricCard("Delta R2", `${{deltaR2 >= 0 ? "+" : ""}}${{deltaR2.toFixed(4)}}`, "НС vs Классика", deltaR2 > 0 ? COLORS.green : COLORS.red),
        metricCard("Delta RMSE", `${{deltaRmse >= 0 ? "+" : ""}}${{deltaRmse.toFixed(2)}}`, "MPa", deltaRmse < 0 ? COLORS.green : COLORS.red),
        metricCard("Архитектура НС", rn.architecture || "-", rn.params ? `${{rn.params.toLocaleString("ru-RU")}} параметров` : "-", COLORS.purple),
      ].join("");
      document.getElementById("classic-scatter-title").textContent =
        `${{selectedClassic}} · ${{scatterPlot("classic-scatter", selectedClassic, COLORS.txt)}}`;
      document.getElementById("nn-scatter-title").textContent =
        `${{selectedNn}} · ${{scatterPlot("nn-scatter", selectedNn, COLORS.orange)}}`;
      predictionTable();
      updateButtons();
    }}

    document.querySelectorAll(".classic-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        selectedClassic = btn.dataset.model;
        updateDashboard();
      }});
    }});
    document.querySelectorAll(".nn-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        selectedNn = btn.dataset.model;
        updateDashboard();
      }});
    }});
    updateDashboard();
  </script>
</body>
</html>
"""


def main() -> None:
    results, predictions, y_test, histories, dataset = _load_results()
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    html = _build_html(results, predictions, y_test, histories, dataset)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"Static site written to: {DOCS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
