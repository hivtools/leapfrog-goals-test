"""
Interactive comparison dashboard — Spectrum extract vs leapfrog-goals.

Usage:
    uv run shiny run app.py

Pre-requisites:
    - Run scripts/run_extract.py to produce the extract XLSX in output/extract/
    - Run scripts/run_goals_model.py to produce .npz files in output/goals/

The dashboard auto-detects available data on startup.
"""

from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, reactive, render, ui

import leapfrog_compare.config as config
from leapfrog_compare.extract_reader import read_extract_xlsx, get_pjnz_names, filter_extract
from leapfrog_compare.indicator_map import INDICATOR_MAP, get_indicator_names, compute_goals_series
from leapfrog_compare.pjnz_runner import load_goals_output

EXTRACT_FILE = config.EXTRACT_OUTPUT_FILE
GOALS_DIR = config.GOALS_OUTPUT_DIR

ALL_INDICATOR_NAMES = get_indicator_names()

_DEFAULT_YEAR_MIN = 1970
_DEFAULT_YEAR_MAX = 2030


def _find_extract_file() -> Path | None:
    return EXTRACT_FILE if EXTRACT_FILE.exists() else None


def _find_goals_files() -> dict[str, Path]:
    if not GOALS_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(GOALS_DIR.glob("*.npz"))}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.script(src="https://cdn.plot.ly/plotly-latest.min.js"),
    ),
    ui.page_sidebar(
    ui.sidebar(
        ui.h5("Filters"),
        ui.input_selectize(
            "pjnz",
            label="PJNZ",
            choices=[],
        ),
        ui.hr(),
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=ALL_INDICATOR_NAMES,
            multiple=True,
            selected=ALL_INDICATOR_NAMES[:3],
            options={"plugins": ["remove_button"]},
        ),
        ui.hr(),
        ui.input_slider(
            "year_range",
            "Year range",
            min=_DEFAULT_YEAR_MIN,
            max=_DEFAULT_YEAR_MAX,
            value=[_DEFAULT_YEAR_MIN, _DEFAULT_YEAR_MAX],
            step=1,
            sep="",
        ),
        ui.hr(),
        ui.input_action_button("refresh", "Refresh data", class_="btn-sm btn-secondary"),
        ui.hr(),
        ui.output_text("status_text"),
        width=320,
    ),
    ui.card(
        ui.card_header("Spectrum vs Leapfrog"),
        ui.output_ui("comparison_plot"),
        full_screen=True,
    ),
    title="Leapfrog Comparison",
    fillable=True,
)
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    @reactive.calc
    def _loaded_data():
        """Load (or reload) extract and goals data. Re-runs when Refresh is clicked."""
        _ = input.refresh()

        df = None
        xlsx = _find_extract_file()
        if xlsx is not None:
            try:
                df = read_extract_xlsx(xlsx)
            except Exception as exc:  # noqa: BLE001
                print(f"[app] Failed to load extract XLSX: {exc}")

        cache: dict = {}
        for stem, path in _find_goals_files().items():
            try:
                out, years = load_goals_output(path)
                cache[stem] = (out, years)
            except Exception as exc:  # noqa: BLE001
                print(f"[app] Failed to load goals output for {stem}: {exc}")

        extract_stems = set(get_pjnz_names(df)) if df is not None else set()
        all_stems = sorted(extract_stems | set(cache.keys()))

        return df, cache, all_stems

    @reactive.effect
    def _update_controls():
        df, cache, all_stems = _loaded_data()

        ui.update_select("pjnz", choices=all_stems, selected=all_stems[0] if all_stems else None)

        year_vals: list[int] = []
        for _, (_, years) in cache.items():
            year_vals.extend(int(y) for y in years)
        if df is not None and "year" in df.columns:
            year_vals.extend(int(y) for y in df["year"].dropna())
        if year_vals:
            y_min, y_max = min(year_vals), max(year_vals)
            ui.update_slider("year_range", min=y_min, max=y_max, value=[y_min, y_max])

    @output
    @render.text
    def status_text():
        df, cache, _ = _loaded_data()
        parts = []
        if df is None:
            parts.append("Extract: not loaded")
        else:
            parts.append(f"Extract: {df['pjnz_stem'].nunique()} PJNZ(s)")
        parts.append(f"Goals: {len(cache)} PJNZ(s)")
        return "\n".join(parts)

    @output
    @render.ui
    def comparison_plot():
        df, cache, all_stems = _loaded_data()

        pjnz = input.pjnz() or (all_stems[0] if all_stems else None)
        selected_indicators: tuple[str, ...] = input.indicators()
        year_start, year_end = input.year_range()

        if not pjnz or not selected_indicators:
            return ui.p("No data available." if not all_stems else "Select at least one indicator.")

        n = len(selected_indicators)
        ncols = min(n, 2)
        nrows = (n + ncols - 1) // ncols

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=list(selected_indicators),
            shared_xaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        for idx, indicator in enumerate(selected_indicators):
            row = idx // ncols + 1
            col = idx % ncols + 1

            # --- Extract series ---
            if df is not None:
                defn = INDICATOR_MAP[indicator]
                ext_series = filter_extract(
                    df,
                    pjnz_stem=pjnz,
                    indicator=defn.extract_name,
                    configuration=defn.extract_configuration,
                )
                if not ext_series.empty:
                    s = ext_series[(ext_series.index >= year_start) & (ext_series.index <= year_end)]
                    if not s.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=s.index.tolist(),
                                y=s.values.tolist(),
                                mode="lines",
                                name="Spectrum",
                                line=dict(color="#1f77b4", width=2),
                                legendgroup="Spectrum",
                                showlegend=(idx == 0),
                            ),
                            row=row,
                            col=col,
                        )

            # --- Goals series ---
            if pjnz in cache:
                goals_output, goals_years = cache[pjnz]
                try:
                    values = compute_goals_series(indicator, goals_output)
                    mask = (goals_years >= year_start) & (goals_years <= year_end)
                    x_years = goals_years[mask].tolist()
                    y_values = values[mask].tolist()
                    if x_years:
                        fig.add_trace(
                            go.Scatter(
                                x=x_years,
                                y=y_values,
                                mode="lines",
                                name="Leapfrog",
                                line=dict(color="#ff7f0e", width=2, dash="dash"),
                                legendgroup="Leapfrog",
                                showlegend=(idx == 0),
                            ),
                            row=row,
                            col=col,
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"[app] Could not compute {indicator} for {pjnz}: {exc}")

        fig.update_layout(
            height=max(400, 320 * nrows),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_text=f"Comparison — {pjnz}",
            margin=dict(t=80, b=40, l=60, r=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#e5e5e5",
            range=[year_start - 1, year_end + 1],
            tickformat="d",
        )
        fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")

        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.HTML(html)


app = App(app_ui, server)
