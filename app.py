"""
Interactive comparison dashboard — Spectrum vs leapfrog-goals.

Usage:
    uv run shiny run app.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, reactive, render, ui

import leapfrog_compare.config as config
from leapfrog_compare.indicator_map import (
    AGE_LABELS, ALL_AGES_INDICATORS, INDICATORS_1549,
    RISK_GROUPS, compute_rg_goals, compute_rg_spectrum,
)
from leapfrog_compare.pjnz_runner import run_pjnz

# Colors assigned per demographic group (e.g. "Male", "Female", "Total", "0-4").
# Leapfrog = solid line; Spectrum = dashed line in the same color.
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2",
]

_ALL_AGES_NAMES = list(ALL_AGES_INDICATORS.keys())
_1549_NAMES = list(INDICATORS_1549.keys())
_DEFAULT_YEAR_MIN = 1970
_DEFAULT_YEAR_MAX = 2030
_N_AGE_GROUPS = len(AGE_LABELS)

_pjnz_files: dict[str, Path] = {
    p.stem: p
    for p in sorted(config.PJNZ_DIR.expanduser().glob("*.PJNZ"))
}
_pjnz_stems = list(_pjnz_files.keys())


def _series_for_age_cell(
    all_series: list[tuple[str, np.ndarray]],
    age_label: str,
) -> list[tuple[str, np.ndarray]]:
    """Extract series for one age-group column from a fully disaggregated series list."""
    cell: list[tuple[str, np.ndarray]] = []
    for label, values in all_series:
        if " / " in label:
            a_part, s_part = label.split(" / ", 1)
            if a_part == age_label:
                cell.append((s_part, values))
        elif label == age_label:
            cell.append(("Total", values))
    if not cell:
        # No age breakdown for this indicator — fall back to totals
        for label, values in all_series:
            if " / " not in label:
                cell.append((label, values))
    return cell


def _dp_aim_label(demo: str) -> str:
    return "Leapfrog DP/AIM" if demo == "Total" else f"Leapfrog DP/AIM {demo}"


def _goals_label(demo: str) -> str:
    return "Leapfrog Goals" if demo == "Total" else f"Leapfrog Goals {demo}"


def _spectrum_label(demo: str) -> str:
    return "Spectrum" if demo == "Total" else f"Spectrum {demo}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.script(src="https://cdn.plot.ly/plotly-latest.min.js")
    ),
    ui.page_sidebar(
        ui.sidebar(
            ui.h5("Filters"),
            ui.input_selectize(
                "pjnz",
                label="PJNZ",
                choices=_pjnz_stems,
                selected=_pjnz_stems[0] if _pjnz_stems else None,
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
            width=260,
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "All ages",
                ui.div(
                    ui.input_selectize(
                        "indicators_all",
                        label="Indicators",
                        choices=_ALL_AGES_NAMES,
                        multiple=True,
                        selected=_ALL_AGES_NAMES[:3],
                        options={"plugins": ["remove_button"]},
                    ),
                    ui.div(
                        ui.input_checkbox("disagg_age", "By age group", value=False),
                        ui.input_checkbox("disagg_sex_all", "By sex", value=False),
                        style="display: flex; gap: 24px; margin-top: 6px; margin-bottom: 6px;",
                    ),
                    style="padding: 10px 12px 4px 12px;",
                ),
                ui.div(
                    ui.output_ui("all_ages_plot"),
                    style="overflow-x: auto; overflow-y: auto;",
                ),
            ),
            ui.nav_panel(
                "15-49",
                ui.div(
                    ui.input_selectize(
                        "indicators_1549",
                        label="Indicators",
                        choices=_1549_NAMES,
                        multiple=True,
                        selected=_1549_NAMES,
                        options={"plugins": ["remove_button"]},
                    ),
                    ui.div(
                        ui.input_checkbox("disagg_sex_1549", "By sex", value=False),
                        style="margin-top: 6px; margin-bottom: 6px;",
                    ),
                    style="padding: 10px 12px 4px 12px;",
                ),
                ui.div(
                    ui.output_ui("plot_1549"),
                    style="overflow-x: auto; overflow-y: auto;",
                ),
            ),
            ui.nav_panel(
                "Risk groups",
                ui.div(
                    ui.div(
                        ui.input_checkbox("disagg_sex_rg", "By sex", value=False),
                        style="margin-top: 6px; margin-bottom: 6px;",
                    ),
                    style="padding: 10px 12px 4px 12px;",
                ),
                ui.div(
                    ui.output_ui("risk_groups_plot"),
                    style="overflow-x: auto; overflow-y: auto;",
                ),
            ),
            id="main_tabs",
            selected="15-49",
        ),
        title="Leapfrog Comparison",
        fillable=True,
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def server(input, output, session):

    @reactive.calc
    def _run_pjnz():
        """Returns (data, error_str). Exactly one of the two will be None."""
        pjnz_stem = input.pjnz()
        if not pjnz_stem or pjnz_stem not in _pjnz_files:
            return None, None
        try:
            return run_pjnz(_pjnz_files[pjnz_stem]), None
        except Exception as exc:
            print(f"[app] Failed to run {pjnz_stem}: {exc}")
            return None, str(exc)

    @reactive.effect
    def _update_year_slider():
        result, _ = _run_pjnz()
        if result is None:
            return
        _, _, output_years = result
        y_min, y_max = int(min(output_years)), int(max(output_years))
        ui.update_slider("year_range", min=y_min, max=y_max, value=[y_min, y_max])

    def _loading_ui(error: str | None):
        if error:
            return ui.div(
                ui.p(
                    f"Error running model for '{input.pjnz()}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
        msg = (
            "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'."
            if not _pjnz_stems else "Loading..."
        )
        return ui.p(msg)

    def _make_trace_helpers(fig, line_width: float):
        demo_colors: dict[str, str] = {}
        palette_idx = 0
        legend_shown: set[str] = set()

        def _color_for(demo: str) -> str:
            nonlocal palette_idx
            if demo not in demo_colors:
                demo_colors[demo] = _PALETTE[palette_idx % len(_PALETTE)]
                palette_idx += 1
            return demo_colors[demo]

        def add_trace(x, y, trace_name, demo, dash, row, col):
            show_leg = trace_name not in legend_shown
            if show_leg:
                legend_shown.add(trace_name)
            y_clean = [None if isinstance(v, float) and np.isnan(v) else v for v in y]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_clean,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=_color_for(demo), width=line_width, dash=dash),
                    legendgroup=trace_name,
                    showlegend=show_leg,
                ),
                row=row,
                col=col,
            )

        return add_trace

    # -----------------------------------------------------------------------
    # All-ages tab
    # -----------------------------------------------------------------------

    @render.ui
    def all_ages_plot():
        input.main_tabs()
        result, error = _run_pjnz()
        if result is None:
            return _loading_ui(error)

        modvars, goals_output, output_years = result
        selected = input.indicators_all()
        year_start, year_end = input.year_range()
        disagg_age = input.disagg_age()
        disagg_sex = input.disagg_sex_all()

        if not selected:
            return ui.p("Select at least one indicator.")

        years_arr = np.array(list(output_years))
        mask = (years_arr >= year_start) & (years_arr <= year_end)
        x_years = years_arr[mask].tolist()
        first_year = int(min(output_years))
        n_inds = len(selected)

        _age_label_set = set(AGE_LABELS)

        def _align_spec(spec_values: np.ndarray):
            year_idx = years_arr - first_year
            valid = (year_idx >= 0) & (year_idx < len(spec_values))
            combined = mask & valid
            return years_arr[combined].tolist(), spec_values[year_idx[combined].astype(int)].tolist()

        def _has_age_labels(series: list[tuple[str, np.ndarray]]) -> bool:
            for label, _ in series:
                part = label.split(" / ")[0] if " / " in label else label
                if part in _age_label_set:
                    return True
            return False

        # -------------------------------------------------------------------
        # Age-faceted layout: rows = indicators, cols = age groups
        # -------------------------------------------------------------------
        if disagg_age:
            ncols = _N_AGE_GROUPS
            fig_width = max(1600, ncols * 110)
            fig_height = max(300, n_inds * 220)

            fig = make_subplots(
                rows=n_inds,
                cols=ncols,
                row_titles=list(selected),
                column_titles=AGE_LABELS,
                shared_xaxes="columns",
                shared_yaxes=False,
                vertical_spacing=max(0.015, 0.25 / max(n_inds, 1)),
                horizontal_spacing=0.01,
            )
            add_trace = _make_trace_helpers(fig, line_width=1.5)

            for ind_idx, indicator in enumerate(selected):
                row = ind_idx + 1
                ind_def = ALL_AGES_INDICATORS[indicator]
                all_series = ind_def.compute_leapfrog(goals_output, True, disagg_sex)

                spec_all: list[tuple[str, np.ndarray]] = []
                if ind_def.compute_spectrum is not None:
                    try:
                        spec_all = ind_def.compute_spectrum(modvars, True, disagg_sex)
                    except Exception as exc:
                        print(f"[app] Spectrum disagg failed for {indicator}: {exc}")
                spec_has_ages = _has_age_labels(spec_all)

                for age_idx, age_label in enumerate(AGE_LABELS):
                    col = age_idx + 1
                    for demo, values in _series_for_age_cell(all_series, age_label):
                        add_trace(x_years, values[mask].tolist(), _dp_aim_label(demo), demo, None, row, col)
                    if spec_has_ages:
                        for demo, spec_values in _series_for_age_cell(spec_all, age_label):
                            spec_x, spec_y = _align_spec(spec_values)
                            if spec_x:
                                add_trace(spec_x, spec_y, _spectrum_label(demo), demo, "dash", row, col)

            fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor="#e5e5e5")
            fig.update_yaxes(
                showgrid=True, gridcolor="#e5e5e5", rangemode="tozero",
                tickfont=dict(size=8),
            )
            for col in range(1, ncols + 1):
                fig.update_xaxes(
                    showticklabels=True, tickformat="d", tickangle=90,
                    tickfont=dict(size=8), row=n_inds, col=col,
                )
            fig.update_layout(
                width=fig_width,
                height=fig_height,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title_text=f"Comparison — {input.pjnz()}",
                margin=dict(t=80, b=60, l=60, r=120),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

        # -------------------------------------------------------------------
        # Simple layout: one column, rows = indicators
        # -------------------------------------------------------------------
        else:
            fig_height = max(400, n_inds * 300)
            fig = make_subplots(
                rows=n_inds,
                cols=1,
                subplot_titles=list(selected),
                shared_xaxes=False,
                vertical_spacing=max(0.04, 0.3 / max(n_inds, 1)),
            )
            add_trace = _make_trace_helpers(fig, line_width=2)

            for ind_idx, indicator in enumerate(selected):
                row = ind_idx + 1
                ind_def = ALL_AGES_INDICATORS[indicator]

                for demo, values in ind_def.compute_leapfrog(goals_output, False, disagg_sex):
                    add_trace(x_years, values[mask].tolist(), _dp_aim_label(demo), demo, None, row, 1)

                if ind_def.compute_spectrum is not None:
                    try:
                        for demo, spec_values in ind_def.compute_spectrum(modvars, False, disagg_sex):
                            spec_x, spec_y = _align_spec(spec_values)
                            if spec_x:
                                add_trace(spec_x, spec_y, _spectrum_label(demo), demo, "dash", row, 1)
                    except Exception as exc:
                        print(f"[app] Spectrum failed for {indicator}: {exc}")

            fig.update_xaxes(
                showgrid=True, gridcolor="#e5e5e5",
                range=[year_start - 1, year_end + 1],
                tickformat="d", tickangle=45,
            )
            fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")
            fig.update_layout(
                height=fig_height,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title_text=f"Comparison — {input.pjnz()}",
                margin=dict(t=80, b=40, l=60, r=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))

    # -----------------------------------------------------------------------
    # 15-49 tab
    # -----------------------------------------------------------------------

    @render.ui
    def plot_1549():
        input.main_tabs()
        result, error = _run_pjnz()
        if result is None:
            return _loading_ui(error)

        modvars, goals_output, output_years = result
        selected = input.indicators_1549()
        year_start, year_end = input.year_range()
        disagg_sex = input.disagg_sex_1549()

        if not selected:
            return ui.p("Select at least one indicator.")

        years_arr = np.array(list(output_years))
        mask = (years_arr >= year_start) & (years_arr <= year_end)
        x_years = years_arr[mask].tolist()
        first_year = int(min(output_years))
        n_inds = len(selected)

        def _align_spec(spec_values: np.ndarray):
            year_idx = years_arr - first_year
            valid = (year_idx >= 0) & (year_idx < len(spec_values))
            combined = mask & valid
            return years_arr[combined].tolist(), spec_values[year_idx[combined].astype(int)].tolist()

        fig_height = max(400, n_inds * 300)
        fig = make_subplots(
            rows=n_inds,
            cols=1,
            subplot_titles=list(selected),
            shared_xaxes=False,
            vertical_spacing=max(0.04, 0.3 / max(n_inds, 1)),
        )
        add_trace = _make_trace_helpers(fig, line_width=2)

        for ind_idx, indicator in enumerate(selected):
            row = ind_idx + 1
            ind_def = INDICATORS_1549[indicator]

            for demo, values in ind_def.compute_leapfrog(goals_output, disagg_sex):
                add_trace(x_years, values[mask].tolist(), _dp_aim_label(demo), demo, None, row, 1)

            if ind_def.compute_spectrum is not None:
                try:
                    for demo, spec_values in ind_def.compute_spectrum(modvars, disagg_sex):
                        spec_x, spec_y = _align_spec(spec_values)
                        if spec_x:
                            add_trace(spec_x, spec_y, _spectrum_label(demo), demo, "dash", row, 1)
                except Exception as exc:
                    print(f"[app] Spectrum failed for {indicator}: {exc}")

            if ind_def.compute_goals is not None:
                try:
                    for demo, values in ind_def.compute_goals(goals_output, disagg_sex):
                        add_trace(x_years, values[mask].tolist(), _goals_label(demo), demo, "dot", row, 1)
                except Exception as exc:
                    print(f"[app] Goals failed for {indicator}: {exc}")

        fig.update_xaxes(
            showgrid=True, gridcolor="#e5e5e5",
            range=[year_start - 1, year_end + 1],
            tickformat="d", tickangle=45,
        )
        fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")
        fig.update_layout(
            height=fig_height,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_text=f"Comparison — {input.pjnz()}",
            margin=dict(t=80, b=40, l=60, r=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))

    # -----------------------------------------------------------------------
    # Risk groups tab
    # -----------------------------------------------------------------------

    @render.ui
    def risk_groups_plot():
        input.main_tabs()
        result, error = _run_pjnz()
        if result is None:
            return _loading_ui(error)

        modvars, goals_output, output_years = result
        year_start, year_end = input.year_range()
        disagg_sex = input.disagg_sex_rg()

        years_arr = np.array(list(output_years))
        mask = (years_arr >= year_start) & (years_arr <= year_end)
        x_years = years_arr[mask].tolist()
        first_year = int(min(output_years))
        n_rg = len(RISK_GROUPS)
        rg_row = {rg_name: i + 1 for i, (rg_name, _rg) in enumerate(RISK_GROUPS)}

        def _align_spec(spec_values: np.ndarray):
            year_idx = years_arr - first_year
            valid = (year_idx >= 0) & (year_idx < len(spec_values))
            combined = mask & valid
            return years_arr[combined].tolist(), spec_values[year_idx[combined].astype(int)].tolist()

        fig = make_subplots(
            rows=n_rg,
            cols=1,
            subplot_titles=[rg_name for rg_name, _ in RISK_GROUPS],
            shared_xaxes=False,
            vertical_spacing=max(0.04, 0.3 / max(n_rg, 1)),
        )
        add_trace = _make_trace_helpers(fig, line_width=2)

        try:
            for rg_name, demo, values in compute_rg_goals(goals_output, disagg_sex):
                add_trace(
                    x_years, values[mask].tolist(),
                    _goals_label(demo), demo, None, rg_row[rg_name], 1,
                )
        except Exception as exc:
            print(f"[app] Risk groups Goals failed: {exc}")

        try:
            for rg_name, demo, spec_values in compute_rg_spectrum(modvars, disagg_sex):
                spec_x, spec_y = _align_spec(spec_values)
                if spec_x:
                    add_trace(
                        spec_x, spec_y,
                        _spectrum_label(demo), demo, "dash", rg_row[rg_name], 1,
                    )
        except Exception as exc:
            print(f"[app] Risk groups Spectrum failed: {exc}")

        fig.update_xaxes(
            showgrid=True, gridcolor="#e5e5e5",
            range=[year_start - 1, year_end + 1],
            tickformat="d", tickangle=45,
        )
        fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5", rangemode="tozero")
        fig.update_layout(
            height=max(500, n_rg * 250),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            title_text=f"Risk groups — {input.pjnz()}",
            margin=dict(t=80, b=40, l=60, r=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False))


app = App(app_ui, server)
