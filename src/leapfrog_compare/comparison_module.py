"""
Reusable Shiny modules for a comparison tab made of a shared data panel (PJNZ +
year-range selector, one per top-level tab) and any number of plot panels
(indicator selector + disaggregation checkboxes + rendered plot, one per inner
sub-tab), all reading from the SAME underlying data load. This lets a
top-level tab expose an arbitrary, easily-extended set of inner "plot type"
sub-tabs (e.g. "All ages", "15-49", "Risk groups", ...) without re-selecting
the PJNZ or re-running the model per sub-tab.

Typical wiring for one top-level tab with two sub-tabs:

    data_run, year_range = data_panel_server("goals_data", pjnz_files=..., run_fn=...)
    plot_panel_server("goals_allages", data_run=data_run, year_range=year_range,
                       indicator_map=..., sources=..., age_labels=...)
    plot_panel_server("goals_1549", data_run=data_run, year_range=year_range,
                       indicator_map=..., sources=..., age_labels=..., show_age_checkbox=False)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from shiny import module, reactive, render, ui

from leapfrog_compare.indicator_map import cd4_facet_desc
from leapfrog_compare.plotting import (
    ComparisonSource, default_visible_keys, render_age_profile, render_averted_barchart,
    render_comparison, render_multi_pjnz_comparison, render_multi_risk_group_comparison,
    render_risk_group_comparison, visible_sources,
)

# (data_by_source, output_years)
RunFn = Callable[[Path], tuple[dict[str, Any], range]]
# (data_by_source, output_years), takes a `force` kwarg to bypass any cache
RunFnWithForce = Callable[..., tuple[dict[str, Any], range]]
# Reactive getters (e.g. a module-level @reactive.poll/@reactive.calc in app.py)
# so newly added/removed/edited PJNZ files in PJNZ_DIR are picked up live,
# without restarting the app.
PjnzFilesGetter = Callable[[], dict[str, Path]]
PjnzChoicesGetter = Callable[[], "list[str] | dict[str, str]"]
PjnzStemsGetter = Callable[[], list[str]]


# ---------------------------------------------------------------------------
# Data panel: PJNZ selector + year range, shared across a top-level tab's sub-tabs
# ---------------------------------------------------------------------------

@module.ui
def data_panel_ui(
    *,
    pjnz_choices: list[str] | dict[str, str],
    year_min: int,
    year_max: int,
    show_rerun_button: bool = False,
):
    children = [
        ui.h5("Filters"),
        ui.input_selectize(
            "pjnz",
            label="PJNZ",
            choices=pjnz_choices,
            selected=next(iter(pjnz_choices), None),
        ),
        ui.hr(),
        ui.input_slider(
            "year_range",
            "Year range",
            min=year_min,
            max=year_max,
            value=[year_min, year_max],
            step=1,
            sep="",
        ),
    ]
    if show_rerun_button:
        children += [ui.hr(), ui.input_action_button("rerun", "Re-run models")]
    return ui.sidebar(*children, width=320)


@module.server
def data_panel_server(
    input,
    output,
    session,
    *,
    pjnz_files: PjnzFilesGetter,
    pjnz_choices: PjnzChoicesGetter,
    run_fn: RunFn | RunFnWithForce,
    show_rerun_button: bool = False,
):
    """Returns (data_run, year_range, pjnz_label) — plain callables, each
    readable from any other module's server function to establish the same
    reactive dependency as if called locally.

    `pjnz_files`/`pjnz_choices` are reactive getters (backed by a shared
    @reactive.poll in app.py) rather than plain dicts/lists, so that files
    added/removed/edited in PJNZ_DIR are picked up without restarting the app:
    the PJNZ dropdown's choices refresh live, and `data_run` re-runs the model
    for the current selection whenever the underlying file changes on disk."""
    _last_seen_rerun_clicks = 0

    @reactive.calc
    def data_run():
        """Returns (result, error_str). Exactly one of the two will be None."""
        nonlocal _last_seen_rerun_clicks
        pjnz_stem = input.pjnz()
        files = pjnz_files()
        if not pjnz_stem or pjnz_stem not in files:
            return None, None

        force = False
        if show_rerun_button:
            clicks = input.rerun()
            if clicks > _last_seen_rerun_clicks:
                force = True
            _last_seen_rerun_clicks = clicks

        try:
            if show_rerun_button:
                return run_fn(files[pjnz_stem], force=force), None
            return run_fn(files[pjnz_stem]), None
        except Exception as exc:
            print(f"[comparison_module] Failed to run {pjnz_stem}: {exc}")
            return None, str(exc)

    @reactive.effect
    def _update_year_slider():
        result, _ = data_run()
        if result is None:
            return
        _, output_years = result
        y_min, y_max = int(min(output_years)), int(max(output_years))
        ui.update_slider("year_range", min=y_min, max=y_max, value=[y_min, y_max])

    @reactive.effect
    def _refresh_pjnz_choices():
        """Re-fires whenever pjnz_choices() changes (i.e. PJNZ_DIR's poll detects
        an added/removed/edited file), pushing the new dropdown choices to the
        client. Preserves the current selection if it's still valid, otherwise
        falls back to the first available choice."""
        choices = pjnz_choices()
        keys = list(choices.keys()) if isinstance(choices, dict) else list(choices)
        with reactive.isolate():
            current = input.pjnz()
        selected = current if current in keys else (keys[0] if keys else None)
        ui.update_selectize("pjnz", choices=choices, selected=selected)

    def year_range():
        return input.year_range()

    def pjnz_label():
        return input.pjnz()

    return data_run, year_range, pjnz_label


# ---------------------------------------------------------------------------
# Plot panel: indicator selector + disaggregation checkboxes + rendered plot
# ---------------------------------------------------------------------------

@module.ui
def plot_panel_ui(
    *,
    indicator_names: list[str],
    default_indicators: list[str],
    show_age_checkbox: bool = True,
):
    disagg_children = []
    if show_age_checkbox:
        disagg_children.append(ui.input_checkbox("disagg_age", "By age group", value=False))
    disagg_children.append(ui.input_checkbox("disagg_sex", "By sex", value=False))

    return ui.div(
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=indicator_names,
            multiple=True,
            selected=default_indicators,
            options={"plugins": ["remove_button"]},
        ),
        ui.div(*disagg_children, style="display: flex; gap: 20px; margin: 6px 0 12px 0;"),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def plot_panel_server(
    input,
    output,
    session,
    *,
    data_run: Callable[[], tuple],
    year_range: Callable[[], tuple[int, int]],
    pjnz_label: Callable[[], str],
    indicator_map: dict[str, Any],
    sources: list[ComparisonSource],
    age_labels: list[str],
    show_age_checkbox: bool = True,
    no_pjnz_message: str = "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'.",
):
    @output
    @render.ui
    def comparison_plot():
        result, error = data_run()
        if result is None:
            if error:
                return ui.div(
                    ui.p(
                        f"Error running model for '{pjnz_label()}':",
                        style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                    ),
                    ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
                )
            return ui.p(no_pjnz_message)

        data_by_source, output_years = result
        selected_indicators = input.indicators()
        year_start, year_end = year_range()
        disagg_age = input.disagg_age() if show_age_checkbox else False
        disagg_sex = input.disagg_sex()

        if not selected_indicators:
            return ui.p("Select at least one indicator.")

        html = render_comparison(
            indicator_map=indicator_map,
            data_by_source=data_by_source,
            sources=sources,
            selected_indicators=selected_indicators,
            output_years=output_years,
            year_start=year_start,
            year_end=year_end,
            disagg_age=disagg_age,
            disagg_sex=disagg_sex,
            title=f"Comparison — {pjnz_label()}",
            age_labels=age_labels,
        )
        return ui.HTML(html)


# ---------------------------------------------------------------------------
# Resource-needs panel: like plot_panel, but with a single-select "Dataset"
# dropdown that switches which indicator_map is used (leapfrog Goals'
# num_people_reached vs resources_required, each with its own map keyed by the
# same intervention names) ahead of the usual indicator multiselect. No age/sex
# disaggregation controls — resource-needs arrays carry neither axis. Used by
# the Goals tab's "Resource needs" sub-tab.
# ---------------------------------------------------------------------------

@module.ui
def resource_needs_panel_ui(
    *,
    dataset_labels: list[str],
    indicator_names: list[str],
    default_indicators: list[str],
):
    return ui.div(
        ui.input_select(
            "dataset",
            label="Dataset",
            choices=dataset_labels,
            selected=dataset_labels[0] if dataset_labels else None,
        ),
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=indicator_names,
            multiple=True,
            selected=default_indicators,
            options={"plugins": ["remove_button"]},
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def resource_needs_panel_server(
    input,
    output,
    session,
    *,
    data_run: Callable[[], tuple],
    year_range: Callable[[], tuple[int, int]],
    pjnz_label: Callable[[], str],
    dataset_maps: dict[str, dict[str, Any]],
    sources: list[ComparisonSource],
    no_pjnz_message: str = "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'.",
):
    @output
    @render.ui
    def comparison_plot():
        result, error = data_run()
        if result is None:
            if error:
                return ui.div(
                    ui.p(
                        f"Error running model for '{pjnz_label()}':",
                        style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                    ),
                    ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
                )
            return ui.p(no_pjnz_message)

        data_by_source, output_years = result
        dataset = input.dataset()
        indicator_map = dataset_maps.get(dataset) or next(iter(dataset_maps.values()))
        selected_indicators = input.indicators()
        year_start, year_end = year_range()

        if not selected_indicators:
            return ui.p("Select at least one indicator.")

        html = render_comparison(
            indicator_map=indicator_map,
            data_by_source=data_by_source,
            sources=sources,
            selected_indicators=selected_indicators,
            output_years=output_years,
            year_start=year_start,
            year_end=year_end,
            disagg_age=False,
            disagg_sex=False,
            title=f"{dataset} — {pjnz_label()}",
            age_labels=[],
        )
        return ui.HTML(html)


# ---------------------------------------------------------------------------
# Risk-group panel: a single-select Indicator dropdown (choosing between
# risk-group-faceted indicators, e.g. population share vs. new infections,
# each with its own compute_fns/title/units — see indicator_map.RiskGroupIndicatorDef)
# + a "By sex" checkbox + a one-row-per-risk-group rendered plot. Used by the
# Goals tab's "Risk groups" sub-tab.
# ---------------------------------------------------------------------------

@module.ui
def risk_group_panel_ui(*, indicator_names: list[str]):
    return ui.div(
        ui.input_selectize(
            "indicator",
            label="Indicator",
            choices=indicator_names,
            selected=indicator_names[0] if indicator_names else None,
        ),
        ui.div(
            ui.input_checkbox("disagg_sex", "By sex", value=False),
            style="margin: 6px 0 12px 0;",
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def risk_group_panel_server(
    input,
    output,
    session,
    *,
    data_run: Callable[[], tuple],
    year_range: Callable[[], tuple[int, int]],
    pjnz_label: Callable[[], str],
    risk_groups: list[tuple[str, int]],
    sources: list[ComparisonSource],
    # Indicator name -> object with `.compute_fns`, `.title_prefix`, `.y_title`
    # (duck-typed to indicator_map.RiskGroupIndicatorDef).
    indicator_map: dict[str, Any],
    no_pjnz_message: str = "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'.",
):
    @output
    @render.ui
    def comparison_plot():
        result, error = data_run()
        if result is None:
            if error:
                return ui.div(
                    ui.p(
                        f"Error running model for '{pjnz_label()}':",
                        style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                    ),
                    ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
                )
            return ui.p(no_pjnz_message)

        indicator = input.indicator()
        if not indicator or indicator not in indicator_map:
            return ui.p("Select an indicator.")

        ind_def = indicator_map[indicator]
        data_by_source, output_years = result
        year_start, year_end = year_range()
        disagg_sex = input.disagg_sex()

        html = render_risk_group_comparison(
            risk_groups=risk_groups,
            sources=sources,
            data_by_source=data_by_source,
            compute_fns=ind_def.compute_fns,
            output_years=output_years,
            year_start=year_start,
            year_end=year_end,
            disagg_sex=disagg_sex,
            title=f"{ind_def.title_prefix} — {pjnz_label()}",
            y_title=ind_def.y_title,
        )
        return ui.HTML(html)


# ---------------------------------------------------------------------------
# Age-profile panel: indicator multiselect (same pattern as plot_panel_ui) +
# a year dropdown (instead of the shared year-range slider) driving a rendered
# plot with single-year-of-age on the x-axis and one line per (source, sex),
# one row per selected indicator — see plotting.render_age_profile. Used by
# the AIM/Goals tabs' "By age" sub-tabs.
# ---------------------------------------------------------------------------

@module.ui
def age_profile_panel_ui(
    *,
    indicator_names: list[str],
    default_indicators: list[str],
    year_min: int,
    year_max: int,
):
    years = [str(y) for y in range(year_min, year_max + 1)]
    return ui.div(
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=indicator_names,
            multiple=True,
            selected=default_indicators,
            options={"plugins": ["remove_button"]},
        ),
        ui.input_selectize(
            "year",
            label="Year",
            choices=years,
            selected=years[-1] if years else None,
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def age_profile_panel_server(
    input,
    output,
    session,
    *,
    data_run: Callable[[], tuple],
    pjnz_label: Callable[[], str],
    indicator_map: dict[str, Any],
    sources: list[ComparisonSource],
    title_prefix: str = "By age",
    no_pjnz_message: str = "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'.",
):
    @reactive.effect
    def _update_year_choices():
        result, _ = data_run()
        if result is None:
            return
        _, output_years = result
        years = [str(y) for y in output_years]
        with reactive.isolate():
            current = input.year()
        selected = current if current in years else (years[-1] if years else None)
        ui.update_selectize("year", choices=years, selected=selected)

    @output
    @render.ui
    def comparison_plot():
        result, error = data_run()
        if result is None:
            if error:
                return ui.div(
                    ui.p(
                        f"Error running model for '{pjnz_label()}':",
                        style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                    ),
                    ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
                )
            return ui.p(no_pjnz_message)

        selected_indicators = input.indicators()
        year_str = input.year()
        if not selected_indicators:
            return ui.p("Select at least one indicator.")
        if not year_str:
            return ui.p("Select a year.")

        data_by_source, output_years = result
        html = render_age_profile(
            indicator_map=indicator_map,
            data_by_source=data_by_source,
            sources=sources,
            selected_indicators=selected_indicators,
            output_years=output_years,
            year=int(year_str),
            title=f"{title_prefix} — {year_str} — {pjnz_label()}",
        )
        return ui.HTML(html)


# ---------------------------------------------------------------------------
# Facet panel: indicator dropdown (single-select) + "By sex" checkbox + a
# one-row-per-facet-group rendered plot (render_risk_group_comparison), where
# the facet group is CD4 stage rather than risk group and which indicator's
# compute_fns/labels are active is chosen dynamically via the dropdown. Used
# by the AIM/Goals tabs' "0-14" sub-tabs.
# ---------------------------------------------------------------------------

@module.ui
def facet_panel_ui(*, indicator_names: list[str]):
    return ui.div(
        ui.input_selectize(
            "indicator",
            label="Indicator",
            choices=indicator_names,
            selected=indicator_names[0] if indicator_names else None,
        ),
        ui.div(
            ui.input_checkbox("disagg_sex", "By sex", value=False),
            style="margin: 6px 0 12px 0;",
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def facet_panel_server(
    input,
    output,
    session,
    *,
    data_run: Callable[[], tuple],
    year_range: Callable[[], tuple[int, int]],
    pjnz_label: Callable[[], str],
    # Indicator name -> object with `.cd4_labels: list[str]` and
    # `.compute_fns: dict[str, Callable]` (duck-typed to
    # indicator_map.ChildCD4IndicatorDef).
    facet_map: dict[str, Any],
    sources: list[ComparisonSource],
    title_prefix: str,
    no_pjnz_message: str = "No PJNZ files found, check 'PJNZ_DIR' in 'config.py'.",
):
    @output
    @render.ui
    def comparison_plot():
        result, error = data_run()
        if result is None:
            if error:
                return ui.div(
                    ui.p(
                        f"Error running model for '{pjnz_label()}':",
                        style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                    ),
                    ui.pre(error, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
                )
            return ui.p(no_pjnz_message)

        indicator = input.indicator()
        if not indicator or indicator not in facet_map:
            return ui.p("Select an indicator.")

        data_by_source, output_years = result
        year_start, year_end = year_range()
        ind_def = facet_map[indicator]

        facet_desc = cd4_facet_desc(ind_def.cd4_labels)

        html = render_risk_group_comparison(
            risk_groups=[(lbl, i) for i, lbl in enumerate(ind_def.cd4_labels)],
            sources=sources,
            data_by_source=data_by_source,
            compute_fns=ind_def.compute_fns,
            output_years=output_years,
            year_start=year_start,
            year_end=year_end,
            disagg_sex=input.disagg_sex(),
            title=f"{title_prefix} {facet_desc} — {indicator} — {pjnz_label()}",
        )
        return ui.HTML(html)


# ---------------------------------------------------------------------------
# Multi PJNZ data panel: a multi-select PJNZ picker + a Goals/DP-AIM "Model"
# switch (determines both the file pool and which run function processes each
# selected file) + a year-range slider spanning the UNION of all selected
# files' own year ranges. Unlike `data_panel_ui`/`data_panel_server` (one
# PJNZ, shared across a top-level tab's sub-tabs), this drives the dedicated
# "Multi PJNZ" top-level tab, whose whole point is comparing 2+ files at once.
#
# Each selected file's model run is cached in memory for the session, keyed
# by stem and invalidated by an (mtime_ns, size) fingerprint check (same idea
# as app.py's `_pjnz_fingerprint`, applied per-file) — so switching a file out
# of the selection and back in, or re-selecting the same file after switching
# Model and back, does not re-run its model.
# ---------------------------------------------------------------------------

@module.ui
def multi_pjnz_panel_ui(
    *,
    goals_choices: list[str],
    year_min: int,
    year_max: int,
):
    return ui.sidebar(
        ui.h5("Filters"),
        ui.input_radio_buttons(
            "model", "Model", choices=["Goals", "DP/AIM"], selected="Goals",
        ),
        ui.input_selectize(
            "pjnz",
            label="PJNZ files",
            choices=goals_choices,
            multiple=True,
            selected=goals_choices[:1],
            options={"plugins": ["remove_button"]},
        ),
        ui.hr(),
        ui.input_slider(
            "year_range",
            "Year range",
            min=year_min,
            max=year_max,
            value=[year_min, year_max],
            step=1,
            sep="",
        ),
        width=320,
    )


@module.server
def multi_pjnz_panel_server(
    input,
    output,
    session,
    *,
    pjnz_files: PjnzFilesGetter,
    pjnz_stems_goals: PjnzStemsGetter,
    pjnz_stems_aim: PjnzStemsGetter,
    goals_run_fn: RunFn,
    aim_run_fn: RunFn,
):
    """Returns (data_by_pjnz, year_range, model) — plain callables, same pattern as
    `data_panel_server`.

    `data_by_pjnz()` returns `(data, errors)`:
      - `data`: dict[stem, (data_by_source, output_years)] for every currently
        selected file whose model run succeeded (or was served from cache).
      - `errors`: dict[stem, str] for every currently selected file whose run raised.
    """
    _cache: dict[str, tuple[tuple[int, int], tuple[dict[str, Any], range]]] = {}

    def _current_stems() -> list[str]:
        return pjnz_stems_goals() if input.model() == "Goals" else pjnz_stems_aim()

    def _current_run_fn() -> RunFn:
        return goals_run_fn if input.model() == "Goals" else aim_run_fn

    def _run_cached(stem: str, path: Path) -> tuple[dict[str, Any], range]:
        stat = path.stat()
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        cached = _cache.get(stem)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        result = _current_run_fn()(path)
        _cache[stem] = (fingerprint, result)
        return result

    @reactive.effect
    def _refresh_pjnz_choices():
        """Re-fires whenever the Model switch changes, or the active pool's file
        listing changes. Preserves selections still valid in the new pool; when
        switching Model, the previous selection is never valid in the new pool
        (Goals-classified and AIM-only stems are disjoint), so this naturally
        auto-selects the first file in the new pool."""
        stems = _current_stems()
        with reactive.isolate():
            current = list(input.pjnz() or ())
        still_valid = [s for s in current if s in stems]
        selected = still_valid if still_valid else (stems[:1] if stems else [])
        ui.update_selectize("pjnz", choices=stems, selected=selected)

    @reactive.calc
    def data_by_pjnz():
        files = pjnz_files()
        selected = input.pjnz() or ()
        data: dict[str, tuple[dict[str, Any], range]] = {}
        errors: dict[str, str] = {}
        for stem in selected:
            path = files.get(stem)
            if path is None:
                continue
            try:
                data[stem] = _run_cached(stem, path)
            except Exception as exc:
                print(f"[comparison_module] Failed to run {stem}: {exc}")
                errors[stem] = str(exc)
        return data, errors

    @reactive.effect
    def _update_year_slider():
        data, _errors = data_by_pjnz()
        if not data:
            return
        all_years = [y for _src, output_years in data.values() for y in (min(output_years), max(output_years))]
        y_min, y_max = min(all_years), max(all_years)
        ui.update_slider("year_range", min=y_min, max=y_max, value=[y_min, y_max])

    def year_range():
        return input.year_range()

    def model():
        return input.model()

    return data_by_pjnz, year_range, model


# ---------------------------------------------------------------------------
# Multi PJNZ plot panel: indicator multiselect (same pattern as plot_panel_ui)
# + a rendered plot with one line per (PJNZ file, source) — see
# plotting.render_multi_pjnz_comparison. No age/sex disaggregation controls
# (v1 is totals-only, see ADR-0003) — colour is reserved for the PJNZ file.
# ---------------------------------------------------------------------------

@module.ui
def multi_plot_panel_ui(
    *,
    indicator_names: list[str],
    default_indicators: list[str],
    initial_sources: list[ComparisonSource],
):
    return ui.div(
        ui.input_checkbox_group(
            "visible_sources",
            "Show lines",
            choices={s.key: s.label for s in initial_sources},
            selected=default_visible_keys(initial_sources),
        ),
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=indicator_names,
            multiple=True,
            selected=default_indicators,
            options={"plugins": ["remove_button"]},
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def multi_plot_panel_server(
    input,
    output,
    session,
    *,
    data_by_pjnz: Callable[[], tuple[dict[str, tuple], dict[str, str]]],
    year_range: Callable[[], tuple[int, int]],
    indicator_map: dict[str, Any],
    sources: Callable[[], list[ComparisonSource]],
    no_pjnz_message: str = "Select at least one PJNZ file.",
):
    @reactive.effect
    def _refresh_source_choices():
        """Re-fires whenever the Model switch changes `sources()`'s value (Goals
        vs DP/AIM have different source keys/labels) — same trigger pattern as
        `_refresh_pjnz_choices`. Resets the checkbox selection back to
        `default_visible`, mirroring how the PJNZ selection also resets on the
        same switch."""
        srcs = sources()
        ui.update_checkbox_group(
            "visible_sources",
            choices={s.key: s.label for s in srcs},
            selected=default_visible_keys(srcs),
        )

    @output
    @render.ui
    def comparison_plot():
        data, errors = data_by_pjnz()

        error_banner = [
            ui.div(
                ui.p(
                    f"Error running model for '{stem}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(msg, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
            for stem, msg in errors.items()
        ]

        if not data:
            if error_banner:
                return ui.div(*error_banner)
            return ui.p(no_pjnz_message)

        selected_indicators = input.indicators()
        if not selected_indicators:
            return ui.div(*error_banner, ui.p("Select at least one indicator."))

        year_start, year_end = year_range()
        stems = list(data.keys())
        data_by_source_map = {stem: result[0] for stem, result in data.items()}
        output_years_by_pjnz = {stem: result[1] for stem, result in data.items()}

        active_sources = visible_sources(sources(), input.visible_sources() or ())
        fig = render_multi_pjnz_comparison(
            indicator_map=indicator_map,
            data_by_pjnz=data_by_source_map,
            output_years_by_pjnz=output_years_by_pjnz,
            sources=active_sources,
            selected_indicators=selected_indicators,
            pjnz_stems=stems,
            year_start=year_start,
            year_end=year_end,
            title="Multi PJNZ comparison",
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.div(*error_banner, ui.HTML(html))


# ---------------------------------------------------------------------------
# Multi PJNZ resource-needs panel: multi_plot_panel plus the single-select
# "Dataset" dropdown from resource_needs_panel_ui — one line per (PJNZ file,
# source) per intervention, for whichever of num_people_reached /
# resources_required is selected. Used by the Multi PJNZ tab's "Resource needs"
# sub-tab. Always wired with _RESOURCE_NEEDS_SOURCES (goals/spectrum): resource
# needs only exist for the Goals model, so under Model=DP/AIM these render
# nothing, same missing-key-skip convention as the Multi risk-group sub-tab.
# ---------------------------------------------------------------------------

@module.ui
def multi_resource_needs_panel_ui(
    *,
    dataset_labels: list[str],
    indicator_names: list[str],
    default_indicators: list[str],
    initial_sources: list[ComparisonSource],
):
    return ui.div(
        ui.input_checkbox_group(
            "visible_sources",
            "Show lines",
            choices={s.key: s.label for s in initial_sources},
            selected=default_visible_keys(initial_sources),
        ),
        ui.input_select(
            "dataset",
            label="Dataset",
            choices=dataset_labels,
            selected=dataset_labels[0] if dataset_labels else None,
        ),
        ui.input_selectize(
            "indicators",
            label="Indicators",
            choices=indicator_names,
            multiple=True,
            selected=default_indicators,
            options={"plugins": ["remove_button"]},
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def multi_resource_needs_panel_server(
    input,
    output,
    session,
    *,
    data_by_pjnz: Callable[[], tuple[dict[str, tuple], dict[str, str]]],
    year_range: Callable[[], tuple[int, int]],
    dataset_maps: dict[str, dict[str, Any]],
    sources: list[ComparisonSource],
    no_pjnz_message: str = "Select at least one PJNZ file.",
):
    @output
    @render.ui
    def comparison_plot():
        data, errors = data_by_pjnz()

        error_banner = [
            ui.div(
                ui.p(
                    f"Error running model for '{stem}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(msg, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
            for stem, msg in errors.items()
        ]

        if not data:
            if error_banner:
                return ui.div(*error_banner)
            return ui.p(no_pjnz_message)

        selected_indicators = input.indicators()
        if not selected_indicators:
            return ui.div(*error_banner, ui.p("Select at least one indicator."))

        dataset = input.dataset()
        indicator_map = dataset_maps.get(dataset) or next(iter(dataset_maps.values()))
        year_start, year_end = year_range()
        stems = list(data.keys())
        data_by_source_map = {stem: result[0] for stem, result in data.items()}
        output_years_by_pjnz = {stem: result[1] for stem, result in data.items()}

        active_sources = visible_sources(sources, input.visible_sources() or ())
        fig = render_multi_pjnz_comparison(
            indicator_map=indicator_map,
            data_by_pjnz=data_by_source_map,
            output_years_by_pjnz=output_years_by_pjnz,
            sources=active_sources,
            selected_indicators=selected_indicators,
            pjnz_stems=stems,
            year_start=year_start,
            year_end=year_end,
            title=f"{dataset} — Multi PJNZ comparison",
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.div(*error_banner, ui.HTML(html))


# ---------------------------------------------------------------------------
# Multi PJNZ risk-group panel: a single-select Indicator dropdown (same
# indicator_map as the single-PJNZ risk_group_panel above; no "By sex"
# checkbox — v1 multi-file plots are totals-only per ADR-0003) driving a
# rendered plot with one line per (PJNZ file, source) pair per row — see
# plotting.render_multi_risk_group_comparison. Used by the Multi PJNZ tab's
# "Risk groups" sub-tab.
# ---------------------------------------------------------------------------

@module.ui
def multi_risk_group_panel_ui(*, sources: list[ComparisonSource], indicator_names: list[str]):
    return ui.div(
        ui.input_checkbox_group(
            "visible_sources",
            "Show lines",
            choices={s.key: s.label for s in sources},
            selected=default_visible_keys(sources),
        ),
        ui.input_selectize(
            "indicator",
            label="Indicator",
            choices=indicator_names,
            selected=indicator_names[0] if indicator_names else None,
        ),
        ui.output_ui("comparison_plot"),
        style="overflow-x: auto; overflow-y: auto; padding-top: 12px;",
    )


@module.server
def multi_risk_group_panel_server(
    input,
    output,
    session,
    *,
    data_by_pjnz: Callable[[], tuple[dict[str, tuple], dict[str, str]]],
    year_range: Callable[[], tuple[int, int]],
    risk_groups: list[tuple[str, int]],
    sources: list[ComparisonSource],
    # Indicator name -> object with `.compute_fns`, `.title_prefix`, `.y_title`
    # (duck-typed to indicator_map.RiskGroupIndicatorDef).
    indicator_map: dict[str, Any],
    no_pjnz_message: str = "Select at least one PJNZ file.",
):
    @output
    @render.ui
    def comparison_plot():
        data, errors = data_by_pjnz()

        error_banner = [
            ui.div(
                ui.p(
                    f"Error running model for '{stem}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(msg, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
            for stem, msg in errors.items()
        ]

        if not data:
            if error_banner:
                return ui.div(*error_banner)
            return ui.p(no_pjnz_message)

        indicator = input.indicator()
        if not indicator or indicator not in indicator_map:
            return ui.div(*error_banner, ui.p("Select an indicator."))

        ind_def = indicator_map[indicator]
        year_start, year_end = year_range()
        stems = list(data.keys())
        data_by_source_map = {stem: result[0] for stem, result in data.items()}
        output_years_by_pjnz = {stem: result[1] for stem, result in data.items()}

        active_sources = visible_sources(sources, input.visible_sources() or ())
        fig = render_multi_risk_group_comparison(
            risk_groups=risk_groups,
            sources=active_sources,
            data_by_pjnz=data_by_source_map,
            output_years_by_pjnz=output_years_by_pjnz,
            compute_fns=ind_def.compute_fns,
            pjnz_stems=stems,
            year_start=year_start,
            year_end=year_end,
            title=f"{ind_def.title_prefix} — Multi PJNZ comparison",
            y_title=ind_def.y_title,
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.div(*error_banner, ui.HTML(html))


# ---------------------------------------------------------------------------
# Averted panel: baseline-PJNZ selector + indicator dropdown, embedded in the
# panel body, driving a grouped bar chart of (baseline total - this PJNZ's
# total) per selected PJNZ file, aggregated over age/sex/year — see
# plotting.render_averted_barchart. Same `data_by_pjnz`/`year_range`/`sources`
# dependency pattern as `multi_plot_panel_server`/`multi_facet_panel_server`:
# reads from the shared multi_pjnz_panel_server, so only the files currently
# checked in the sidebar's "PJNZ files" multiselect are run and available as
# baseline/comparison, and `sources` tracks the sidebar's Goals/DP-AIM Model
# switch the same way every other Multi PJNZ sub-tab does.
# ---------------------------------------------------------------------------

@module.ui
def averted_panel_ui(*, indicator_choices: dict[str, str]):
    return ui.div(
        ui.div(
            ui.input_selectize("baseline", label="Baseline PJNZ", choices=[]),
            ui.input_selectize(
                "indicator",
                label="Indicator",
                choices=indicator_choices,
                selected=next(iter(indicator_choices), None),
            ),
            style="display: flex; gap: 20px; max-width: 640px;",
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def averted_panel_server(
    input,
    output,
    session,
    *,
    data_by_pjnz: Callable[[], tuple[dict[str, tuple], dict[str, str]]],
    year_range: Callable[[], tuple[int, int]],
    indicator_map: dict[str, Any],
    # Display label (e.g. "Infections averted") -> indicator_map key (e.g.
    # "New HIV infections").
    indicator_choices: dict[str, str],
    sources: Callable[[], list[ComparisonSource]],
    no_pjnz_message: str = "Select at least one PJNZ file.",
):
    @reactive.effect
    def _refresh_baseline_choices():
        """Re-fires whenever the sidebar's file selection (or Model switch,
        which swaps the whole file pool) changes `data_by_pjnz()`'s keys —
        same trigger pattern as `_refresh_pjnz_choices` elsewhere in this
        module. Preserves the current baseline if it's still selected."""
        data, _errors = data_by_pjnz()
        stems = list(data.keys())
        with reactive.isolate():
            current = input.baseline()
        selected = current if current in stems else (stems[0] if stems else None)
        ui.update_selectize("baseline", choices=stems, selected=selected)

    @output
    @render.ui
    def comparison_plot():
        data, errors = data_by_pjnz()

        error_banner = [
            ui.div(
                ui.p(
                    f"Error running model for '{stem}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(msg, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
            for stem, msg in errors.items()
        ]

        if not data:
            if error_banner:
                return ui.div(*error_banner)
            return ui.p(no_pjnz_message)

        baseline_stem = input.baseline()
        if not baseline_stem or baseline_stem not in data:
            return ui.div(*error_banner, ui.p("Select a baseline PJNZ."))

        indicator_label = input.indicator()
        indicator_name = indicator_choices.get(indicator_label)
        if indicator_name is None:
            return ui.div(*error_banner, ui.p("Select an indicator."))

        if len(data) < 2:
            return ui.div(*error_banner, ui.p("Select at least one more PJNZ file besides the baseline."))

        year_start, year_end = year_range()
        stems = list(data.keys())
        data_by_source_map = {stem: result[0] for stem, result in data.items()}
        output_years_by_pjnz = {stem: result[1] for stem, result in data.items()}

        fig = render_averted_barchart(
            indicator_map=indicator_map,
            data_by_pjnz=data_by_source_map,
            output_years_by_pjnz=output_years_by_pjnz,
            sources=sources(),
            indicator=indicator_name,
            baseline_stem=baseline_stem,
            pjnz_stems=stems,
            year_start=year_start,
            year_end=year_end,
            title=f"{indicator_label} — baseline: {baseline_stem}",
            y_title=indicator_label,
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.div(*error_banner, ui.HTML(html))


# ---------------------------------------------------------------------------
# Multi PJNZ facet panel: the multi-file counterpart of `facet_panel_ui/server`
# — a single-select indicator dropdown (no "By sex" checkbox, v1 multi-file
# plots are totals-only per ADR-0003) driving the one-row-per-CD4-stage layout
# with one line per (PJNZ file, source) pair per row — see
# plotting.render_multi_risk_group_comparison. Used by the Multi PJNZ tab's
# "0-14" sub-tab.
# ---------------------------------------------------------------------------

@module.ui
def multi_facet_panel_ui(*, indicator_names: list[str], initial_sources: list[ComparisonSource]):
    return ui.div(
        ui.input_checkbox_group(
            "visible_sources",
            "Show lines",
            choices={s.key: s.label for s in initial_sources},
            selected=default_visible_keys(initial_sources),
        ),
        ui.input_selectize(
            "indicator",
            label="Indicator",
            choices=indicator_names,
            selected=indicator_names[0] if indicator_names else None,
        ),
        ui.div(
            ui.output_ui("comparison_plot"),
            style="overflow-x: auto; overflow-y: auto;",
        ),
        style="padding-top: 12px;",
    )


@module.server
def multi_facet_panel_server(
    input,
    output,
    session,
    *,
    data_by_pjnz: Callable[[], tuple[dict[str, tuple], dict[str, str]]],
    year_range: Callable[[], tuple[int, int]],
    # Indicator name -> object with `.cd4_labels: list[str]` and
    # `.compute_fns: dict[str, Callable]` (duck-typed to
    # indicator_map.ChildCD4IndicatorDef) — same shape as `facet_panel_server`'s
    # `facet_map`.
    facet_map: dict[str, Any],
    sources: Callable[[], list[ComparisonSource]],
    title_prefix: str,
    no_pjnz_message: str = "Select at least one PJNZ file.",
):
    @reactive.effect
    def _refresh_source_choices():
        """Same reactive-refresh pattern as `multi_plot_panel_server` — `sources()`
        changes value when the Model switch flips."""
        srcs = sources()
        ui.update_checkbox_group(
            "visible_sources",
            choices={s.key: s.label for s in srcs},
            selected=default_visible_keys(srcs),
        )

    @output
    @render.ui
    def comparison_plot():
        data, errors = data_by_pjnz()

        error_banner = [
            ui.div(
                ui.p(
                    f"Error running model for '{stem}':",
                    style="font-weight:bold; color:#c0392b; margin-bottom:4px;",
                ),
                ui.pre(msg, style="white-space:pre-wrap; color:#c0392b; font-size:0.85em;"),
            )
            for stem, msg in errors.items()
        ]

        if not data:
            if error_banner:
                return ui.div(*error_banner)
            return ui.p(no_pjnz_message)

        indicator = input.indicator()
        if not indicator or indicator not in facet_map:
            return ui.div(*error_banner, ui.p("Select an indicator."))

        year_start, year_end = year_range()
        stems = list(data.keys())
        data_by_source_map = {stem: result[0] for stem, result in data.items()}
        output_years_by_pjnz = {stem: result[1] for stem, result in data.items()}
        ind_def = facet_map[indicator]
        facet_desc = cd4_facet_desc(ind_def.cd4_labels)

        active_sources = visible_sources(sources(), input.visible_sources() or ())
        fig = render_multi_risk_group_comparison(
            risk_groups=[(lbl, i) for i, lbl in enumerate(ind_def.cd4_labels)],
            sources=active_sources,
            data_by_pjnz=data_by_source_map,
            output_years_by_pjnz=output_years_by_pjnz,
            compute_fns=ind_def.compute_fns,
            pjnz_stems=stems,
            year_start=year_start,
            year_end=year_end,
            title=f"{title_prefix} {facet_desc} — {indicator} — Multi PJNZ comparison",
        )
        html = fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        return ui.div(*error_banner, ui.HTML(html))
