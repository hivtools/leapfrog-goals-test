"""
Interactive comparison dashboard — AIM vs Goals vs EPPASM leapfrog comparisons.

Usage:
    uv run shiny run app.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from shiny import App, reactive, ui

import leapfrog_compare.config as config
from leapfrog_compare.comparison_module import (
    age_profile_panel_server, age_profile_panel_ui, averted_panel_server, averted_panel_ui,
    data_panel_server, data_panel_ui, facet_panel_server, facet_panel_ui, multi_facet_panel_server,
    multi_facet_panel_ui, multi_pjnz_panel_server, multi_pjnz_panel_ui, multi_plot_panel_server,
    multi_plot_panel_ui, multi_resource_needs_panel_server, multi_resource_needs_panel_ui,
    multi_risk_group_panel_server, multi_risk_group_panel_ui,
    plot_panel_server, plot_panel_ui, resource_needs_panel_server, resource_needs_panel_ui,
    risk_group_panel_server, risk_group_panel_ui,
)
from leapfrog_compare.eppasm_indicator_map import (
    EPPASM_ALL_AGES_INDICATOR_NAMES, EPPASM_FIFTEEN_49_INDICATOR_NAMES,
    EPPASM_AGE_LABELS, EPPASM_INDICATOR_MAP,
)
from leapfrog_compare.eppasm_runner import run_eppasm_both
from leapfrog_compare.indicator_map import (
    AGE_LABELS, AGE_PROFILE_INDICATOR_NAMES, ALL_AGES_INDICATOR_NAMES, CHILD_CD4_INDICATOR_MAP,
    CHILD_CD4_INDICATOR_NAMES, DEATHS_INDICATOR_NAMES, FIFTEEN_49_INDICATOR_NAMES, INDICATOR_MAP,
    PEOPLE_REACHED_INDICATOR_MAP, RESOURCES_REQUIRED_INDICATOR_MAP, RESOURCE_NEEDS_INDICATOR_NAMES,
    RISK_GROUP_INDICATOR_MAP, RISK_GROUP_INDICATOR_NAMES, RISK_GROUPS,
)
from leapfrog_compare.pjnz_classify import is_goals_pjnz
from leapfrog_compare.pjnz_runner import run_pjnz
from leapfrog_compare.plotting import ComparisonSource
from leapfrog_compare.spectrum_runner import run_spectrum

_DEFAULT_YEAR_MIN = 1970
_DEFAULT_YEAR_MAX = 2030

def _iter_pjnz_files():
    """Yield *.pjnz files in PJNZ_DIR, matched case-insensitively — glob() is
    case-sensitive on Linux, and PJNZ files are sometimes named with a
    lowercase extension (e.g. Angola_Final_200526.pjnz)."""
    return (p for p in config.PJNZ_DIR.expanduser().iterdir() if p.suffix.lower() == ".pjnz")


def _scan_pjnz_files() -> dict[str, Path]:
    return {p.stem: p for p in sorted(_iter_pjnz_files())}


def _pjnz_fingerprint() -> list[tuple[str, int, int]]:
    """Cheap, frequently-polled signature of PJNZ_DIR's contents: (name, mtime_ns,
    size) per file, sorted for stable comparison. Changes whenever a file is
    added, removed, or edited — without opening/parsing anything."""
    return sorted(
        (p.name, p.stat().st_mtime_ns, p.stat().st_size)
        for p in _iter_pjnz_files()
    )


@reactive.poll(_pjnz_fingerprint, config.PJNZ_POLL_INTERVAL_SECS)
def pjnz_files() -> dict[str, Path]:
    """All .PJNZ files in PJNZ_DIR, keyed by stem. Re-scanned only when
    _pjnz_fingerprint() changes, so files added/removed/edited in PJNZ_DIR are
    picked up within one poll interval — no app restart needed. Declared at
    module level (not per-session) since all sessions share the same directory."""
    return _scan_pjnz_files()


# Classify each PJNZ as "Goals" (has a .HV member — ran Spectrum's Goals/HIV
# module) or "AIM" (doesn't). The AIM tab only offers AIM-only files, the Goals
# tab only offers Goals-capable files, and EPPASM offers all of them labelled.
@reactive.calc
def pjnz_is_goals() -> dict[str, bool]:
    return {stem: is_goals_pjnz(path) for stem, path in pjnz_files().items()}


@reactive.calc
def pjnz_stems_goals() -> list[str]:
    is_goals = pjnz_is_goals()
    return [s for s in pjnz_files() if is_goals[s]]


@reactive.calc
def pjnz_stems_aim() -> list[str]:
    is_goals = pjnz_is_goals()
    return [s for s in pjnz_files() if not is_goals[s]]


@reactive.calc
def pjnz_choices_eppasm() -> dict[str, str]:
    is_goals = pjnz_is_goals()
    return {s: f"{s} (Goals)" if is_goals[s] else f"{s} (AIM)" for s in pjnz_files()}


# One-off snapshot used only to paint the initial (pre-session) UI; the
# reactive getters above keep the live choices in sync with PJNZ_DIR for each
# session afterwards (wired up in data_panel_server).
_pjnz_files_initial = _scan_pjnz_files()
_pjnz_is_goals_initial = {stem: is_goals_pjnz(path) for stem, path in _pjnz_files_initial.items()}
_pjnz_stems_aim_initial = [s for s in _pjnz_files_initial if not _pjnz_is_goals_initial[s]]
_pjnz_stems_goals_initial = [s for s in _pjnz_files_initial if _pjnz_is_goals_initial[s]]
_pjnz_choices_eppasm_initial = {
    s: f"{s} (Goals)" if _pjnz_is_goals_initial[s] else f"{s} (AIM)" for s in _pjnz_files_initial
}

_GOALS_SOURCES = [
    ComparisonSource(key="dp_aim", label="Leapfrog DP/AIM", dash=None, primary=True, default_visible=True),
    ComparisonSource(key="spectrum", label="Spectrum", dash="dash", needs_offset_align=True),
    ComparisonSource(key="goals", label="Leapfrog Goals", dash="dot", supports_age_facet=False),
]

_AIM_SOURCES = [
    ComparisonSource(key="dp_aim", label="Leapfrog AIM", dash=None, primary=True, default_visible=True),
    ComparisonSource(key="spectrum_aim", label="Spectrum", dash="dash", needs_offset_align=True),
]

_EPPASM_SOURCES = [
    ComparisonSource(key="eppasm", label="eppasm", dash=None, primary=True),
    ComparisonSource(key="eppasm_lf", label="eppasm-leapfrog", dash="dash"),
]

# "Resource needs" tab: leapfrog Goals' num_people_reached / resources_required
# vs the equivalent Spectrum modvars. No "dp_aim" source — resource needs are a
# Goals-module output with no DP/AIM equivalent. "spectrum" reads raw modvars,
# so needs_offset_align like every other raw-modvar source.
_RESOURCE_NEEDS_SOURCES = [
    ComparisonSource(key="goals", label="Leapfrog Goals", dash=None, default_visible=True),
    ComparisonSource(key="spectrum", label="Spectrum", dash="dash", needs_offset_align=True),
]


def _goals_run_fn(pjnz_path: Path):
    modvars, goals_output, output_years = run_pjnz(pjnz_path)
    return {"dp_aim": goals_output, "spectrum": modvars, "goals": goals_output}, output_years


def _aim_run_fn(pjnz_path: Path):
    modvars, leapfrog_output, output_years = run_spectrum(pjnz_path)
    return {"dp_aim": leapfrog_output, "spectrum_aim": modvars}, output_years


def _eppasm_run_fn(pjnz_path: Path, force: bool = False):
    return run_eppasm_both(pjnz_path, force=force)


# ---------------------------------------------------------------------------
# Sub-tab definitions: adding a new "type of plot" to a top-level tab is just
# adding one more SubTab here (and one more entry in the matching TOP_TABS list).
# ---------------------------------------------------------------------------

@dataclass
class SubTab:
    id: str
    label: str
    indicator_names: list[str]
    default_indicators: list[str]
    indicator_map: dict
    sources: list[ComparisonSource]
    age_labels: list[str]
    show_age_checkbox: bool = True


@dataclass
class RiskGroupSubTab:
    """A sub-tab with the dedicated one-row-per-risk-group layout (no age faceting)
    — see plotting.render_risk_group_comparison. `indicator_map` picks between
    risk-group-faceted indicators (population share vs. new infections) via a
    single-select dropdown — see comparison_module.risk_group_panel_ui/server and
    indicator_map.RiskGroupIndicatorDef."""
    id: str
    label: str
    indicator_names: list[str]
    indicator_map: dict[str, Any]


@dataclass
class FacetSubTab:
    """A sub-tab with a single-select indicator dropdown driving the same
    one-row-per-group layout as RiskGroupSubTab (plotting.render_risk_group_comparison),
    but the active `compute_fns`/group list is chosen dynamically per selected
    indicator via `facet_map` — see comparison_module.facet_panel_ui/server.
    Unlike RiskGroupSubTab, `sources` lives on the sub-tab itself (not a single
    module-level constant), since the AIM and Goals '0-14' tabs need different
    source keys (spectrum_aim vs spectrum)."""
    id: str
    label: str
    indicator_names: list[str]
    facet_map: dict[str, Any]
    sources: list[ComparisonSource]
    title_prefix: str
    wip_note: str | None = None


@dataclass
class AgeProfileSubTab:
    """A sub-tab with a single-year-of-age profile: an indicator multiselect
    (same pattern as SubTab) + a year dropdown (instead of the shared
    year-range slider) driving one plot per selected indicator, each with age
    (0-80) on the x-axis and one line per (source, sex) — see
    comparison_module.age_profile_panel_ui/server and plotting.render_age_profile.
    Only sources whose `IndicatorDef.age_profile` actually has an entry for a
    given indicator contribute a line (e.g. on the Goals tab, most indicators
    only have single-age data via dp_aim; spectrum/goals mostly do not)."""
    id: str
    label: str
    indicator_map: dict
    sources: list[ComparisonSource]
    indicator_names: list[str] = field(default_factory=lambda: AGE_PROFILE_INDICATOR_NAMES)
    default_indicators: list[str] = field(default_factory=lambda: AGE_PROFILE_INDICATOR_NAMES[:3])
    title_prefix: str = "By age"


@dataclass
class MultiSubTab:
    """A Multi PJNZ sub-tab: an indicator multiselect driving one line per
    (PJNZ file, source) pair — see comparison_module.multi_plot_panel_ui/server
    and plotting.render_multi_pjnz_comparison. Unlike SubTab, there is no
    `sources`/`indicator_map` field here: `indicator_map` is always the shared
    INDICATOR_MAP, and `sources` is chosen dynamically from the Model switch
    (`_GOALS_SOURCES`/`_AIM_SOURCES`, reused verbatim — see ADR-0002)."""
    id: str
    label: str
    indicator_names: list[str]
    default_indicators: list[str]


@dataclass
class MultiRiskGroupSubTab:
    """A Multi PJNZ sub-tab with the dedicated one-row-per-risk-group layout — see
    comparison_module.multi_risk_group_panel_ui/server and
    plotting.render_multi_risk_group_comparison. Totals-only, no "By sex" checkbox
    (v1 multi-file plots drop sex disaggregation entirely, per ADR-0003). Always wired
    with `_GOALS_RISKGROUP_SOURCES`: risk-group data only exists under the "goals"/
    "spectrum" keys, so under Model=DP/AIM these sub-tabs render nothing, matching the
    missing-key-skip convention every other Multi PJNZ sub-tab already relies on.
    `indicator_map` picks between risk-group-faceted indicators via a single-select
    dropdown, same as RiskGroupSubTab."""
    id: str
    label: str
    indicator_names: list[str]
    indicator_map: dict[str, Any]


@dataclass
class MultiFacetSubTab:
    """A Multi PJNZ sub-tab with a single-select indicator dropdown driving the
    one-row-per-CD4-stage layout — see comparison_module.multi_facet_panel_ui/server
    and plotting.render_multi_risk_group_comparison. Like MultiSubTab (not
    MultiRiskGroupSubTab), `sources` is chosen dynamically from the Model switch:
    unlike risk-group data (goals/spectrum only), CHILD_CD4_INDICATOR_MAP entries
    define both spectrum and spectrum_aim keys, so CD4 child data exists under
    both Goals and DP/AIM."""
    id: str
    label: str
    indicator_names: list[str]
    facet_map: dict[str, Any]
    title_prefix: str


@dataclass
class AvertedSubTab:
    """A Multi PJNZ sub-tab with a baseline-PJNZ selector + indicator dropdown
    (embedded in the panel body) driving a grouped bar chart of (baseline
    total - PJNZ total), aggregated over age/sex/year — see
    comparison_module.averted_panel_ui/server and plotting.render_averted_barchart.
    Reads from the shared multi_pjnz_panel_server's data_by_pjnz/year_range/model,
    same as MultiSubTab: only the files checked in the sidebar's "PJNZ files"
    multiselect are run and available as baseline/comparison, and the source
    keys (dp_aim/spectrum vs dp_aim/spectrum_aim) follow the sidebar's Model switch."""
    id: str
    label: str
    indicator_choices: dict[str, str]


# Display label -> INDICATOR_MAP key. "New HIV infections"/"AIDS deaths" are the
# all-ages indicators with plain dp_aim + spectrum disagg entries and no "goals"
# key (see indicator_map.py) — exactly the two-source blue-leapfrog/orange-spectrum
# shape the averted bar chart wants.
_AVERTED_INDICATOR_CHOICES = {
    "Infections averted": "New HIV infections",
    "Deaths averted": "AIDS deaths",
}

_GOALS_RISKGROUP_SOURCES = [
    ComparisonSource(key="goals", label="Leapfrog Goals", dash=None, default_visible=True),
    # HV_AdultsTag/HV_NewInfectionsTag (read by compute_rg_spectrum/
    # compute_new_infections_rg_spectrum) are the same raw modvar arrays
    # _spec_newinf_disagg et al. read for _GOALS_SOURCES, which needs
    # needs_offset_align=True — same requirement applies here.
    ComparisonSource(key="spectrum", label="Spectrum", dash="dash", needs_offset_align=True),
]

_GOALS_CHILD_CD4_SOURCES = [
    ComparisonSource(key="dp_aim", label="Leapfrog Goals", dash=None, default_visible=True),
    ComparisonSource(key="spectrum", label="Spectrum", dash="dash", needs_offset_align=True),
]


_AIM_SUBTABS = [
    SubTab(
        id="aim_allages", label="All ages",
        indicator_names=ALL_AGES_INDICATOR_NAMES, default_indicators=ALL_AGES_INDICATOR_NAMES[:3],
        indicator_map=INDICATOR_MAP, sources=_AIM_SOURCES, age_labels=AGE_LABELS,
    ),
    SubTab(
        id="aim_1549", label="15-49",
        indicator_names=FIFTEEN_49_INDICATOR_NAMES, default_indicators=FIFTEEN_49_INDICATOR_NAMES,
        indicator_map=INDICATOR_MAP, sources=_AIM_SOURCES, age_labels=AGE_LABELS,
        show_age_checkbox=False,
    ),
    SubTab(
        id="aim_deaths", label="Deaths",
        indicator_names=DEATHS_INDICATOR_NAMES, default_indicators=DEATHS_INDICATOR_NAMES[:3],
        indicator_map=INDICATOR_MAP, sources=_AIM_SOURCES, age_labels=AGE_LABELS,
        show_age_checkbox=False,
    ),
]

_GOALS_SUBTABS = [
    SubTab(
        id="goals_allages", label="All ages",
        indicator_names=ALL_AGES_INDICATOR_NAMES, default_indicators=ALL_AGES_INDICATOR_NAMES[:3],
        indicator_map=INDICATOR_MAP, sources=_GOALS_SOURCES, age_labels=AGE_LABELS,
    ),
    SubTab(
        id="goals_1549", label="15-49",
        indicator_names=FIFTEEN_49_INDICATOR_NAMES, default_indicators=FIFTEEN_49_INDICATOR_NAMES,
        indicator_map=INDICATOR_MAP, sources=_GOALS_SOURCES, age_labels=AGE_LABELS,
        show_age_checkbox=False,
    ),
    SubTab(
        id="goals_deaths", label="Deaths",
        indicator_names=DEATHS_INDICATOR_NAMES, default_indicators=DEATHS_INDICATOR_NAMES[:3],
        indicator_map=INDICATOR_MAP, sources=_GOALS_SOURCES, age_labels=AGE_LABELS,
        show_age_checkbox=False,
    ),
]

# "Dataset" dropdown choices for the "Resource needs" sub-tab: label -> the
# per-intervention IndicatorDef map reading that Goals output array.
_RESOURCE_NEEDS_DATASETS = {
    "People reached": PEOPLE_REACHED_INDICATOR_MAP,
    "Resources required": RESOURCES_REQUIRED_INDICATOR_MAP,
}


@dataclass
class ResourceNeedsSubTab:
    """A sub-tab with a single-select "Dataset" dropdown (num_people_reached vs
    resources_required) ahead of the usual indicator multiselect — see
    comparison_module.resource_needs_panel_ui/server (single-PJNZ) and
    multi_resource_needs_panel_ui/server (Multi PJNZ). `dataset_maps` is
    label -> indicator_map; `sources` is always the goals/spectrum pair
    (`_RESOURCE_NEEDS_SOURCES`)."""
    id: str
    label: str
    dataset_maps: dict[str, dict]
    indicator_names: list[str]
    default_indicators: list[str]
    sources: list[ComparisonSource]


_GOALS_RESOURCE_NEEDS_SUBTABS = [
    ResourceNeedsSubTab(
        id="goals_resource_needs", label="Resource needs",
        dataset_maps=_RESOURCE_NEEDS_DATASETS,
        indicator_names=RESOURCE_NEEDS_INDICATOR_NAMES,
        default_indicators=RESOURCE_NEEDS_INDICATOR_NAMES[:3],
        sources=_RESOURCE_NEEDS_SOURCES,
    ),
]

_MULTI_RESOURCE_NEEDS_SUBTABS = [
    ResourceNeedsSubTab(
        id="multi_resource_needs", label="Resource needs",
        dataset_maps=_RESOURCE_NEEDS_DATASETS,
        indicator_names=RESOURCE_NEEDS_INDICATOR_NAMES,
        default_indicators=RESOURCE_NEEDS_INDICATOR_NAMES[:3],
        sources=_RESOURCE_NEEDS_SOURCES,
    ),
]

_GOALS_RISKGROUP_SUBTABS = [
    RiskGroupSubTab(
        id="goals_riskgroups", label="Risk groups",
        indicator_names=RISK_GROUP_INDICATOR_NAMES, indicator_map=RISK_GROUP_INDICATOR_MAP,
    ),
]

_AIM_CHILD_SUBTABS = [
    FacetSubTab(
        id="aim_child_cd4", label="0-14",
        indicator_names=CHILD_CD4_INDICATOR_NAMES, facet_map=CHILD_CD4_INDICATOR_MAP,
        sources=_AIM_SOURCES, title_prefix="Child",
    ),
]

_AIM_AGEPROFILE_SUBTABS = [
    AgeProfileSubTab(
        id="aim_age_profile", label="By age",
        indicator_map=INDICATOR_MAP, sources=_AIM_SOURCES,
    ),
]

_GOALS_AGEPROFILE_SUBTABS = [
    AgeProfileSubTab(
        id="goals_age_profile", label="By age",
        indicator_map=INDICATOR_MAP, sources=_GOALS_SOURCES,
    ),
]

_GOALS_CHILD_SUBTABS = [
    FacetSubTab(
        id="goals_child_cd4", label="0-14",
        indicator_names=CHILD_CD4_INDICATOR_NAMES, facet_map=CHILD_CD4_INDICATOR_MAP,
        sources=_GOALS_CHILD_CD4_SOURCES, title_prefix="Child",
    ),
]

_MULTI_SUBTABS = [
    MultiSubTab(
        id="multi_allages", label="All ages",
        indicator_names=ALL_AGES_INDICATOR_NAMES,
        default_indicators=["New HIV infections", "AIDS deaths"],
    ),
    MultiSubTab(
        id="multi_1549", label="15-49",
        indicator_names=FIFTEEN_49_INDICATOR_NAMES,
        default_indicators=FIFTEEN_49_INDICATOR_NAMES,
    ),
    MultiSubTab(
        id="multi_deaths", label="Deaths",
        indicator_names=DEATHS_INDICATOR_NAMES,
        default_indicators=DEATHS_INDICATOR_NAMES[:3],
    ),
]

_MULTI_RISKGROUP_SUBTABS = [
    MultiRiskGroupSubTab(
        id="multi_riskgroups", label="Risk groups",
        indicator_names=RISK_GROUP_INDICATOR_NAMES, indicator_map=RISK_GROUP_INDICATOR_MAP,
    ),
]

_MULTI_CHILD_SUBTABS = [
    MultiFacetSubTab(
        id="multi_child_cd4", label="0-14",
        indicator_names=CHILD_CD4_INDICATOR_NAMES, facet_map=CHILD_CD4_INDICATOR_MAP,
        title_prefix="Child",
    ),
]

_MULTI_AVERTED_SUBTABS = [
    AvertedSubTab(
        id="multi_averted", label="Averted",
        indicator_choices=_AVERTED_INDICATOR_CHOICES,
    ),
]

_EPPASM_SUBTABS = [
    SubTab(
        id="eppasm_allages", label="All ages",
        indicator_names=EPPASM_ALL_AGES_INDICATOR_NAMES, default_indicators=EPPASM_ALL_AGES_INDICATOR_NAMES[:3],
        indicator_map=EPPASM_INDICATOR_MAP, sources=_EPPASM_SOURCES, age_labels=EPPASM_AGE_LABELS,
    ),
    SubTab(
        id="eppasm_1549", label="15-49",
        indicator_names=EPPASM_FIFTEEN_49_INDICATOR_NAMES, default_indicators=EPPASM_FIFTEEN_49_INDICATOR_NAMES,
        indicator_map=EPPASM_INDICATOR_MAP, sources=_EPPASM_SOURCES, age_labels=EPPASM_AGE_LABELS,
        show_age_checkbox=False,
    ),
]


# ---------------------------------------------------------------------------
# UI / server composition: one nav_panel per top-level tab, one inner
# navset_tab per top-level tab's SubTabs, sharing a single data_panel.
# ---------------------------------------------------------------------------

def _wip_banner(note: str):
    return ui.div(
        ui.strong("Work in progress: "),
        note,
        style=(
            "background:#fff3cd; border:1px solid #ffe08a; border-radius:4px; "
            "padding:8px 12px; margin-bottom:12px; color:#664d03; font-size:0.9em;"
        ),
    )


def _build_tab_ui(
    top_id: str,
    title: str,
    sub_tabs: list[SubTab],
    *,
    pjnz_choices: list[str] | dict[str, str] = (),
    show_rerun_button: bool = False,
    risk_group_subtabs: list[RiskGroupSubTab] = (),
    facet_subtabs: list[FacetSubTab] = (),
    age_profile_subtabs: list[AgeProfileSubTab] = (),
    resource_needs_subtabs: list[ResourceNeedsSubTab] = (),
    wip_note: str | None = None,
):
    banner = [_wip_banner(wip_note)] if wip_note else []
    return ui.nav_panel(
        title,
        *banner,
        ui.layout_sidebar(
            data_panel_ui(
                top_id,
                pjnz_choices=pjnz_choices,
                year_min=_DEFAULT_YEAR_MIN,
                year_max=_DEFAULT_YEAR_MAX,
                show_rerun_button=show_rerun_button,
            ),
            ui.navset_tab(*[
                ui.nav_panel(
                    st.label,
                    plot_panel_ui(
                        st.id,
                        indicator_names=st.indicator_names,
                        default_indicators=st.default_indicators,
                        show_age_checkbox=st.show_age_checkbox,
                    ),
                )
                for st in sub_tabs
            ], *[
                ui.nav_panel(rgt.label, risk_group_panel_ui(rgt.id, indicator_names=rgt.indicator_names))
                for rgt in risk_group_subtabs
            ], *[
                ui.nav_panel(
                    ft.label,
                    *([_wip_banner(ft.wip_note)] if ft.wip_note else []),
                    facet_panel_ui(ft.id, indicator_names=ft.indicator_names),
                )
                for ft in facet_subtabs
            ], *[
                ui.nav_panel(
                    apt.label,
                    age_profile_panel_ui(
                        apt.id,
                        indicator_names=apt.indicator_names,
                        default_indicators=apt.default_indicators,
                        year_min=_DEFAULT_YEAR_MIN, year_max=_DEFAULT_YEAR_MAX,
                    ),
                )
                for apt in age_profile_subtabs
            ], *[
                ui.nav_panel(
                    rnt.label,
                    resource_needs_panel_ui(
                        rnt.id,
                        dataset_labels=list(rnt.dataset_maps),
                        indicator_names=rnt.indicator_names,
                        default_indicators=rnt.default_indicators,
                    ),
                )
                for rnt in resource_needs_subtabs
            ]),
            fillable=True,
        ),
    )


def _wire_tab_server(
    top_id: str,
    run_fn,
    sub_tabs: list[SubTab],
    *,
    pjnz_choices: Callable[[], list[str] | dict[str, str]],
    show_rerun_button: bool = False,
    risk_group_subtabs: list[RiskGroupSubTab] = (),
    facet_subtabs: list[FacetSubTab] = (),
    age_profile_subtabs: list[AgeProfileSubTab] = (),
    resource_needs_subtabs: list[ResourceNeedsSubTab] = (),
):
    data_run, year_range, pjnz_label = data_panel_server(
        top_id, pjnz_files=pjnz_files, pjnz_choices=pjnz_choices,
        run_fn=run_fn, show_rerun_button=show_rerun_button,
    )
    for st in sub_tabs:
        plot_panel_server(
            st.id,
            data_run=data_run,
            year_range=year_range,
            pjnz_label=pjnz_label,
            indicator_map=st.indicator_map,
            sources=st.sources,
            age_labels=st.age_labels,
            show_age_checkbox=st.show_age_checkbox,
        )
    for rgt in risk_group_subtabs:
        risk_group_panel_server(
            rgt.id,
            data_run=data_run,
            year_range=year_range,
            pjnz_label=pjnz_label,
            risk_groups=RISK_GROUPS,
            sources=_GOALS_RISKGROUP_SOURCES,
            indicator_map=rgt.indicator_map,
        )
    for ft in facet_subtabs:
        facet_panel_server(
            ft.id,
            data_run=data_run,
            year_range=year_range,
            pjnz_label=pjnz_label,
            facet_map=ft.facet_map,
            sources=ft.sources,
            title_prefix=ft.title_prefix,
        )
    for apt in age_profile_subtabs:
        age_profile_panel_server(
            apt.id,
            data_run=data_run,
            pjnz_label=pjnz_label,
            indicator_map=apt.indicator_map,
            sources=apt.sources,
            title_prefix=apt.title_prefix,
        )
    for rnt in resource_needs_subtabs:
        resource_needs_panel_server(
            rnt.id,
            data_run=data_run,
            year_range=year_range,
            pjnz_label=pjnz_label,
            dataset_maps=rnt.dataset_maps,
            sources=rnt.sources,
        )


def _build_multi_tab_ui(
    top_id: str,
    title: str,
    sub_tabs: list[MultiSubTab],
    *,
    risk_group_subtabs: list[MultiRiskGroupSubTab] = (),
    facet_subtabs: list[MultiFacetSubTab] = (),
    averted_subtabs: list[AvertedSubTab] = (),
    resource_needs_subtabs: list[ResourceNeedsSubTab] = (),
):
    return ui.nav_panel(
        title,
        ui.layout_sidebar(
            multi_pjnz_panel_ui(
                top_id,
                goals_choices=_pjnz_stems_goals_initial,
                year_min=_DEFAULT_YEAR_MIN,
                year_max=_DEFAULT_YEAR_MAX,
            ),
            ui.navset_tab(*[
                ui.nav_panel(
                    st.label,
                    multi_plot_panel_ui(
                        st.id,
                        indicator_names=st.indicator_names,
                        default_indicators=st.default_indicators,
                        initial_sources=_GOALS_SOURCES,
                    ),
                )
                for st in sub_tabs
            ], *[
                ui.nav_panel(
                    rgt.label,
                    multi_risk_group_panel_ui(
                        rgt.id, sources=_GOALS_RISKGROUP_SOURCES, indicator_names=rgt.indicator_names,
                    ),
                )
                for rgt in risk_group_subtabs
            ], *[
                ui.nav_panel(
                    ft.label,
                    multi_facet_panel_ui(
                        ft.id,
                        indicator_names=ft.indicator_names,
                        initial_sources=_GOALS_CHILD_CD4_SOURCES,
                    ),
                )
                for ft in facet_subtabs
            ], *[
                ui.nav_panel(
                    at.label,
                    averted_panel_ui(at.id, indicator_choices=at.indicator_choices),
                )
                for at in averted_subtabs
            ], *[
                ui.nav_panel(
                    rnt.label,
                    multi_resource_needs_panel_ui(
                        rnt.id,
                        dataset_labels=list(rnt.dataset_maps),
                        indicator_names=rnt.indicator_names,
                        default_indicators=rnt.default_indicators,
                        initial_sources=rnt.sources,
                    ),
                )
                for rnt in resource_needs_subtabs
            ]),
            fillable=True,
        ),
    )


def _wire_multi_tab_server(
    top_id: str,
    sub_tabs: list[MultiSubTab],
    *,
    risk_group_subtabs: list[MultiRiskGroupSubTab] = (),
    facet_subtabs: list[MultiFacetSubTab] = (),
    averted_subtabs: list[AvertedSubTab] = (),
    resource_needs_subtabs: list[ResourceNeedsSubTab] = (),
):
    data_by_pjnz, year_range, model = multi_pjnz_panel_server(
        top_id,
        pjnz_files=pjnz_files,
        pjnz_stems_goals=pjnz_stems_goals,
        pjnz_stems_aim=pjnz_stems_aim,
        goals_run_fn=_goals_run_fn,
        aim_run_fn=_aim_run_fn,
    )
    for st in sub_tabs:
        multi_plot_panel_server(
            st.id,
            data_by_pjnz=data_by_pjnz,
            year_range=year_range,
            indicator_map=INDICATOR_MAP,
            sources=lambda: _GOALS_SOURCES if model() == "Goals" else _AIM_SOURCES,
        )
    for rgt in risk_group_subtabs:
        multi_risk_group_panel_server(
            rgt.id,
            data_by_pjnz=data_by_pjnz,
            year_range=year_range,
            risk_groups=RISK_GROUPS,
            sources=_GOALS_RISKGROUP_SOURCES,
            indicator_map=rgt.indicator_map,
        )
    for ft in facet_subtabs:
        multi_facet_panel_server(
            ft.id,
            data_by_pjnz=data_by_pjnz,
            year_range=year_range,
            facet_map=ft.facet_map,
            sources=lambda: _GOALS_CHILD_CD4_SOURCES if model() == "Goals" else _AIM_SOURCES,
            title_prefix=ft.title_prefix,
        )
    for at in averted_subtabs:
        averted_panel_server(
            at.id,
            data_by_pjnz=data_by_pjnz,
            year_range=year_range,
            indicator_map=INDICATOR_MAP,
            indicator_choices=at.indicator_choices,
            sources=lambda: _GOALS_SOURCES if model() == "Goals" else _AIM_SOURCES,
        )
    for rnt in resource_needs_subtabs:
        multi_resource_needs_panel_server(
            rnt.id,
            data_by_pjnz=data_by_pjnz,
            year_range=year_range,
            dataset_maps=rnt.dataset_maps,
            sources=rnt.sources,
        )


app_ui = ui.page_navbar(
    _build_tab_ui(
        "aim", "AIM", _AIM_SUBTABS, pjnz_choices=_pjnz_stems_aim_initial,
        facet_subtabs=_AIM_CHILD_SUBTABS, age_profile_subtabs=_AIM_AGEPROFILE_SUBTABS,
    ),
    _build_tab_ui(
        "goals", "Goals", _GOALS_SUBTABS,
        pjnz_choices=_pjnz_stems_goals_initial, risk_group_subtabs=_GOALS_RISKGROUP_SUBTABS,
        facet_subtabs=_GOALS_CHILD_SUBTABS, age_profile_subtabs=_GOALS_AGEPROFILE_SUBTABS,
        resource_needs_subtabs=_GOALS_RESOURCE_NEEDS_SUBTABS,
    ),
    _build_tab_ui(
        "eppasm", "EPPASM", _EPPASM_SUBTABS,
        pjnz_choices=_pjnz_choices_eppasm_initial, show_rerun_button=True,
    ),
    _build_multi_tab_ui(
        "multi", "Multi PJNZ", _MULTI_SUBTABS,
        risk_group_subtabs=_MULTI_RISKGROUP_SUBTABS, facet_subtabs=_MULTI_CHILD_SUBTABS,
        averted_subtabs=_MULTI_AVERTED_SUBTABS,
        resource_needs_subtabs=_MULTI_RESOURCE_NEEDS_SUBTABS,
    ),
    id="main_nav",
    title="Leapfrog Comparison",
    header=ui.head_content(
        ui.tags.script(src="https://cdn.plot.ly/plotly-latest.min.js"),
        ui.tags.link(
            rel="icon",
            href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐸</text></svg>",
        ),
    ),
    fillable=True,
)


def server(input, output, session):
    _wire_tab_server(
        "aim", _aim_run_fn, _AIM_SUBTABS, facet_subtabs=_AIM_CHILD_SUBTABS,
        age_profile_subtabs=_AIM_AGEPROFILE_SUBTABS, pjnz_choices=pjnz_stems_aim,
    )
    _wire_tab_server(
        "goals", _goals_run_fn, _GOALS_SUBTABS,
        risk_group_subtabs=_GOALS_RISKGROUP_SUBTABS, facet_subtabs=_GOALS_CHILD_SUBTABS,
        age_profile_subtabs=_GOALS_AGEPROFILE_SUBTABS,
        resource_needs_subtabs=_GOALS_RESOURCE_NEEDS_SUBTABS, pjnz_choices=pjnz_stems_goals,
    )
    _wire_tab_server(
        "eppasm", _eppasm_run_fn, _EPPASM_SUBTABS, show_rerun_button=True,
        pjnz_choices=pjnz_choices_eppasm,
    )
    _wire_multi_tab_server(
        "multi", _MULTI_SUBTABS,
        risk_group_subtabs=_MULTI_RISKGROUP_SUBTABS, facet_subtabs=_MULTI_CHILD_SUBTABS,
        averted_subtabs=_MULTI_AVERTED_SUBTABS,
        resource_needs_subtabs=_MULTI_RESOURCE_NEEDS_SUBTABS,
    )


app = App(app_ui, server)
