"""App."""

import altair as alt  # type: ignore
import streamlit as st  # type: ignore
import i18n  # type: ignore
import os # type: ignore

from penn_chime.presentation import (
    build_download_link,
    display_header,
    display_sidebar,
    draw_census_table,
    draw_projected_admissions_table,
    draw_raw_sir_simulation_table,
    hide_menu_style,
    show_additional_projections,
    show_more_info_about_this_tool,
    write_definitions,
    write_footer,
)
from penn_chime.settings import DEFAULTS
from penn_chime.models import sim_sir_df, build_admissions_df, build_census_df
from penn_chime.charts import (additional_projections_chart,
                               admitted_patients_chart,
                               new_admissions_chart,
                               chart_descriptions)

i18n.set('filename_format', '{locale}.{format}')
i18n.set('locale', 'ja')
i18n.set('fallback', 'en')
i18n.load_path.append(os.path.dirname(__file__) + '/../locales')

# This is somewhat dangerous:
# Hide the main menu with "Rerun", "run on Save", "clear cache", and "record a screencast"
# This should not be hidden in prod, but removed
# In dev, this should be shown
st.markdown(hide_menu_style, unsafe_allow_html=True)

p = display_sidebar(st, DEFAULTS)

display_header(st, p)

if st.checkbox(i18n.t("Show more info about this tool")):
    notes = i18n.t("The total size of the susceptible population will be the entire catchment area for Penn Medicine entities (HUP, PAH, PMC, CCH)")
    show_more_info_about_this_tool(st=st, parameters=p, inputs=DEFAULTS, notes=notes)


# begin format data
admissions_df = build_admissions_df(p=p)  # p.n_days, *p.dispositions)
census_df = build_census_df(admissions_df, parameters=p)
# end format data

st.subheader(i18n.t("New Admissions"))
st.markdown(i18n.t("Projected number of **daily** COVID-19 admissions at Penn hospitals"))
new_admit_chart = new_admissions_chart(alt, admissions_df, parameters=p)
st.altair_chart(
    new_admit_chart, use_container_width=True
)

st.markdown(chart_descriptions(new_admit_chart))

if st.checkbox(i18n.t("Show Projected Admissions in tabular form")):
    if st.checkbox(i18n.t("Show Daily Counts")):
        draw_projected_admissions_table(st, admissions_df, as_date=p.as_date, daily_count=True)
    else:
        draw_projected_admissions_table(st, admissions_df, as_date=p.as_date, daily_count=False)
    build_download_link(st,
        filename="projected_admissions.csv",
        df=admissions_df,
        parameters=p
    )
st.subheader(i18n.t("Admitted Patients (Census)"))
st.markdown(
    i18n.t("Projected **census** of COVID-19 patients, accounting for arrivals and discharges at Penn hospitals")
)
census_chart = admitted_patients_chart(alt=alt, census=census_df, parameters=p)
st.altair_chart(
    census_chart, use_container_width=True
)
st.markdown(chart_descriptions(census_chart, suffix=i18n.t(" Census")))
if st.checkbox(i18n.t("Show Projected Census in tabular form")):
    if st.checkbox(i18n.t("Show Daily Census Counts")):
        draw_census_table(st, admissions_df, as_date=p.as_date, daily_count=True)
    else:
        draw_census_table(st, census_df, as_date=p.as_date, daily_count=False)
    build_download_link(st,
        filename="projected_census.csv",
        df=census_df,
        parameters=p
    )

st.markdown(
    i18n.t("**Click the checkbox below to view additional data generated by this simulation**")
)
if st.checkbox(i18n.t("Show Additional Projections")):
    show_additional_projections(
        st, alt, additional_projections_chart, parameters=p
    )
    if st.checkbox(i18n.t("Show Raw SIR Simulation Data")):
        draw_raw_sir_simulation_table(st, parameters=p)
write_definitions(st)
write_footer(st)
