"""App."""

import os

import altair as alt  # type: ignore
import streamlit as st  # type: ignore
import i18n  # type: ignore
import os # type: ignore

i18n.set('filename_format', '{locale}.{format}')
i18n.set('locale', 'ja')
i18n.set('fallback', 'en')
i18n.load_path.append(os.path.dirname(__file__) + '/penn_chime/locales')

from penn_chime.parameters import Parameters
from penn_chime.presentation import (
    display_download_link,
    display_footer,
    display_header,
    display_sidebar,
    hide_menu_style,
)
from penn_chime.models import SimSirModel
from penn_chime.charts import (
    build_admits_chart,
    build_census_chart,
    build_descriptions,
    build_sim_sir_w_date_chart,
    build_table,
)

# This is somewhat dangerous:
# Hide the main menu with "Rerun", "run on Save", "clear cache", and "record a screencast"
# This should not be hidden in prod, but removed
# In dev, this should be shown
st.markdown(hide_menu_style, unsafe_allow_html=True)

d = Parameters.create(os.environ, [])
p = display_sidebar(st, d)
m = SimSirModel(p)

display_header(st, m, p)

st.subheader(i18n.t("app-new-admissions-title"))
st.markdown(i18n.t("app-new-admissions-text"))
admits_chart = build_admits_chart(alt=alt, admits_floor_df=m.admits_floor_df, max_y_axis=p.max_y_axis)
st.altair_chart(admits_chart, use_container_width=True)
st.markdown(build_descriptions(chart=admits_chart, labels=p.labels, prefix="admits_"))
display_download_link(
    st,
    p,
    filename=f"{p.current_date}_projected_admits.csv",
    df=m.admits_df,
)

if st.checkbox(i18n.t("app-show-new-tabular-form")):
    admits_modulo = 1
    if not st.checkbox(i18n.t("app-show-daily-new-counts")):
        admits_modulo = 7
    table_df = build_table(
        df=m.admits_floor_df,
        labels=p.labels,
        modulo=admits_modulo)
    st.table(table_df)


st.subheader(i18n.t("app-admitted-patients-title"))
st.markdown(i18n.t("app-admitted-patients-text"))
census_chart = build_census_chart(alt=alt, census_floor_df=m.census_floor_df, max_y_axis=p.max_y_axis)
st.altair_chart(census_chart, use_container_width=True)
st.markdown(build_descriptions(chart=census_chart, labels=p.labels, prefix="census_"))
display_download_link(
    st,
    p,
    filename=f"{p.current_date}_projected_census.csv",
    df=m.census_df,
)

if st.checkbox(i18n.t("app-show-census-tabular-form")):
    census_modulo = 1
    if not st.checkbox(i18n.t("app-show-daily-census-counts")):
        census_modulo = 7
    table_df = build_table(
        df=m.census_floor_df,
        labels=p.labels,
        modulo=census_modulo)
    st.table(table_df)


st.subheader(i18n.t("app-SIR-title"))
st.markdown(i18n.t("app-SIR-text"))
sim_sir_w_date_chart = build_sim_sir_w_date_chart(alt=alt, sim_sir_w_date_floor_df=m.sim_sir_w_date_floor_df)
st.altair_chart(sim_sir_w_date_chart, use_container_width=True)
display_download_link(
    st,
    p,
    filename=f"{p.current_date}_sim_sir_w_date.csv",
    df=m.sim_sir_w_date_df,
)

if st.checkbox(i18n.t("app-show-sir-tabular-form")):
    table_df = build_table(
        df=m.sim_sir_w_date_floor_df,
        labels=p.labels)
    st.table(table_df)

display_footer(st)
