from datetime import datetime
from math import ceil
from typing import Dict, Optional

from altair import Chart
import pandas as pd
import i18n
import numpy as np

from .constants import DATE_FORMAT
from .parameters import Parameters


def build_admits_chart(
    *, alt, admits_floor_df: pd.DataFrame, max_y_axis: Optional[int] = None
) -> Chart:
    """Build admits chart."""
    y_scale = alt.Scale()
    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)

    x = dict(shorthand="date:T", title=i18n.t("charts-date"), axis=alt.Axis(format=(DATE_FORMAT)))
    y = dict(shorthand="value:Q", title=i18n.t("charts-daily-admissions"), scale=y_scale)
    color = "key:N"
    tooltip = ["date:T", alt.Tooltip("value:Q", format=".0f", title="Admit"), "key:N"]

    # TODO fix the fold to allow any number of dispositions
    points = (
        alt.Chart()
        .transform_fold(fold=["admits_hospitalized", "admits_icu", "admits_ventilated"])
        .encode(x=alt.X(**x), y=alt.Y(**y), color=color, tooltip=tooltip)
        .mark_line(point=True)
        .encode(
            x=alt.X(**x),
            y=alt.Y(**y),
            color=color,
            tooltip=tooltip,
        )
    )
    bar = (
        alt.Chart()
        .encode(x=alt.X(**x))
        .transform_filter(alt.datum.day == 0)
        .mark_rule(color="black", opacity=0.35, size=2)
    )
    return (
        alt.layer(points, bar, data=admits_floor_df)
        .configure_legend(orient="bottom")
        .interactive()
    )


def build_census_chart(
    *, alt, census_floor_df: pd.DataFrame, max_y_axis: Optional[int] = None
) -> Chart:
    """Build census chart."""
    y_scale = alt.Scale()
    if max_y_axis:
        y_scale.domain = (0, max_y_axis)

    x = dict(shorthand="date:T", title=i18n.t("charts-date"), axis=alt.Axis(format=(DATE_FORMAT)))
    y = dict(shorthand="value:Q", title=i18n.t("charts-census"), scale=y_scale)
    color = "key:N"
    tooltip = ["date:T", alt.Tooltip("value:Q", format=".0f", title="Census"), "key:N"]

    # TODO fix the fold to allow any number of dispositions
    points = (
        alt.Chart()
        .transform_fold(fold=["census_hospitalized", "census_icu", "census_ventilated"])
        .encode(x=alt.X(**x), y=alt.Y(**y), color=color, tooltip=tooltip)
        .mark_line(point=True)
        .encode(
            x=alt.X(**x),
            y=alt.Y(**y),
            color=color,
            tooltip=tooltip,
        )
    )
    bar = (
        alt.Chart()
        .encode(x=alt.X(**x))
        .transform_filter(alt.datum.day == 0)
        .mark_rule(color="black", opacity=0.35, size=2)
    )
    return (
        alt.layer(points, bar, data=census_floor_df)
        .configure_legend(orient="bottom")
        .interactive()
    )


def build_sim_sir_w_date_chart(
    *, alt, sim_sir_w_date_floor_df: pd.DataFrame, max_y_axis: Optional[int] = None
) -> Chart:
    """Build sim sir w date chart."""
    y_scale = alt.Scale()
    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)

    x = dict(shorthand="date:T", title=i18n.t("charts-date"), axis=alt.Axis(format=(DATE_FORMAT)))
    y = dict(shorthand="value:Q", title=i18n.t("charts-count"), scale=y_scale)
    color = "key:N"
    tooltip = ["key:N", "value:Q"]

    # TODO fix the fold to allow any number of dispositions
    points = (
        alt.Chart()
        .transform_fold(fold=["susceptible", "infected", "recovered"])
        .encode(x=alt.X(**x), y=alt.Y(**y), color=color, tooltip=tooltip)
        .mark_line()
        .encode(
            x=alt.X(**x),
            y=alt.Y(**y),
            color=color,
            tooltip=tooltip,
        )
    )
    bar = (
        alt.Chart()
        .encode(x=alt.X(**x))
        .transform_filter(alt.datum.day == 0)
        .mark_rule(color="black", opacity=0.35, size=2)
    )
    return (
        alt.layer(points, bar, data=sim_sir_w_date_floor_df)
        .configure_legend(orient="bottom")
        .interactive()
    )


def build_descriptions(
    *,
    chart: Chart,
    labels: Dict[str, str],
    prefix: str = "",
    suffix: str = ""
) -> str:
    """

    :param chart: The alt chart to be used in finding max points
    :param suffix: The assumption is that the charts have similar column names.
                   The census chart adds " Census" to the column names.
                   Make sure to include a space or underscore as appropriate
    :return: Returns a multi-line string description of the results
    """
    messages = []

    cols = ["hospitalized", "icu", "ventilated"]
    asterisk = False
    day = "date" if "date" in chart.data.columns else "day"

    for col in cols:
        if chart.data[prefix+col].idxmax() + 1 == len(chart.data):
            asterisk = True

        # todo: bring this to an optional arg / i18n
        on = datetime.strftime(chart.data[day][chart.data[prefix+col].idxmax()], "%b %d")

        messages.append(
            i18n.t("charts-peak-template").format(
                labels[col],
                suffix,
                ceil(chart.data[prefix+col].max()),
                on,
                "*" if asterisk else "",
            )
        )

    if asterisk:
        messages.append(
            i18n.t("charts-max-upper-bound")
        )
    return "\n\n".join(messages)


def build_table(
    *, df: pd.DataFrame, labels: Dict[str, str], modulo: int = 1
) -> pd.DataFrame:
    table_df = df[np.mod(df.day, modulo) == 0].copy()
    table_df.date = table_df.date.dt.strftime(DATE_FORMAT)
    table_df.rename(labels)
    return table_df
