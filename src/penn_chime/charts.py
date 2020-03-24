
from math import ceil
import datetime

from altair import Chart  # type: ignore
import pandas as pd  # type: ignore
<<<<<<< HEAD
import numpy as np  # type: ignore
import i18n  # type: ignore
=======
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0

from .parameters import Parameters
from .utils import add_date_column
from .presentation import DATE_FORMAT


def new_admissions_chart(
    alt, projection_admits: pd.DataFrame, parameters: Parameters
) -> Chart:
    """docstring"""
    plot_projection_days = parameters.n_days - 10
    max_y_axis = parameters.max_y_axis
    as_date = parameters.as_date

    y_scale = alt.Scale()

    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
<<<<<<< HEAD
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
=======
        x_kwargs = {"shorthand": "date:T", "title": "Date", "axis": alt.Axis(format=(DATE_FORMAT))}
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}

    # TODO fix the fold to allow any number of dispositions
    return (
        alt.Chart(projection_admits.head(plot_projection_days))
<<<<<<< HEAD
        .transform_fold(fold=[i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")])
=======
        .transform_fold(fold=["hospitalized", "icu", "ventilated"])
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
        .mark_line(point=True)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title=i18n.t("Daily admissions"), scale=y_scale),
            color="key:N",
            tooltip=[
                tooltip_dict[as_date],
                alt.Tooltip("value:Q", format=".0f", title=i18n.t("Admissions")),
                "key:N",
            ],
        )
        .interactive()
    )


def admitted_patients_chart(
    alt, census: pd.DataFrame, parameters: Parameters
) -> Chart:
    """docstring"""

    plot_projection_days = parameters.n_days - 10
    max_y_axis = parameters.max_y_axis
    as_date = parameters.as_date
    if as_date:
        census = add_date_column(census)
<<<<<<< HEAD
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
=======
        x_kwargs = {"shorthand": "date:T", "title": "Date", "axis": alt.Axis(format=(DATE_FORMAT))}
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
        idx = "date:T"
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}
        idx = "day"

    y_scale = alt.Scale()

    if max_y_axis:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    # TODO fix the fold to allow any number of dispositions
    return (
        alt.Chart(census.head(plot_projection_days))
<<<<<<< HEAD
        .transform_fold(fold=[i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")])
=======
        .transform_fold(fold=["hospitalized", "icu", "ventilated"])
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
        .mark_line(point=True)
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title=i18n.t("Census"), scale=y_scale),
            color="key:N",
            tooltip=[
                idx,
                alt.Tooltip("value:Q", format=".0f", title=i18n.t("Census")),
                "key:N",
            ],
        )
        .interactive()
    )


def additional_projections_chart(
    alt, model, parameters
) -> Chart:
<<<<<<< HEAD
    i = parameters.infected_v
    r = parameters.recovered_v
    dat = pd.DataFrame({i18n.t("Infected"): i, i18n.t("Recovered"): r})
=======

    # TODO use subselect of df_raw instead of creating a new df
    raw_df = model.raw_df
    dat = pd.DataFrame({
        "infected": raw_df.infected,
        "recovered": raw_df.recovered
    })
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
    dat["day"] = dat.index

    as_date = parameters.as_date
    max_y_axis = parameters.max_y_axis

    if as_date:
        dat = add_date_column(dat)
<<<<<<< HEAD
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
=======
        x_kwargs = {"shorthand": "date:T", "title": "Date", "axis": alt.Axis(format=(DATE_FORMAT))}
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}

    y_scale = alt.Scale()

    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    return (
        alt.Chart(dat)
<<<<<<< HEAD
        .transform_fold(fold=[i18n.t("Infected"), i18n.t("Recovered")])
=======
        .transform_fold(fold=["infected", "recovered"])
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
        .mark_line()
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title=i18n.t("Case Volume"), scale=y_scale),
            tooltip=["key:N", "value:Q"],
            color="key:N",
        )
        .interactive()
    )


def chart_descriptions(chart: Chart, labels, suffix: str = ""):
    """

    :param chart: Chart: The alt chart to be used in finding max points
    :param suffix: str: The assumption is that the charts have similar column names.
                   The census chart adds " Census" to the column names.
                   Make sure to include a space or underscore as appropriate
    :return: str: Returns a multi-line string description of the results
    """
    messages = []

<<<<<<< HEAD
    cols = [i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")]
=======
    cols = ["hospitalized", "icu", "ventilated"]
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
    asterisk = False
    day = "date" if "date" in chart.data.columns else "day"

    for col in cols:
        if chart.data[col].idxmax() + 1 == len(chart.data):
            asterisk = True

        on = chart.data[day][chart.data[col].idxmax()]
        if day == "date":
            on = datetime.datetime.strftime(on, "%b %d")  # todo: bring this to an optional arg / i18n
        else:
            on += 1  # 0 index issue

        messages.append(
<<<<<<< HEAD
            i18n.t("{}{} peaks at {:,} on day {}{}").format(
                col,
=======
            "{}{} peaks at {:,} on day {}{}".format(
                labels[col],
>>>>>>> d9c1f27f013384aa0a4b0f4410b1302129b8e3a0
                suffix,
                ceil(chart.data[col].max()),
                on,
                "*" if asterisk else "",
            )
        )

    if asterisk:
        messages.append(i18n.t("_* The max is at the upper bound of the data, and therefore may not be the actual max_"))
    return "\n\n".join(messages)
