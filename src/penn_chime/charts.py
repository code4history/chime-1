from altair import Chart  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import i18n  # type: ignore

from .parameters import Parameters
from .utils import add_date_column


def new_admissions_chart(
    alt, projection_admits: pd.DataFrame, parameters: Parameters, as_date: bool = False,
) -> Chart:
    """docstring"""
    plot_projection_days = parameters.n_days - 10
    max_y_axis = parameters.max_y_axis

    y_scale = alt.Scale()

    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    tooltip_dict = {False: "day", True: "date:T"}
    if as_date:
        projection_admits = add_date_column(projection_admits)
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}

    return (
        alt.Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=[i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")])
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
    alt, census: pd.DataFrame, parameters: Parameters, as_date: bool = False
) -> Chart:
    """docstring"""

    plot_projection_days = parameters.n_days - 10
    max_y_axis = parameters.max_y_axis
    if as_date:
        census = add_date_column(census)
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
        idx = "date:T"
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}
        idx = "day"

    y_scale = alt.Scale()

    if max_y_axis:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    return (
        alt.Chart(census.head(plot_projection_days))
        .transform_fold(fold=[i18n.t("Hospitalized"), i18n.t("ICU"), i18n.t("Ventilated")])
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
    alt, i: np.ndarray, r: np.ndarray, as_date: bool = False, max_y_axis: int = None
) -> Chart:
    dat = pd.DataFrame({i18n.t("Infected"): i, i18n.t("Recovered"): r})
    dat["day"] = dat.index
    if as_date:
        dat = add_date_column(dat)
        x_kwargs = {"shorthand": "date:T", "title": i18n.t("Date")}
    else:
        x_kwargs = {"shorthand": "day", "title": i18n.t("Days from today")}

    y_scale = alt.Scale()

    if max_y_axis is not None:
        y_scale.domain = (0, max_y_axis)
        y_scale.clamp = True

    return (
        alt.Chart(dat)
        .transform_fold(fold=[i18n.t("Infected"), i18n.t("Recovered")])
        .mark_line()
        .encode(
            x=alt.X(**x_kwargs),
            y=alt.Y("value:Q", title=i18n.t("Case Volume"), scale=y_scale),
            tooltip=["key:N", "value:Q"],
            color="key:N",
        )
        .interactive()
    )
