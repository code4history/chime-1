import altair as alt
import numpy as np
import pandas as pd
import i18n

from .defaults import Constants

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """


########
# Text #
########


def display_header(
    st,
    total_infections,
    initial_infections,
    detection_prob,
    current_hosp,
    hosp_rate,
    S,
    market_share,
    recovery_days,
    r_naught,
    doubling_time,
    relative_contact_rate,
    r_t,
    doubling_time_t,
):

    detection_prob_str = (
        "{detection_prob:.0%}".format(detection_prob=detection_prob)
        if detection_prob is not None
        else "unknown"
    )
    st.markdown(
        i18n.t("Penn Medicine - COVID-19 Hospital Impact Model for Epidemics..."),
        unsafe_allow_html=True,
    )
    st.markdown(
        i18n.t("This tool was developed by...")
    )

    st.markdown(
        i18n.t("The estimated number of currently infected...").format(
            total_infections=total_infections,
            initial_infections=initial_infections,
            detection_prob_str=detection_prob_str,
            current_hosp=current_hosp,
            hosp_rate=hosp_rate,
            S=S,
            market_share=market_share,
            recovery_days=recovery_days,
            r_naught=r_naught,
            doubling_time=doubling_time,
            relative_contact_rate=relative_contact_rate,
            r_t=r_t,
            doubling_time_t=doubling_time_t,
        )
    )

    return None


def show_more_info_about_this_tool(
    st,
    recovery_days,
    doubling_time,
    r_naught,
    relative_contact_rate,
    doubling_time_t,
    r_t,
    inputs: Constants,
    notes: str = ''
):
    """a lot of streamlit writing to screen."""
    st.subheader(
        i18n.t("Discrete-time SIR modeling")
    )
    st.markdown(
        i18n.t("The model consists of individuals who are either...")
    )
    st.markdown(i18n.t("""The dynamics are given by the following 3 equations."""))

    st.latex("S_{t+1} = (-\\beta S_t I_t) + S_t")
    st.latex("I_{t+1} = (\\beta S_t I_t - \\gamma I_t) + I_t")
    st.latex("R_{t+1} = (\\gamma I_t) + R_t")

    st.markdown(
       i18n.t("To project the expected impact to Penn Medicine...")
    )
    st.latex("\\beta = \\tau \\times c")

    st.markdown(
        i18n.t("which is the transmissibility multiplied...").format(
            recovery_days=int(recovery_days)
        )
    )
    st.latex("R_0 = \\beta /\\gamma")

    st.markdown(
        i18n.t("$R_0$ gets bigger when...").format(
            doubling_time=doubling_time,
            recovery_days=recovery_days,
            r_naught=r_naught,
            relative_contact_rate=relative_contact_rate,
            doubling_time_t=doubling_time_t,
            r_t=r_t,
        )
    )
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown(
        """
- Since the rate of new infections in the SIR model is $g = \\beta S - \\gamma$, and we've already computed $\\gamma$, $\\beta$ becomes a function of the initial population size of susceptible individuals.
$$\\beta = (g + \\gamma)$$.


### Initial Conditions

- {notes} \n
""".format(notes=notes) + "- " + "| \n".join(f"{key} = {value} " for key,value in inputs.region.__dict__.items() if key != '_s')
    )
    return None


def write_definitions(st):
    st.subheader("Guidance on Selecting Inputs")
    st.markdown(
        """* **Hospitalized COVID-19 Patients:** The number of patients currently hospitalized with COVID-19 **at your hospital(s)**. This number is used in conjunction with Hospital Market Share and Hospitalization % to estimate the total number of infected individuals in your region.
* **Doubling Time (days):** This parameter drives the rate of new cases during the early phases of the outbreak. The American Hospital Association currently projects doubling rates between 7 and 10 days. This is the doubling time you expect under status quo conditions. To account for reduced contact and other public health interventions, modify the _Social distancing_ input.
* **Social distancing (% reduction in person-to-person physical contact):** This parameter allows users to explore how reduction in interpersonal contact & transmission (hand-washing) might slow the rate of new infections. It is your estimate of how much social contact reduction is being achieved in your region relative to the status quo. While it is unclear how much any given policy might affect social contact (eg. school closures or remote work), this parameter lets you see how projections change with percentage reductions in social contact.
* **Hospitalization %(total infections):** Percentage of **all** infected cases which will need hospitalization.
* **ICU %(total infections):** Percentage of **all** infected cases which will need to be treated in an ICU.
* **Ventilated %(total infections):** Percentage of **all** infected cases which will need mechanical ventilation.
* **Hospital Length of Stay:** Average number of days of treatment needed for hospitalized COVID-19 patients.
* **ICU Length of Stay:** Average number of days of ICU treatment needed for ICU COVID-19 patients.
* **Vent Length of Stay:**  Average number of days of ventilation needed for ventilated COVID-19 patients.
* **Hospital Market Share (%):** The proportion of patients in the region that are likely to come to your hospital (as opposed to other hospitals in the region) when they get sick. One way to estimate this is to look at all of the hospitals in your region and add up all of the beds. The number of beds at your hospital divided by the total number of beds in the region times 100 will give you a reasonable starting estimate.
* **Regional Population:** Total population size of the catchment region of your hospital(s).
* **Currently Known Regional Infections**: The number of infections reported in your hospital's catchment region. This is only used to compute detection rate - **it will not change projections**. This input is used to estimate the detection rate of infected individuals.
    """
    )


def write_footer(st):
    st.subheader("References & Acknowledgements")
    st.markdown(
        """* AHA Webinar, Feb 26, James Lawler, MD, an associate professor University of Nebraska Medical Center, What Healthcare Leaders Need To Know: Preparing for the COVID-19
* We would like to recognize the valuable assistance in consultation and review of model assumptions by Michael Z. Levy, PhD, Associate Professor of Epidemiology, Department of Biostatistics, Epidemiology and Informatics at the Perelman School of Medicine
    """
    )
    st.markdown("© 2020, The Trustees of the University of Pennsylvania")


##########
# Charts #
##########


def new_admissions_chart(
    alt, projection_admits: pd.DataFrame, plot_projection_days: int
) -> alt.Chart:
    """docstring"""
    projection_admits = projection_admits.rename(
        columns={"hosp": "Hospitalized", "icu": "ICU", "vent": "Ventilated"}
    )
    return (
        alt.Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=["Hospitalized", "ICU", "Ventilated"])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title="Days from today"),
            y=alt.Y("value:Q", title="Daily admissions"),
            color="key:N",
            tooltip=[
                "day",
                alt.Tooltip("value:Q", format=".0f", title="Admissions"),
                "key:N",
            ],
        )
        .interactive()
    )


def admitted_patients_chart(alt, census: pd.DataFrame, plot_projection_days: int) -> alt.Chart:
    """docstring"""
    census = census.rename(
        columns={
            "hosp": "Hospital Census",
            "icu": "ICU Census",
            "vent": "Ventilated Census",
        }
    )

    return (
        alt.Chart(census.head(plot_projection_days))
        .transform_fold(fold=["Hospital Census", "ICU Census", "Ventilated Census"])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title="Days from today"),
            y=alt.Y("value:Q", title="Census"),
            color="key:N",
            tooltip=[
                "day",
                alt.Tooltip("value:Q", format=".0f", title="Census"),
                "key:N",
            ],
        )
        .interactive()
    )


def additional_projections_chart(alt, i: np.ndarray, r: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({"Infected": i, "Recovered": r})

    return (
        alt.Chart(dat.reset_index())
        .transform_fold(fold=["Infected", "Recovered"])
        .mark_line()
        .encode(
            x=alt.X("index", title="Days from today"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"],
            color="key:N",
        )
        .interactive()
    )


def show_additional_projections(st, alt, charting_func, i, r):
    st.subheader(
        "The number of infected and recovered individuals in the hospital catchment region at any given moment"
    )

    st.altair_chart(charting_func(alt, i, r), use_container_width=True)


##########
# Tables #
##########


def draw_projected_admissions_table(st, projection_admits: pd.DataFrame):
    admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
    admits_table["day"] = admits_table.index
    admits_table.index = range(admits_table.shape[0])
    admits_table = admits_table.fillna(0).astype(int)

    st.table(admits_table)
    return None

def draw_census_table(st, census_df: pd.DataFrame):
    census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
    census_table.index = range(census_table.shape[0])
    census_table.loc[0, :] = 0
    census_table = census_table.dropna().astype(int)

    st.table(census_table)
    return None


def draw_raw_sir_simulation_table(st, n_days, s, i, r):
    days = np.array(range(0, n_days + 1))
    data_list = [days, s, i, r]
    data_dict = dict(zip(["day", "susceptible", "infections", "recovered"], data_list))
    projection_area = pd.DataFrame.from_dict(data_dict)
    infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
    infect_table.index = range(infect_table.shape[0])

    st.table(infect_table)
