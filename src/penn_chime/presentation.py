"""effectful functions for streamlit io"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import i18n # type: ignore

from .defaults import Constants, RateLos
from .utils import add_date_column, dataframe_to_base64
from .parameters import Parameters

DATE_FORMAT = "%b, %d"  # see https://strftime.org
DOCS_URL = "https://code-for-philly.gitbook.io/chime"

FLOAT_INPUT_MIN = 0.001
FLOAT_INPUT_STEP = FLOAT_INPUT_MIN

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """


########
# Text #
########


def display_header(st, m, p):

    detection_prob_str = (
        "{detection_prob:.0%}".format(detection_prob=m.detection_probability)
        if m.detection_probability
        else "unknown"
    )

    infection_warning_str = (
        i18n.t("(Warning: The number of known infections is greater than...")
        if p.known_infected > m.infected
        else ""
    )

    infected_population_warning_str = (
        i18n.t("(Warning: The number of estimated infections is greater than...")
        if m.infected > p.population
        else ""
    )

    st.markdown(
        i18n.t("COVID-19 Hospital Impact Model for Epidemics..."),
        unsafe_allow_html=True,
    )
    st.markdown(
        i18n.t("[Documentation](https://code-for-philly.gitbook.io/chime/)...")
    )
    st.markdown(
        i18n.t("Admissions and Census calculations were...").format(p.change_date())
    )
    st.markdown(
        i18n.t("This tool was developed by...").format(docs_url=DOCS_URL)
    )

    st.markdown(
        i18n.t("The estimated number of currently infected...").format(
            total_infections=m.infected,
            initial_infections=p.known_infected,
            detection_prob_str=detection_prob_str,
            current_hosp=p.current_hospitalized,
            hosp_rate=p.hospitalized.rate,
            S=p.population,
            market_share=p.market_share,
            recovery_days=p.recovery_days,
            r_naught=m.r_naught,
            doubling_time=p.doubling_time,
            relative_contact_rate=p.relative_contact_rate,
            r_t=m.r_t,
            doubling_time_t=abs(m.doubling_time_t),
            impact_statement=(i18n.t("halves the infections every") if m.r_t < 1 else i18n.t("reduces the doubling time to")),
            daily_growth=m.daily_growth,
            daily_growth_t=m.daily_growth_t,
            docs_url=DOCS_URL,
            infection_warning_str=infection_warning_str,
            infected_population_warning_str=infected_population_warning_str
        )
    )

    return None


class Input:
    """Helper to separate Streamlit input definition from creation/rendering"""
    def __init__(self, st_obj, label, value, kwargs):
        self.st_obj = st_obj
        self.label = label
        self.value = value
        self.kwargs = kwargs

    def __call__(self):
        return self.st_obj(self.label, value=self.value, **self.kwargs)


class NumberInput(Input):
    def __init__(self, st_obj, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None):
        kwargs = dict(min_value=min_value, max_value=max_value, step=step, format=format, key=key)
        super().__init__(st_obj.number_input, label, value, kwargs)


class PercentInput(NumberInput):
    def __init__(self, st_obj, label, min_value=0.0, max_value=100.0, value=None, step=FLOAT_INPUT_STEP, format="%f", key=None):
        super().__init__(st_obj, label, min_value, max_value, value * 100.0, step, format, key)

    def __call__(self):
        return super().__call__() / 100.0


class CheckboxInput(Input):
    def __init__(self, st_obj, label, value=None, key=None):
        kwargs = dict(key=key)
        super().__init__(st_obj.checkbox, label, value, kwargs)


def display_sidebar(st, d: Constants) -> Parameters:
    # Initialize variables
    # these functions create input elements and bind the values they are set to
    # to the variables they are set equal to
    # it's kindof like ember or angular if you are familiar with those

    if d.known_infected < 1:
        raise ValueError(i18n.t("Known cases must be larger than one to enable predictions."))
    st_obj = st.sidebar
    current_hospitalized_input = NumberInput(
        st_obj,
        i18n.t("Currently Hospitalized COVID-19 Patients"),
        min_value=0,
        value=d.current_hospitalized,
        step=1,
        format="%i",
    )
    n_days_input = NumberInput(
        st_obj,
        i18n.t("Number of days to project"),
        min_value=30,
        value=d.n_days,
        step=1,
        format="%i",
    )
    doubling_time_input = NumberInput(
        st_obj,
        i18n.t("Doubling time before social distancing (days)"),
        min_value=FLOAT_INPUT_MIN,
        value=d.doubling_time,
        step=FLOAT_INPUT_STEP,
        format="%f",
    )
    relative_contact_pct_input = PercentInput(
        st_obj,
        i18n.t("Social distancing (% reduction in social contact)"),
        value=d.relative_contact_rate,
    )
    hospitalized_pct_input = PercentInput(
        st_obj,
        i18n.t("Hospitalization %(total infections)"),
        value=d.hospitalized.rate,
    )
    icu_pct_input = PercentInput(
        st_obj,
        i18n.t("ICU %(total infections)"),
        value=d.icu.rate,
    )
    ventilated_pct_input = PercentInput(
        st_obj,
        i18n.t("Ventilated %(total infections)"),
        value=d.ventilated.rate,
    )
    hospitalized_los_input = NumberInput(
        st_obj,
        i18n.t("Hospital Length of Stay"),
        min_value=0,
        value=d.hospitalized.length_of_stay,
        step=1,
        format="%i",
    )
    icu_los_input = NumberInput(
        st_obj,
        i18n.t("ICU Length of Stay"),
        min_value=0,
        value=d.icu.length_of_stay,
        step=1,
        format="%i",
    )
    ventilated_los_input = NumberInput(
        st_obj,
        i18n.t("Vent Length of Stay"),
        min_value=0,
        value=d.ventilated.length_of_stay,
        step=1,
        format="%i",
    )
    market_share_pct_input = PercentInput(
        st_obj,
        i18n.t("Hospital Market Share (%)"),
        min_value=FLOAT_INPUT_MIN,
        value=d.market_share,
    )
    population_input = NumberInput(
        st_obj,
        i18n.t("Regional Population"),
        min_value=1,
        value=d.region.population,
        step=1,
        format="%i",
    )
    known_infected_input = NumberInput(
        st_obj,
        i18n.t("Currently Known Regional Infections (only used to compute detection rate - does not change projections)"),
        min_value=0,
        value=d.known_infected,
        step=1,
        format="%i",
    )
    as_date_input = CheckboxInput(st_obj, i18n.t("Present result as dates instead of days"), value=False)
    max_y_axis_set_input = CheckboxInput(st_obj, i18n.t("Set the Y-axis on graphs to a static value"))
    max_y_axis_input = NumberInput(st_obj, i18n.t("Y-axis static value"), value=500, format="%i", step=25)

    # Build in desired order
    st.sidebar.markdown(i18n.t("### Regional Parameters [ℹ]({docs_url}/what-is-chime/parameters)").format(docs_url=DOCS_URL))
    population = population_input()
    market_share = market_share_pct_input()
    known_infected = known_infected_input()
    current_hospitalized = current_hospitalized_input()

    st.sidebar.markdown(i18n.t("### Spread and Contact Parameters [ℹ]({docs_url}/what-is-chime/parameters)")
                        .format(docs_url=DOCS_URL))
    doubling_time = doubling_time_input()
    relative_contact_rate = relative_contact_pct_input()

    st.sidebar.markdown(i18n.t("### Severity Parameters [ℹ]({docs_url}/what-is-chime/parameters)").format(docs_url=DOCS_URL))
    hospitalized_rate = hospitalized_pct_input()
    icu_rate = icu_pct_input()
    ventilated_rate = ventilated_pct_input()
    hospitalized_los = hospitalized_los_input()
    icu_los = icu_los_input()
    ventilated_los = ventilated_los_input()

    st.sidebar.markdown(i18n.t("### Display Parameters [ℹ]({docs_url}/what-is-chime/parameters)").format(docs_url=DOCS_URL))
    n_days = n_days_input()
    max_y_axis_set = max_y_axis_set_input()
    as_date = as_date_input()

    max_y_axis = None
    if max_y_axis_set:
        max_y_axis = max_y_axis_input()

    return Parameters(
        as_date=as_date,
        current_hospitalized=current_hospitalized,
        market_share=market_share,
        known_infected=known_infected,
        doubling_time=doubling_time,

        max_y_axis=max_y_axis,
        n_days=n_days,
        relative_contact_rate=relative_contact_rate,
        population=population,

        hospitalized=RateLos(hospitalized_rate, hospitalized_los),
        icu=RateLos(icu_rate, icu_los),
        ventilated=RateLos(ventilated_rate, ventilated_los),
    )


def show_more_info_about_this_tool(st, model, parameters, defaults, notes: str = ""):
    """a lot of streamlit writing to screen."""
    st.subheader(
        i18n.t("Discrete-time SIR modeling")
    )
    st.markdown(
        i18n.t("The model consists of individuals who are either...")
    )
    st.markdown(i18n.t("The dynamics are given by the following 3 equations."))

    st.latex("S_{t+1} = (-\\beta S_t I_t) + S_t")
    st.latex("I_{t+1} = (\\beta S_t I_t - \\gamma I_t) + I_t")
    st.latex("R_{t+1} = (\\gamma I_t) + R_t")

    st.markdown(
       i18n.t("To project the expected impact to Penn Medicine...")
    )
    st.latex("\\beta = \\tau \\times c")

    st.markdown(
        i18n.t("which is the transmissibility multiplied...").format(
            recovery_days=int(parameters.recovery_days)
        )
    )
    st.latex("R_0 = \\beta /\\gamma")

    st.markdown(
        i18n.t("$R_0$ gets bigger when...").format(
            doubling_time=parameters.doubling_time,
            recovery_days=parameters.recovery_days,
            r_naught=model.r_naught,
            relative_contact_rate=parameters.relative_contact_rate,
            doubling_time_t=model.doubling_time_t,
            r_t=model.r_t,
        )
    )
    st.latex("g = 2^{1/T_d} - 1")

    st.markdown(
        i18n.t("Since the rate of new infections in the SIR model...").format(
            notes=notes
        )
        + "- "
        + "| \n".join(
            f"{key} = {value} "
            for key, value in defaults.region.__dict__.items()
            if key != "_s"
        )
    )
    return None


def write_definitions(st):
    st.subheader(i18n.t("Guidance on Selecting Inputs"))
    st.markdown(
        i18n.t("This information has been moved to...").format(docs_url=DOCS_URL)
    )


def write_footer(st):
    st.subheader(i18n.t("References & Acknowledgements"))
    st.markdown(
        i18n.t("* AHA Webinar, Feb 26, James Lawler, MD, ...")
    )
    st.markdown(i18n.t("© 2020, The Trustees of the University of Pennsylvania"))


def show_additional_projections(
    st, alt, charting_func, model, parameters
):
    st.subheader(
        i18n.t("The number of infected and recovered individuals in the hospital catchment region at any given moment")
    )

    st.altair_chart(
        charting_func(
            alt,
            model=model,
            parameters=parameters
        ),
        use_container_width=True,
    )


##########
# Tables #
##########


def draw_projected_admissions_table(
    st, projection_admits: pd.DataFrame, labels, day_range, as_date: bool = False
):
    admits_table = projection_admits[np.mod(projection_admits.index, day_range) == 0].copy()
    admits_table["day"] = admits_table.index
    admits_table.index = range(admits_table.shape[0])
    admits_table = admits_table.fillna(0).astype(int)

    if as_date:
        admits_table = add_date_column(
            admits_table, drop_day_column=True, date_format=DATE_FORMAT
        )
    admits_table.rename(labels)
    st.table(admits_table)
    return None


def draw_census_table(st, census_df: pd.DataFrame, labels, day_range, as_date: bool = False):
    census_table = census_df[np.mod(census_df.index, day_range) == 0].copy()
    census_table.index = range(census_table.shape[0])
    census_table.loc[0, :] = 0
    census_table = census_table.dropna().astype(int)

    if as_date:
        census_table = add_date_column(
            census_table, drop_day_column=True, date_format=DATE_FORMAT
        )

    census_table.rename(labels)
    st.table(census_table)
    return None


def draw_raw_sir_simulation_table(st, model, parameters):
    as_date = parameters.as_date
    projection_area = model.raw_df
    infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
    infect_table.index = range(infect_table.shape[0])
    infect_table["day"] = infect_table.day.astype(int)

    if as_date:
        infect_table = add_date_column(
            infect_table, drop_day_column=True, date_format=DATE_FORMAT
        )

    st.table(infect_table)
    build_download_link(st,
        filename="raw_sir_simulation_data.csv",
        df=projection_area,
        parameters=parameters
    )

def build_download_link(st, filename: str, df: pd.DataFrame, parameters: Parameters):
    if parameters.as_date:
        df = add_date_column(df, drop_day_column=True, date_format="%Y-%m-%d")

    csv = dataframe_to_base64(df)
    st.markdown("""
        <a download="{filename}" href="data:file/csv;base64,{csv}">{description}</a>
""".format(csv=csv,filename=filename, description=i18n.t("Download full table as CSV")), unsafe_allow_html=True)
