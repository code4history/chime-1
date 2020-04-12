## 入力値のガイダンス
* **入院しているCOVID-19患者数:**
    現在、**この病院に**COVID-19で入院している患者数です。
    この値は病院の患者シェアや入院率と組み合わせて用いられ、地域の感染者数の総計を推定します。
* **倍加時間 (日数):**
    このパラメータは、アウトブレイク初期段階での新規症例の割合に影響します。
    米国病院協会 (AHA) は現在、7日間から10日間で患者数は倍増すると予測しています。
    これは、現在の状態で予想される倍加時間です。
    社会的接触や他の公衆衛生に影響する要素の減少を表現する方法として、_社会的距離戦略_パラメータを修正してください。
* **Social distancing (% reduction in person-to-person physical contact):**
    This parameter allows users to explore how reduction in interpersonal contact & transmission (hand-washing) might slow the rate of new infections.
    It is your estimate of how much social contact reduction is being achieved in your region relative to the status quo.
    While it is unclear how much any given policy might affect social contact (eg. school closures or remote work), this parameter lets you see how projections change with percentage reductions in social contact.
* **Hospitalization %(total infections):
    ** Percentage of **all** infected cases which will need hospitalization.
* **ICU %(total infections):**
    Percentage of **all** infected cases which will need to be treated in an ICU.
* **Ventilated %(total infections):**
    Percentage of **all** infected cases which will need mechanical ventilation.
* **Hospital Length of Stay:**
    Average number of days of treatment needed for hospitalized COVID-19 patients.
* **ICU Length of Stay:**
    Average number of days of ICU treatment needed for ICU COVID-19 patients.
* **Vent Length of Stay:**
    Average number of days of ventilation needed for ventilated COVID-19 patients.
* **Hospital Market Share (%):**
    The proportion of patients in the region that are likely to come to your hospital (as opposed to other hospitals in the region) when they get sick.
    One way to estimate this is to look at all of the hospitals in your region and add up all of the beds.
    The number of beds at your hospital divided by the total number of beds in the region times 100 will give you a reasonable starting estimate.
* **Regional Population:**
    Total population size of the catchment region of your hospital(s).
* **Currently Known Regional Infections**:
    The number of infections reported in your hospital's catchment region.
    This is only used to compute detection rate - **it will not change projections**.
    This input is used to estimate the detection rate of infected individuals.
