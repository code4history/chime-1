*このツールはペンシルバニア州立大学の医療部門であるPenn Medicineの、[予測医療チーム](http://predictivehealthcare.pennmedicine.org/)によって開発されました。質問とコメントは、我々の[コンタクトページ](http://predictivehealthcare.pennmedicine.org/contact/)をご覧ください。
ソースコードは[Github](https://github.com/CodeForPhilly/chime)で配布されています。
開発に参加されたい方は、我々の[Slack チャンネル](https://codeforphilly.org/chat?channel=covid19-chime-penn)にご参加ください！*

現在の予測感染者人数は**{total_infections:.0f}**人です。
この地域で確認された感染事例は**{initial_infections}**例で、検出率は**{detection_prob_str}**にあたります。この結果は入院患者数(**{current_hosp}**)、患者の入院割合(**{hosp_rate:.0%}**)、地域の人口(**{S}**)および病院の地域での患者シェア(**{market_share:.0%}**)を元に計算されています。

{infected_population_warning_str}

初期倍加時間が**{doubling_time}**日で、回復までの時間が**{recovery_days}**日であれば、**`R0`**は
**{r_naught:.2f}**となり、日々の増加率は**{daily_growth:.2f}%**となります。

**緩和**: アウトブレイク発生後の社会的接触が**{relative_contact_rate:.0%}**減少すると、
倍加時間は**{doubling_time_t:.1f}**日に改善され、実効$R_t$は**${r_t:.2f}$**となります。
