"""Aligned time-series plots with Belgian local-time labels."""
import json

import pandas as pd

from comparison import SCENARIOS


def local_time(value):
    return pd.Timestamp(value).tz_convert('Europe/Brussels').strftime('%d-%m-%Y %H:%M %Z')


def time_axis(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Use epoch milliseconds for data, domain and ticks. This avoids temporal
    # coercion in Streamlit/Vega and preserves elapsed time across DST changes.
    # Only labels are localized, independently of the browser timezone.
    minutes = (end-start).total_seconds()/60
    step = next(m for m in [1, 5, 15, 30, 60, 180, 360, 720, 1440, 2880, 10080]
                if minutes/m <= 6)
    ticks = pd.date_range(start.tz_convert('UTC'), end.tz_convert('UTC'), freq=f'{step}min')
    labels = {str(t.value//1_000_000): t.tz_convert('Europe/Brussels').strftime('%d/%m %H:%M')
              for t in ticks}
    return dict(
        field='tijd', type='quantitative', title='Datum en tijd (Belgische lokale tijd)',
        scale=dict(domain=[start.value//1_000_000, end.value//1_000_000], nice=False, zero=False),
        axis=dict(values=[int(t) for t in labels],
                  labelExpr=f'{json.dumps(labels)}[toNumber(datum.value)]',
                  labels=True, ticks=True, tickSize=5,
                  labelAngle=0, labelOverlap=True, grid=True))


def chart_spec(times, values, title, unit, x_axis, color='#2878B5', *, interval_seconds=None):
    times = pd.DatetimeIndex(times)
    data = pd.DataFrame(dict(tijd=[t.value//1_000_000 for t in times],
                             lokale_tijd=[local_time(t) for t in times], waarde=values))
    spec = dict(
        data=dict(values=data.to_dict('records')), title=title, height=180,
        mark=dict(type='line', point=dict(filled=True, size=24, color=color),
                  strokeWidth=2, color=color, tooltip=True),
        encoding=dict(
            x=x_axis,
            y=dict(field='waarde', type='quantitative', title=unit,
                   scale=dict(zero=True), axis=dict(minExtent=65, maxExtent=65)),
            tooltip=[dict(field='lokale_tijd', type='nominal', title='Lokale tijd'),
                     dict(field='waarde', type='quantitative', title=unit)]))
    if interval_seconds is not None:
        data['interval_einde'] = data.tijd + interval_seconds*1000
        spec['data']['values'] = data.to_dict('records')
        spec['mark'] = dict(type='bar', orient='vertical', color=color, opacity=0.8, tooltip=True)
        spec['encoding']['x2'] = dict(field='interval_einde')
    return spec


def comparison_series(pair, cid, title, unit, x_axis, *, depth=False):
    records = []
    for label, saved in zip(SCENARIOS, pair):
        run = saved['result']
        times = run.times+pd.Timedelta(seconds=run.dt_s) if depth else run.times
        values = saved['depths'][cid] if depth else run.discharges[cid]
        series = chart_spec(times, values, title, unit, x_axis)
        records.extend(dict(row, scenario=label) for row in series['data']['values'])
    series['data']['values'] = records
    # Let the scenario encoding color both the lines and their points.
    series['mark'].pop('color')
    series['mark']['point'].pop('color')
    series['encoding']['color'] = dict(
        field='scenario', type='nominal', title=None,
        scale=dict(domain=list(SCENARIOS), range=['#D17B20', '#2878B5']),
        legend=dict(orient='top'))
    series['encoding']['tooltip'].insert(0, dict(field='scenario', type='nominal', title='Run'))
    return series


def result_charts(run, rain, depths, cid, name, *, comparison=None):
    end_times = run.times+pd.Timedelta(seconds=run.dt_s)
    x_axis = time_axis(run.times[0], end_times[-1])
    # Show the same dry tail as the model, instead of ending the rainfall plot
    # at the last uploaded/downloaded rain interval.
    rainfall = rain.set_index('datum').rf.reindex(run.times, fill_value=0.)
    spec = dict(
        vconcat=[
            chart_spec(run.times, rainfall, f'{name} · neerslag', 'Neerslag [mm/stap]', x_axis,
                       interval_seconds=run.dt_s),
            chart_spec(run.times, run.discharges[cid], f'{name} · uitgaand debiet',
                       'Debiet [m³/s]', x_axis, '#D17B20'),
            chart_spec(end_times, depths[cid], f'{name} · maximale waterdiepte',
                       'Waterdiepte [m boven terrein]', x_axis, '#259477')],
        resolve=dict(scale=dict(x='shared', y='independent')), spacing=20)
    if comparison is not None:
        spec['vconcat'][1] = comparison_series(
            comparison, cid, f'{name} · uitgaand debiet', 'Debiet [m³/s]', x_axis)
        spec['vconcat'][2] = comparison_series(
            comparison, cid, f'{name} · maximale waterdiepte',
            'Waterdiepte [m boven terrein]', x_axis, depth=True)
        spec['resolve']['scale']['color'] = 'shared'
    return spec
