"""Compatibility and peak statistics for the latest two measure scenarios."""
import numpy as np
import pandas as pd

from app_support import fingerprint

WITHOUT = 'Zonder maatregelen'
WITH = 'Met maatregelen'
SCENARIOS = (WITHOUT, WITH)


def scenario(config):
    return WITH if config['inputs']['measures'] else WITHOUT


def comparison_key(config):
    """Exclude only measures; retain input bytes, actual rain and all settings."""
    return fingerprint(dict(config, inputs={key: value for key, value in config['inputs'].items()
                                            if key != 'measures'}))


def matching_runs(history, current_config):
    """Return the ordered pair only when both runs also match current inputs."""
    if not all(label in history for label in SCENARIOS):
        return None
    pair = tuple(history[label] for label in SCENARIOS)
    current_key = comparison_key(current_config)
    if all(comparison_key(saved['metadata']['configuration']) == current_key for saved in pair):
        return pair
    return None


def comparison_table(pair, cid):
    columns = {}
    for label, saved in zip(SCENARIOS, pair):
        run = saved['result']
        discharge = run.discharges[cid]
        peak_index = int(np.argmax(discharge))  # First occurrence of equal maxima.
        peak = float(discharge[peak_index])
        hours = ((run.times[peak_index]-run.times[0]).total_seconds()/3600
                 if peak > 0 else np.nan)
        columns[label] = [hours, peak, float(np.max(saved['depths'][cid]))]
    table = pd.DataFrame(columns, index=['Tijd tot piekdebiet [uren]', 'Piekdebiet [m³/s]',
                                         'Piekwaterdiepte [m]'])
    table['Verschil'] = table[WITH]-table[WITHOUT]
    return table
