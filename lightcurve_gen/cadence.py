"""
Observation-time (cadence) helpers: regular grid, user-supplied CSV, and
Rubin scheduler cadences, plus the shared baseline-day cutoff.

``get_random_rubin_ra_dec`` is a pure function (no instance state). The
rest take the owning ``LightCurveGenerator`` instance (``gen``) since they
need ``band_names``/``baseline_days``/``total_time``.
"""

import numpy as np
import pandas as pd
import astropy.units as au
import astropy.coordinates as coord

from rubin_sim_pipeline import get_rubin_cadence


def get_random_rubin_ra_dec(N=1):
    ra_points = coord.Angle(np.random.uniform(low=0, high=360, size=N) * au.degree)
    ra_points = ra_points.wrap_at(180 * au.degree)
    p = (
        np.sin(np.random.uniform(low=-72, high=12, size=N) * au.deg) - np.sin(-72 * au.deg)
    ) / (np.sin(12 * au.deg) - np.sin(-72 * au.deg))
    dec_points = coord.Angle(
        ((((np.arcsin(2 * p - 1).to(au.deg) + 90 * au.deg) / (180 * au.deg)) * 84) - 72)
        * au.deg
    )
    return ra_points, dec_points


def _apply_baseline_cut(gen, snapshot_timestamps):
    if gen.baseline_days is None:
        return {band: np.array(times, dtype=float) for band, times in snapshot_timestamps.items()}
    return {
        band: np.array(times, dtype=float)[np.array(times, dtype=float) <= gen.baseline_days]
        for band, times in snapshot_timestamps.items()
    }


def get_rubin_observation_times_properties(gen, ra_points, dec_points,
                                            columns_required=['observationStartMJD', 'filter']):
    rubin_df_output = get_rubin_cadence(ra_points.value, dec_points.value)
    missing_columns = [col for col in columns_required if col not in rubin_df_output.columns]
    print(f"Rubin cadence DataFrame columns: {rubin_df_output.columns.tolist()}")
    if missing_columns:
        raise KeyError(f"Missing required columns from Rubin cadence DataFrame: {missing_columns}")
    # Convert MJD to relative days from first observation, grouped by filter/band
    rubin_df_output = rubin_df_output.sort_values('observationStartMJD')
    t0 = rubin_df_output['observationStartMJD'].min()
    rubin_df_output['time'] = rubin_df_output['observationStartMJD'] - t0
    # Map Rubin single-letter filters to band_names
    snapshot_timestamps = {}
    for band_name in gen.band_names:
        # band_name may be 'u','g','r',... or a longer name — match on last character
        rubin_filter = band_name[-1] if len(band_name) > 1 else band_name
        band_rows = rubin_df_output[rubin_df_output['filter'] == rubin_filter]
        times = np.sort(band_rows['time'].values.astype(float))
        snapshot_timestamps[band_name] = times
    return _apply_baseline_cut(gen, snapshot_timestamps)


def get_observation_times(gen, observation_mode, cadence_file=None, sampling_freq=3,
                           ra_points=None, dec_points=None):
    if observation_mode == 'regular_cadence':
        # we are basically creating a random offset in times when different bands are observed
        offset = np.random.choice(np.arange(20), len(gen.band_names))
        snapshot_timestamps = {
            band: np.arange(offset[i], gen.total_time, sampling_freq)
            for i, band in enumerate(gen.band_names)
        }
        return _apply_baseline_cut(gen, snapshot_timestamps)
    elif observation_mode == 'user_cadence':
        if not cadence_file:
            raise ValueError("observation_mode='user_cadence' requires a cadence_file")
        cadence_df = pd.read_csv(cadence_file)
        if not {'band', 'time'}.issubset(cadence_df.columns):
            raise ValueError("cadence_file must contain columns: band, time")
        cadence_df = cadence_df[['band', 'time']].copy()
        cadence_df['time'] = cadence_df['time'].astype(float)
        grouped = cadence_df.groupby('band')['time'].apply(lambda s: np.sort(s.values))
        snapshot_timestamps = {
            band: grouped.get(band, np.array([], dtype=float))
            for band in gen.band_names
        }
        return _apply_baseline_cut(gen, snapshot_timestamps)
    elif observation_mode == 'rubin_cadence':
        if ra_points is None or dec_points is None:
            raise ValueError("observation_mode='rubin_cadence' requires ra_points and dec_points")
        return get_rubin_observation_times_properties(gen, ra_points, dec_points)
    else:
        raise ValueError(f"Invalid observation_mode: '{observation_mode}'. "
                         "Use 'regular_cadence', 'user_cadence', or 'rubin_cadence'.")
