"""
Command-line entry point: reads one lens's parameters from a CSV, runs
LightCurveGenerator, and writes the result to HDF5.

Run as: python -m lightcurve_gen.cli --csv_file ... --lens_idx ... ...
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .generator import LightCurveGenerator
from .optics import closest_image_separation_from_positions
from . import io_hdf5


def _load_json_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_time_shifts_from_arrival_times(arrival_times):
    earliest_arrival_time = np.min(arrival_times)
    return arrival_times - earliest_arrival_time


def main():
    parser = argparse.ArgumentParser(description="Generate light curves and save results.")
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--lens_idx", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--observation_mode", type=str, required=False, default=None,
                        choices=["regular_cadence", "user_cadence"],
                        help="Required unless --rubin_only is set.")
    parser.add_argument("--passband_config", type=str, default=None)
    parser.add_argument("--cadence_file", type=str, default=None,
                        help="CSV with columns band,time for the primary (user) cadence.")
    parser.add_argument("--include_rubin", action='store_true',
                        help="If set, also generate Rubin-cadence light curves by querying "
                             "the Rubin scheduler at a random sky position, IN ADDITION to "
                             "the primary/user cadence.")
    parser.add_argument("--rubin_only", action='store_true',
                        help="If set, skip the primary/user cadence entirely and only generate "
                             "Rubin-cadence light curves. Implies --include_rubin, and makes "
                             "--observation_mode unnecessary.")
    parser.add_argument("--baseline_days", type=float, default=None)
    parser.add_argument("--telescope_diameter_m", type=float, default=1)
    parser.add_argument("--reference_diameter_m", type=float, default=2.4)
    parser.add_argument("--diffraction_wavelength_nm", type=float, default=606)
    parser.add_argument("--exposure_time", type=float, default=1200)
    parser.add_argument("--include_noisy_data", action='store_true')
    parser.add_argument("--output_dir", type=str, default="/pscratch/sd/v/vpadma/pycs/data",
                        help="Root output directory. Output is written to "
                             "<output_dir>/custom_telescope/<MMDDYYYY>/<dataset_name><lens_idx>.h5")

    args = parser.parse_args()

    if not args.rubin_only and args.observation_mode is None:
        parser.error("--observation_mode is required unless --rubin_only is set")

    csv_file = args.csv_file
    lens_idx = args.lens_idx
    dataset_name = args.dataset_name
    observation_mode = args.observation_mode
    include_noisy_data = args.include_noisy_data
    passband_config = _load_json_dict(args.passband_config) if args.passband_config else None
    exposure_time = args.exposure_time

    data = pd.read_csv(csv_file, index_col=0).loc[lens_idx]
    num_images = int(data['num_ps_images'])
    kappa_star = np.array(data[[f'micro_kappa_star_{i}' for i in range(num_images)]])
    kappa = np.array(data[[f'micro_kappa_tot_{i}' for i in range(num_images)]])
    shear = np.array(data[[f'micro_shear_{i}' for i in range(num_images)]])
    source_redshift = data['point_source_redshift']
    deflector_redshift = data["deflector_redshift"]
    log_sf_u = data["log_sf"]
    log_tau_u = data["log_tau"]

    arrival_times = np.array(data[[f'image_{i}_arrival_time' for i in range(num_images)]])
    print('supplied arrival times: ', arrival_times)
    time_delays = get_time_shifts_from_arrival_times(arrival_times)
    print('obtained time-delays: ', time_delays)

    if passband_config:
        bands = list(passband_config.keys())
    else:
        bands = ['u', 'g', 'r', 'i', 'z', 'y']

    true_mags = data[[f'ps_{b}_mag_true' for b in bands]]
    magnitudes = np.array([
        [data[f'point_source_light_{b}_magnitude_{i}'] for i in range(num_images)]
        for b in bands
    ])
    bh_mass = data['black_hole_mass_exponent']
    inclination_angle = 0
    eddington_ratio = 0.15
    reference_zeropoint = 27.6

    # Closest image separation for deblending error
    ra_arr = np.array(data[[f"point_source_light_y_ra_image_{i}" for i in range(5)]]).astype(float)
    dec_arr = np.array(data[[f"point_source_light_y_dec_image_{i}" for i in range(5)]]).astype(float)
    ra_masked = ra_arr[~np.isnan(ra_arr)]
    dec_masked = dec_arr[~np.isnan(dec_arr)]
    closest_sep = closest_image_separation_from_positions(ra_masked, dec_masked)

    lc_generator = LightCurveGenerator(
        bands, num_images, deflector_redshift, kappa_star, kappa, shear, source_redshift,
        time_delays, magnitudes, true_mags, bh_mass, inclination_angle, eddington_ratio,
        log_tau_u, log_sf_u, include_low_snr=include_noisy_data,
        passband_config=passband_config,
        cadence_file=args.cadence_file,
        baseline_days=args.baseline_days,
        telescope_diameter_m=args.telescope_diameter_m,
        reference_diameter_m=args.reference_diameter_m,
        reference_zeropoint=reference_zeropoint,
        exposure_time=exposure_time,
        diffraction_wavelength_nm=args.diffraction_wavelength_nm,
        closest_image_separation=closest_sep,
    )

    lc_generator.generate_light_curves_from_snapshots(
        observation_mode=observation_mode,
        include_rubin=args.include_rubin,
        rubin_only=args.rubin_only,
    )

    new_output_dir = os.path.join(args.output_dir, 'custom_telescope', datetime.today().strftime("%m%d%Y"))
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        print('making new directory: ', new_output_dir)
    io_hdf5.save_data(lc_generator, os.path.join(new_output_dir, f'{dataset_name}{lens_idx}.h5'))


if __name__ == "__main__":
    main()
