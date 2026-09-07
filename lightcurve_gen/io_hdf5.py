"""
Writes the generator.py light curve data into a HDF5 and stores some metadata 
about the object.

Two noise models are supported per cadence group:
- The generic aperture/exposure-time model (``optics.compute_analytical_error``),
  used for the primary/user-defined telescope.
- photerr's ``LsstErrorModel`` combined in quadrature with the deblending
  error (``optics.blending_error``), used for the Rubin cadence -- ground-based
  imaging always suffers some deblending, so this is applied unconditionally
  whenever an LsstErrorModel is passed in, not gated behind an optional flag.
Please let us know if you have feedback/spot any bugs!
Contact: pv10@illinois.edu
"""

import numpy as np
import pandas as pd
import h5py

from . import optics


def _compute_lsst_and_blending_errors(gen, err_model, band_name, times, mags):
    """
    Run photerr's LsstErrorModel on this band's magnitudes, then combine its
    photometric error in quadrature with the deblending error. Mirrors the
    original LSST-only script's noise pipeline exactly.

    photerr can drop rows it can't produce a finite error for (e.g. below its
    detection limit) -- so times/mags/errors all come back at whatever
    (possibly shorter) length survives that filtering, together, not
    re-aligned against the original unfiltered arrays.
    """
    df_for_err = pd.DataFrame({'times': times, band_name: mags})
    errdf = err_model(df_for_err)
    errdf = errdf[np.isfinite(errdf).all(axis=1)]

    phot_errs = errdf[f'{band_name}_err'].to_numpy(dtype=float)
    mags_out = errdf[band_name].to_numpy(dtype=float)
    times_out = errdf['times'].to_numpy(dtype=float)

    blending_errs = optics.blending_error(gen, mags_out)
    final_errs = np.sqrt(phot_errs ** 2 + blending_errs ** 2)
    mags_out = gen.add_noise(mags_out, final_errs)

    return mags_out, final_errs, phot_errs, blending_errs, times_out


def _write_cadence_group(gen, f, group_name, lensed_container, ml_container, observation_list,
                          err_model=None):
    """
    Write one cadence's worth of light curves into an HDF5 group.

    err_model : photerr.LsstErrorModel or None
        If given, use it (+ deblending error, combined in quadrature) instead
        of the generic analytical error model for this cadence.
    """
    regular_group = f.create_group(group_name)
    agn_group = regular_group.create_group('lensed_mags')
    ml_group = regular_group.create_group('microlensed_and_lensed_mags')
    observation_group = regular_group.create_group('observation_dates')

    for img_idx, band_dict in lensed_container.items():
        img_grp = agn_group.create_group(f'image_{img_idx + 1}')
        for band, arr in enumerate(band_dict):
            band_name = gen.band_names[band]
            times_arr = np.asarray(observation_list[band_name], dtype=float)
            mags_arr = np.asarray(arr, dtype=float)

            if err_model is not None:
                mags_arr, final_errs, phot_errs, blending_errs, _ = _compute_lsst_and_blending_errors(
                    gen, err_model, band_name, times_arr, mags_arr
                )
                img_grp.create_dataset(band_name, data=mags_arr)
                img_grp.create_dataset(f'{band_name}_err', data=final_errs)
                img_grp.create_dataset(f'{band_name}_phot_err', data=phot_errs)
                img_grp.create_dataset(f'{band_name}_blending_err', data=blending_errs)
            else:
                if len(times_arr) != len(mags_arr):
                    raise ValueError(
                        f"[{group_name}] Length mismatch for band '{band_name}': "
                        f"times={len(times_arr)} vs mags={len(mags_arr)}"
                    )
                mag_errs = optics.compute_analytical_error(gen, mags_arr)
                mags_arr = gen.add_noise(mags_arr, mag_errs)
                img_grp.create_dataset(band_name, data=mags_arr)
                img_grp.create_dataset(f'{band_name}_err', data=mag_errs)

    for img_idx, band_dict in ml_container.items():
        img_grp = ml_group.create_group(f'image_mags_{img_idx + 1}')
        times_grp = observation_group.create_group(f'image_times_{img_idx + 1}')
        for band, arr in enumerate(band_dict):
            band_name = gen.band_names[band]
            times_arr = np.asarray(observation_list[band_name], dtype=float)
            mags_arr = np.asarray(arr, dtype=float)
            img_grp.create_dataset(f'{band_name}_pre_noise', data=mags_arr)

            if err_model is not None:
                mags_arr, final_errs, phot_errs, blending_errs, times_out = _compute_lsst_and_blending_errors(
                    gen, err_model, band_name, times_arr, mags_arr
                )
                print(f'[{group_name}] image {img_idx}; band {band_name}; average mags: ', np.mean(mags_arr))
                print(f'[{group_name}] image {img_idx}; band {band_name}; average photometric error: ',
                      np.mean(phot_errs))
                print(f'[{group_name}] image {img_idx}; band {band_name}; average deblending error: ',
                      np.mean(blending_errs))
                print(f'[{group_name}] image {img_idx}; band {band_name}; average final error: ',
                      np.mean(final_errs))
                img_grp.create_dataset(band_name, data=mags_arr)
                img_grp.create_dataset(f'{band_name}_err', data=final_errs)
                img_grp.create_dataset(f'{band_name}_phot_err', data=phot_errs)
                img_grp.create_dataset(f'{band_name}_blending_err', data=blending_errs)
                times_grp.create_dataset(band_name, data=times_out)
            else:
                if len(times_arr) != len(mags_arr):
                    raise ValueError(
                        f"[{group_name}] Length mismatch for band '{band_name}': "
                        f"times={len(times_arr)} vs mags={len(mags_arr)}"
                    )
                mag_errs = optics.compute_analytical_error(gen, mags_arr)
                mags_arr = gen.add_noise(mags_arr, mag_errs)
                img_grp.create_dataset(band_name, data=mags_arr)
                img_grp.create_dataset(f'{band_name}_err', data=mag_errs)
                times_grp.create_dataset(band_name, data=times_arr)


def save_data(gen, file_path):
    with h5py.File(file_path, 'w') as f:
        metadata_group = f.create_group('metadata')
        metadata_group.create_dataset('band_names', data=gen.band_names)
        metadata_group.create_dataset('num_images', data=gen.num_images)
        metadata_group.create_dataset('deflector_redshift', data=gen.deflector_redshift)
        metadata_group.create_dataset('kappa_star', data=np.array(gen.kappa_star, dtype=np.float64))
        metadata_group.create_dataset('kappa', data=np.array(gen.kappa, dtype=np.float64))
        metadata_group.create_dataset('shear', data=np.array(gen.shear, dtype=np.float64))
        metadata_group.create_dataset('source_redshift', data=gen.source_redshift)
        metadata_group.create_dataset('time_delays', data=np.array(gen.time_delays, dtype=np.float64))
        metadata_group.create_dataset('true_magnitudes', data=np.array(gen.true_magnitudes, dtype=np.float64))
        metadata_group.create_dataset('bh_mass', data=gen.bh_mass)
        metadata_group.create_dataset('inclination_angle', data=gen.inclination_angle)
        metadata_group.create_dataset('eddington_ratio', data=gen.eddington_ratio)
        metadata_group.create_dataset('observer_frame_tau', data=gen.observer_frame_log_tau)
        metadata_group.create_dataset('observer_frame_sf', data=gen.observer_frame_log_sf)
        metadata_group.create_dataset('telescope_diameter_m', data=gen.telescope_diameter_m)
        metadata_group.create_dataset('reference_diameter_m', data=gen.reference_diameter_m)
        metadata_group.create_dataset('psf_fwhm_arcsec', data=gen.psf_fwhm_arcsec)
        metadata_group.create_dataset("closest_image_separation", data=gen.closest_image_separation)

        # Write user cadence only if it was generated (skipped in rubin_only mode)
        if gen.lensed_mags_container:
            _write_cadence_group(
                gen, f, 'user_cadence',
                gen.lensed_mags_container,
                gen.microlensed_mags_container,
                gen.observation_list
            )

        # Write Rubin cadence only if it was generated -- always with the
        # LSST error model + deblending error, never the generic model.
        # photerr is only imported here, so environments that never use
        # --include_rubin/--rubin_only don't need it installed at all.
        if gen.rubin_lensed_mags_container:
            from photerr import LsstErrorModel
            sig_lim = 1 if gen.include_low_snr else 5
            rubin_err_model = LsstErrorModel(nYrObs=1, nVisYr=1, sigLim=sig_lim, decorrelate=False)

            _write_cadence_group(
                gen, f, 'rubin_cadence',
                gen.rubin_lensed_mags_container,
                gen.rubin_microlensed_mags_container,
                gen.rubin_observation_list,
                err_model=rubin_err_model,
            )

    has_user = bool(gen.lensed_mags_container)
    has_rubin = bool(gen.rubin_lensed_mags_container)
    print(f"Saved to {file_path}")
    if has_user and has_rubin:
        print("  Contains: user_cadence + rubin_cadence")
    elif has_rubin:
        print("  Contains: rubin_cadence only")
    else:
        print("  Contains: user_cadence only")
