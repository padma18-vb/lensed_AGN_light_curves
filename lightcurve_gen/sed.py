"""
Passband transmission loading, quasar SED handling, and flux -> AB magnitude
conversion.

Note: we scale the outputs by the static flux map (the variable map flux is 
not calibrated)
We also extrapolate the SED in cases where we want to observe high-redshift AGN
in the UV. 
We looked a sample of light curves to make sure the variability was not overestimated
Please let us know if you have feedback/spot any bugs!
Contact: pv10@illinois.edu
"""

import numpy as np
import astropy.units as au
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from astropy.io import fits
import pandas as pd


def load_in_transmissions(gen):
    filter_data = {}
    for band_name in gen.band_names:
        config = gen.passband_config.get(band_name)
        if config is None:
            raise ValueError(f"No passband config for band '{band_name}'")
        kind = config.get('kind', 'file')
        if kind == 'file':
            file_path = config['file']
            filter_data[band_name] = np.loadtxt(file_path)
        elif kind == 'top_hat':
            lam_min = float(config['lambda_min_nm'])
            lam_max = float(config['lambda_max_nm'])
            resolution = float(config.get('resolution_nm', 1.0))
            lam = np.arange(lam_min, lam_max + resolution, resolution)
            transmission = np.ones_like(lam) * 0.7
            filter_data[band_name] = np.column_stack([lam, transmission])
        else:
            raise ValueError(f"Unsupported passband kind '{kind}' for band '{band_name}'")
    return filter_data


def frame_conversion(gen, quantity, mode='observer_to_rest'):
    if mode == 'observer_to_rest':
        return quantity / (1 + gen.source_redshift)
    elif mode == 'rest_to_observer':
        return quantity * (1 + gen.source_redshift)
    else:
        raise ValueError("Invalid mode. Use 'observer_to_rest' or 'rest_to_observer'.")


def read_in_quasar_fits(file):
    with fits.open(file) as hdul:
        quasar_spectra = hdul[1].data
    quasar_spectra_df = pd.DataFrame({
        'wavelength': quasar_spectra['WAVELENGTH'],
        'flux': quasar_spectra['FLUX']
    }).dropna()
    return quasar_spectra_df


def get_quasar_sed_at_source_redshift(gen, quasar_file='optical_nir_qso_sed_001.fits'):
    quasar_spectra_df = read_in_quasar_fits(quasar_file)
    lambda_sed = quasar_spectra_df['wavelength'].values  # Angstrom
    gen.f_nu_sed = np.array(3.34e4 * quasar_spectra_df['wavelength'] ** 2 * quasar_spectra_df['flux'])  # Jy
    gen.lambda_sed_in_obs_frame = frame_conversion(gen, lambda_sed, 'rest_to_observer')
    print(np.min(gen.lambda_sed_in_obs_frame), np.min(gen.obs_wavelength_nm * 10))
    if np.min(gen.lambda_sed_in_obs_frame) > np.min(gen.obs_wavelength_nm * 10):
        print('we interpolated down')
        index_where_lambda_is_smallest = np.argmin(gen.lambda_sed_in_obs_frame)
        lambda_sed = np.insert(lambda_sed, 0,
                               frame_conversion(gen, np.min(gen.lambda_sed_in_obs_frame), 'observer_to_rest') - 1000)
        print(gen.f_nu_sed[index_where_lambda_is_smallest])
        gen.f_nu_sed = np.insert(gen.f_nu_sed, 0, gen.f_nu_sed[index_where_lambda_is_smallest])
        gen.lambda_sed_in_obs_frame = frame_conversion(gen, lambda_sed, 'rest_to_observer')
    sorted_indices = np.argsort(gen.lambda_sed_in_obs_frame)
    gen.lambda_sed_in_obs_frame = gen.lambda_sed_in_obs_frame[sorted_indices]
    gen.f_nu_sed = gen.f_nu_sed[sorted_indices]
    lambda_sed = lambda_sed[sorted_indices]
    gen.fnu_interp = interp1d(lambda_sed, gen.f_nu_sed, bounds_error=False, kind='linear',
                               fill_value='extrapolate')
    gen.transmission = {}
    for band_name in gen.band_names:
        filter_data = gen.filter_data[band_name]
        lambda_filter = filter_data[:, 0] * 10
        T_lambda = filter_data[:, 1]
        T_interp = interp1d(lambda_filter, T_lambda, bounds_error=False, kind='linear', fill_value='extrapolate')
        transmission = T_interp(gen.lambda_sed_in_obs_frame)
        gen.transmission[band_name] = transmission


def convert_flux_arr_to_mag(gen, flux_arr, band):
    band_name = gen.band_names[band]
    pixel_size = gen.accretion_disk.pixel_size * au.m

    # --- Fixed reference band (band 0) anchors the SED for all bands ---
    ref_band = 0
    lsst_ref_rest = frame_conversion(gen, gen.obs_wavelength_nm[ref_band], 'observer_to_rest')
    lambda_ref_rest = lsst_ref_rest * 10  # Angstrom — fixed anchor

    # Convert disk flux to Jy using the reference band's observer-frame wavelength
    lambda_obs_ref_for_sed = gen.obs_wavelength_nm[ref_band] * 10 * au.AA
    with_units = (flux_arr * (au.W / au.m ** 2 / au.m)).to(au.W / au.m ** 2 / au.Angstrom) * pixel_size ** 2
    finite_mask = np.isfinite(with_units)
    finite_data = np.where(finite_mask, with_units, 0)
    lum_dist = gen.accretion_disk.lum_dist * au.m
    finite_data /= (4 * np.pi * lum_dist ** 2)
    finite_data = finite_data.to(au.erg / au.s / au.cm ** 2 / au.Angstrom)
    finite_data = finite_data.to(au.Jy, equivalencies=au.spectral_density(lambda_obs_ref_for_sed))
    finite_data = np.clip(finite_data, 0.0, None)

    if len(finite_data.shape) == 1:
        axis = None
    elif len(finite_data.shape) == 2:
        axis = (0, 1)
    elif len(finite_data.shape) == 3:
        axis = (1, 2)

    slice_sums = np.sum(finite_data, axis=axis)
    flux_rest_ref = slice_sums.reshape(len(slice_sums), 1)

    # Anchor SED at the fixed reference wavelength — not per-band
    fnu_ref_template = gen.fnu_interp(lambda_ref_rest)
    f_nu_scaled = flux_rest_ref * (gen.f_nu_sed / fnu_ref_template)
    f_nu_obs = frame_conversion(gen, f_nu_scaled, 'rest_to_observer')

    transmission = gen.transmission[band_name]
    fnu_obs_interp = interp1d(gen.lambda_sed_in_obs_frame, f_nu_obs, bounds_error=False,
                               fill_value='extrapolate', kind='linear')

    numerator = trapezoid(fnu_obs_interp(gen.lambda_sed_in_obs_frame) * transmission,
                          gen.lambda_sed_in_obs_frame)
    denominator = trapezoid(transmission, gen.lambda_sed_in_obs_frame)
    fnu_eff_Jy = numerator / denominator
    observed_mags = -2.5 * np.log10(fnu_eff_Jy / 3631)
    return observed_mags


def static_disk_magnitude(gen, band):
    """
    The unlensed, non-variable (quiescent) AB magnitude of the accretion
    disk in this band -- the physically self-consistent absolute
    brightness level the light curve should fluctuate around.

    This replaces tying a band's absolute magnitude to a user-supplied
    catalog value (or to some other band via a filter map): it computes
    the disk's own static surface-intensity map at this band's rest-frame
    wavelength (matching the existing convention in
    ``get_driving_signal``, which calls the same AMOEBA method the same
    way) and runs it through the same anchoring + passband-integration
    pipeline as ``convert_flux_arr_to_mag`` -- so every band, whether it
    matches a reference survey's filter or not, gets an absolute
    magnitude derived directly and consistently from the same disk model,
    luminosity distance, and passband, with no per-band external
    calibration needed.
    """
    rest_frame_wavelength_nm = frame_conversion(gen, gen.obs_wavelength_nm[band], 'observer_to_rest')
    emission = gen.accretion_disk.calculate_surface_intensity_map(rest_frame_wavelength_nm)
    flux_array = np.expand_dims(emission.flux_array, axis=0)  # -> shape (1, ny, nx), one "snapshot"
    return convert_flux_arr_to_mag(gen, flux_array, band)[0]
