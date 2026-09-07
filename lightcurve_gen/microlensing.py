"""
Microlensing magnification-map construction, track extraction, and
per-snapshot magnification sampling.

``generate_mag_tracks``, ``_sample_magnifications_at_times`` and
``_build_microlensed_snapshots`` don't depend on the light curve generator 
class. 

Function generating the micorlensing maps do need access to the light curve
generator class (for kappa_star, shear etc).
Please let us know if you have feedback/spot any bugs!
Contact: pv10@illinois.edu
"""

import numpy as np
import astropy.constants as ac
import astropy.units as au
from scipy.interpolate import interp1d
from skimage.transform import rescale

import amoeba.Util.util as util
from amoeba.Classes.magnification_map import MagnificationMap
from microlensing.IPM.ipm import IPM


def get_mag_map_ipms(gen, half_length=25, num_pixels=5000):
    ipms = {}
    assert len(gen.kappa) == len(gen.kappa_star)
    assert len(gen.kappa) == len(gen.shear)
    if np.nansum(gen.kappa_star > gen.kappa) >= 1:
        gen.kappa_star[gen.kappa_star > gen.kappa] = 0.3 * gen.kappa[gen.kappa_star > gen.kappa]
    for i in range(gen.num_images):
        ipm = IPM(verbose=0, kappa_tot=gen.kappa[i], shear=gen.shear[i], kappa_star=gen.kappa_star[i],
                  rectangular=True, approx=True, half_length_y1=half_length, half_length_y2=half_length,
                  mass_function='kroupa', m_lower=0.08, m_upper=100,
                  num_pixels_y1=num_pixels, num_pixels_y2=num_pixels,
                  write_stars=True)
        ipm.run()
        ipms[f'ipm_{i}'] = ipm
    return ipms


def generate_amoeba_mag_map(gen, image_ind):
    ipm_x = gen.ipms[f'ipm_{image_ind}']
    MagMap = MagnificationMap(
        redshift_source=gen.source_redshift,
        redshift_lens=gen.deflector_redshift,
        magnification_array=ipm_x.magnifications,
        convergence=gen.kappa[image_ind],
        shear=gen.shear[image_ind],
        mean_microlens_mass_in_kg=ipm_x.stars.mean_mass_actual * ac.M_sun.to(au.kg),
        total_microlens_einstein_radii=ipm_x.half_length_y1,
        OmM=0.3,
        H0=70
    )
    return MagMap


def generate_mag_tracks(specific_mag_tracks, image_ind, mag_map_size=3000, border=200):
    if specific_mag_tracks is None:
        return (np.random.randint(border, mag_map_size - border),
                np.random.randint(border, mag_map_size - border))
    else:
        return specific_mag_tracks[image_ind]


def _extract_microlensing_track(gen, convolved_map, specific_mag_tracks, image_ind):
    """
    Extract a valid microlensing path, retrying up to 10 times with new
    start positions if the track leaves the map.
    baseline_days controls how many years the track covers.
    """
    baseline_years = max(gen.baseline_days / 365.0, 1.0)
    x_start_pos, y_start_pos = 300, 300
    max_tries = 10
    for attempt in range(max_tries):
        try:
            result = util.extract_path_on_microlensing_map(
                convolved_map.magnification_array,
                convolved_map.pixel_size,
                700,
                baseline_years,          # years — matches baseline_days exactly
                convolved_map.pixel_shift,
                x_start_position=x_start_pos,
                y_start_position=y_start_pos,
                phi_travel_direction=45
            )
            # Guard against the "leaving array" case that returns a scalar
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise TypeError(f"Expected (x_pos, y_pos) tuple, got: {type(result)}")
            x_pos, y_pos = result
            return x_pos, y_pos
        except TypeError as e:
            print(f"Track attempt {attempt + 1}/{max_tries} failed: {e}")
            start = generate_mag_tracks(specific_mag_tracks, image_ind)
            if isinstance(start, (tuple, list)) and len(start) == 2:
                x_start_pos, y_start_pos = start
            else:
                rng = np.random.default_rng(attempt)
                map_size = convolved_map.magnification_array.shape[0]
                x_start_pos = int(rng.integers(100, map_size - 100))
                y_start_pos = int(rng.integers(100, map_size - 100))
            print(f"  Retrying with start position ({x_start_pos}, {y_start_pos})")

    raise RuntimeError(
        f"Could not extract a valid microlensing track after {max_tries} attempts for image {image_ind}. "
        "Consider a larger magnification map or different start positions."
    )


def _sample_magnifications_at_times(x_pos, y_pos, timestamps, baseline_years,
                                     mag_map, pixel_ratio, snapshot_shape):
    """
    Interpolate the pre-extracted track (x_pos, y_pos) at the given timestamps
    and pull per-snapshot magnification sub-arrays from the mag map.
    """
    path_times = np.linspace(0, baseline_years, len(x_pos))
    x_interp = interp1d(path_times, x_pos, kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_interp = interp1d(path_times, y_pos, kind='cubic', bounds_error=False, fill_value='extrapolate')
    obs_times_years = (timestamps - timestamps[0]) / 365.25
    x_pos_obs = x_interp(obs_times_years)
    y_pos_obs = y_interp(obs_times_years)
    flux_array_rescaled = rescale(snapshot_shape, pixel_ratio)
    side_length = flux_array_rescaled.shape[0]
    magnification_maps = util.pull_subarray_from_grid(
        mag_map.magnification_array, x_pos_obs, y_pos_obs, side_length, side_length
    )
    return magnification_maps, side_length


def _build_microlensed_snapshots(snapshots_array, magnification_maps, pixel_ratio):
    microlensed_snapshots = []
    for j in range(len(snapshots_array)):
        sp = snapshots_array[j]
        flux_array_rescaled = rescale(sp, pixel_ratio)
        new_total_flux = np.nansum(flux_array_rescaled)
        original_total_flux = np.nansum(sp)
        flux_array_rescaled *= original_total_flux / new_total_flux
        disk_response_and_micro = flux_array_rescaled * magnification_maps[j]
        microlensed_snapshots.append(disk_response_and_micro)
    return np.array(microlensed_snapshots)
