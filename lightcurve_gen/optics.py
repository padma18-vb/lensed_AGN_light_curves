"""
Instrument-optics and photometric-error helpers.

These are plain functions (no class dependency) except where noted —
``compute_analytical_error`` and ``blending_error`` take the owning
``LightCurveGenerator`` instance (``gen``) as their first argument because
they read several scalar instrument settings off it; that's simpler than
exploding each into a 4-5 argument function for no real benefit.
"""

import numpy as np
import astropy.units as au


def diffraction_limited_fwhm_arcsec(wavelength_nm, diameter_m):
    """Diffraction-limited PSF FWHM (arcsec) for a circular aperture."""
    wavelength = (wavelength_nm * au.nm).to(au.m)
    diameter = diameter_m * au.m
    theta_rad = 1.028 * (wavelength / diameter).decompose().value
    return theta_rad * 206265.0


def compute_analytical_error(gen, magnitudes):
    """Photon-noise magnitude error, scaled from a reference telescope/zeropoint."""
    area_ratio = (gen.reference_diameter_m / gen.telescope_diameter_m) ** 2
    delta_m = 2.5 * np.log10(area_ratio)
    new_zp = gen.reference_zeropoint - delta_m
    signal = gen.exposure_time * 10 ** (-0.4 * (magnitudes - new_zp))
    snr = signal / np.sqrt(signal)
    error = 2.5 * np.log10(1 + 1 / snr)
    print('average error: ', np.mean(error))
    return error


def blending_error(gen, magnitudes):
    """Analytical deblending error model derived from Dux et al. 2025
    (credit: Martin Millon)."""
    # smallest image separation
    if gen.closest_image_separation is None:
        raise ValueError("Closest image separation is not available.")
    mag_err = 1e-3 * 10 ** (-4.25) * 10 ** (0.27 * magnitudes) * gen.closest_image_separation ** (-0.84)
    # 10^(-4.25±0.50)  ×  10^(0.2695±0.0248 × mag)  ×  sep^(-0.84±0.12)
    return mag_err  # in magnitudes


def closest_image_separation_from_positions(image_positions_ra, image_positions_dec):
    """Minimum angular separation between any pair of lensed images.

    Parameters
    ----------
    image_positions_ra : array_like
        Right-ascension positions of the images in arcseconds (or any
        self-consistent angular unit).
    image_positions_dec : array_like
        Declination positions in the same units.

    Returns
    -------
    float
        Minimum pairwise separation. If only two images are supplied the
        unique separation is returned directly.
    """
    ra = np.asarray(image_positions_ra)
    dec = np.asarray(image_positions_dec)
    if len(ra) == 2:
        return float(np.sqrt((ra[0] - ra[1]) ** 2 + (dec[0] - dec[1]) ** 2))
    separations = [
        np.sqrt((ra[i] - ra[j]) ** 2 + (dec[i] - dec[j]) ** 2)
        for i in range(len(ra))
        for j in range(i + 1, len(ra))
    ]
    pos_seps = [s for s in separations if s > 0]
    return float(np.min(pos_seps))
