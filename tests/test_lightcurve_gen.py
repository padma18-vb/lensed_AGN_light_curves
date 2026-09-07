"""
Unit tests for lightcurve_gen.

These tests exercise the package's telescope-agnostic logic (cadence
generation, error models, baseline cuts, SED anchoring, HDF5 output) using
lightweight fake "gen" objects instead of a real LightCurveGenerator, so
they don't require the real AMOEBA accretion-disk simulation, a real IPM
magnification map, or a real quasar SED FITS file. ``conftest.py`` stubs
the domain-specific packages (amoeba, microlensing, rubin_sim_pipeline,
photerr) so this suite can run without them installed.

This is not a substitute for an end-to-end integration test against the
real research environment (AMOEBA, IPM, a real lens catalogue CSV) --
that's a separate, heavier test that should be run wherever those
packages are actually available.
"""

import types

import numpy as np
import pytest

from lightcurve_gen import optics, sed, cadence, microlensing, io_hdf5


def make_fake_gen(**overrides):
    """A minimal stand-in for LightCurveGenerator, exposing just the
    attributes a given function under test needs. Pass overrides to set
    only what's relevant for a particular test."""
    gen = types.SimpleNamespace(
        band_names=['u', 'g', 'r'],
        total_time=1000,
        baseline_days=730,
        source_redshift=0.0,
        closest_image_separation=None,
        reference_diameter_m=2.4,
        telescope_diameter_m=1.0,
        reference_zeropoint=27.6,
        exposure_time=1200,
    )
    for key, value in overrides.items():
        setattr(gen, key, value)
    return gen


# ---------------------------------------------------------------------------
# optics.py
# ---------------------------------------------------------------------------

class TestDiffractionLimitedFwhm:
    def test_positive(self):
        fwhm = optics.diffraction_limited_fwhm_arcsec(wavelength_nm=600, diameter_m=1.0)
        assert fwhm > 0

    def test_scales_inversely_with_diameter(self):
        fwhm_small_aperture = optics.diffraction_limited_fwhm_arcsec(wavelength_nm=600, diameter_m=1.0)
        fwhm_large_aperture = optics.diffraction_limited_fwhm_arcsec(wavelength_nm=600, diameter_m=2.0)
        assert fwhm_large_aperture == pytest.approx(fwhm_small_aperture / 2, rel=1e-9)


class TestClosestImageSeparation:
    def test_two_images(self):
        ra = np.array([0.0, 3.0])
        dec = np.array([0.0, 4.0])
        sep = optics.closest_image_separation_from_positions(ra, dec)
        assert sep == pytest.approx(5.0)

    def test_multiple_images_returns_minimum_positive_separation(self):
        ra = np.array([0.0, 0.0, 10.0])
        dec = np.array([0.0, 1.0, 0.0])
        sep = optics.closest_image_separation_from_positions(ra, dec)
        assert sep == pytest.approx(1.0)


class TestComputeAnalyticalError:
    def test_brighter_source_has_smaller_error(self):
        gen = make_fake_gen()
        bright_mag = np.array([18.0])
        faint_mag = np.array([24.0])
        err_bright = optics.compute_analytical_error(gen, bright_mag)
        err_faint = optics.compute_analytical_error(gen, faint_mag)
        assert err_bright[0] < err_faint[0]

    def test_larger_telescope_reduces_error(self):
        mags = np.array([22.0])
        small_gen = make_fake_gen(telescope_diameter_m=1.0)
        large_gen = make_fake_gen(telescope_diameter_m=8.0)
        err_small = optics.compute_analytical_error(small_gen, mags)
        err_large = optics.compute_analytical_error(large_gen, mags)
        assert err_large[0] < err_small[0]


class TestBlendingError:
    def test_raises_without_closest_image_separation(self):
        gen = make_fake_gen(closest_image_separation=None)
        with pytest.raises(ValueError):
            optics.blending_error(gen, np.array([20.0]))

    def test_returns_positive_error_when_separation_available(self):
        gen = make_fake_gen(closest_image_separation=0.5)
        err = optics.blending_error(gen, np.array([20.0, 22.0]))
        assert np.all(err > 0)


# ---------------------------------------------------------------------------
# cadence.py
# ---------------------------------------------------------------------------

class TestApplyBaselineCut:
    def test_no_baseline_returns_all_times(self):
        gen = make_fake_gen(baseline_days=None)
        times = {'u': np.array([0.0, 100.0, 900.0])}
        result = cadence._apply_baseline_cut(gen, times)
        np.testing.assert_array_equal(result['u'], times['u'])

    def test_baseline_trims_late_observations(self):
        gen = make_fake_gen(baseline_days=500)
        times = {'u': np.array([0.0, 100.0, 900.0])}
        result = cadence._apply_baseline_cut(gen, times)
        np.testing.assert_array_equal(result['u'], np.array([0.0, 100.0]))


class TestGetObservationTimes:
    def test_regular_cadence_covers_all_bands(self):
        gen = make_fake_gen(band_names=['u', 'g'], total_time=100, baseline_days=None)
        result = cadence.get_observation_times(gen, 'regular_cadence', sampling_freq=10)
        assert set(result.keys()) == {'u', 'g'}
        for band, times in result.items():
            assert len(times) > 0
            assert np.all(np.diff(times) == 10)

    def test_user_cadence_reads_csv(self, tmp_path):
        cadence_file = tmp_path / "cadence.csv"
        cadence_file.write_text("band,time\nu,5\nu,1\ng,3\n")
        gen = make_fake_gen(band_names=['u', 'g'], baseline_days=None)
        result = cadence.get_observation_times(gen, 'user_cadence', cadence_file=str(cadence_file))
        np.testing.assert_array_equal(result['u'], np.array([1.0, 5.0]))
        np.testing.assert_array_equal(result['g'], np.array([3.0]))

    def test_user_cadence_requires_cadence_file(self):
        gen = make_fake_gen()
        with pytest.raises(ValueError):
            cadence.get_observation_times(gen, 'user_cadence', cadence_file=None)

    def test_invalid_observation_mode_raises(self):
        gen = make_fake_gen()
        with pytest.raises(ValueError):
            cadence.get_observation_times(gen, 'not_a_real_mode')


class TestGetRandomRubinRaDec:
    def test_ra_dec_within_survey_footprint(self):
        ra, dec = cadence.get_random_rubin_ra_dec(N=200)
        assert np.all(ra.deg >= -180) and np.all(ra.deg <= 180)
        assert np.all(dec.deg >= -72) and np.all(dec.deg <= 12)


# ---------------------------------------------------------------------------
# microlensing.py
# ---------------------------------------------------------------------------

class TestGenerateMagTracks:
    def test_random_start_within_bounds(self):
        for _ in range(20):
            x, y = microlensing.generate_mag_tracks(None, image_ind=0, mag_map_size=3000, border=200)
            assert 200 <= x <= 2800
            assert 200 <= y <= 2800

    def test_specific_tracks_are_passed_through(self):
        specific = {0: (111, 222), 1: (333, 444)}
        assert microlensing.generate_mag_tracks(specific, image_ind=1) == (333, 444)


# ---------------------------------------------------------------------------
# sed.py
# ---------------------------------------------------------------------------

class TestFrameConversion:
    def test_round_trip(self):
        gen = make_fake_gen(source_redshift=0.5)
        original = np.array([100.0, 200.0])
        rest = sed.frame_conversion(gen, original, 'observer_to_rest')
        back_to_observer = sed.frame_conversion(gen, rest, 'rest_to_observer')
        np.testing.assert_allclose(back_to_observer, original)

    def test_invalid_mode_raises(self):
        gen = make_fake_gen()
        with pytest.raises(ValueError):
            sed.frame_conversion(gen, 1.0, mode='sideways')


class TestConvertFluxArrToMagAnchoring:
    """
    Regression test for the SED-anchoring behavior: every band must be
    anchored to the SAME fixed reference band (band 0's rest wavelength),
    not to the wavelength of the band currently being converted. A prior
    rewrite of this codebase accidentally anchored each band to itself,
    which silently discards the disk model's simulated color information.
    This test fails if that regression is reintroduced.
    """

    def _build_gen(self):
        lambda_grid_aa = np.linspace(3000.0, 9000.0, 601)  # Angstrom, observer frame
        gen = make_fake_gen(
            band_names=['a', 'b'],
            obs_wavelength_nm=np.array([400.0, 800.0]),  # nm -> 4000, 8000 Angstrom
            source_redshift=0.0,
        )
        gen.accretion_disk = types.SimpleNamespace(pixel_size=1.0, lum_dist=1.0e10)
        gen.lambda_sed_in_obs_frame = lambda_grid_aa
        gen.f_nu_sed = np.ones_like(lambda_grid_aa)  # flat template shape
        gen.transmission = {
            'a': np.where((lambda_grid_aa > 3500) & (lambda_grid_aa < 4500), 1.0, 0.0),
            'b': np.where((lambda_grid_aa > 7500) & (lambda_grid_aa < 8500), 1.0, 0.0),
        }
        return gen

    def test_anchor_wavelength_is_the_same_for_every_band(self):
        gen = self._build_gen()
        calls = []

        def fake_fnu_interp(x):
            calls.append(float(np.asarray(x)))
            return 1.0

        gen.fnu_interp = fake_fnu_interp

        flux_arr = np.array([[[1e-10]], [[2e-10]]])  # shape (n_obs, ny, nx)
        sed.convert_flux_arr_to_mag(gen, flux_arr, band=0)
        sed.convert_flux_arr_to_mag(gen, flux_arr, band=1)

        assert len(calls) == 2
        # Same anchor wavelength regardless of which band was converted --
        # both calls should be band 0's rest-frame wavelength (4000 Angstrom
        # here, since source_redshift=0).
        assert calls[0] == pytest.approx(calls[1])
        assert calls[0] == pytest.approx(4000.0)

    def test_returns_finite_magnitudes(self):
        gen = self._build_gen()
        gen.fnu_interp = lambda x: 1.0
        flux_arr = np.array([[[1e-10]], [[2e-10]]])
        mags_a = sed.convert_flux_arr_to_mag(gen, flux_arr, band=0)
        mags_b = sed.convert_flux_arr_to_mag(gen, flux_arr, band=1)
        assert np.all(np.isfinite(mags_a))
        assert np.all(np.isfinite(mags_b))


class TestStaticDiskMagnitude:
    """
    static_disk_magnitude derives a band's absolute magnitude directly from
    the disk's own static (non-variable) emission, instead of a
    user-supplied catalog value or a filter-map anchor to some other band.
    """

    def _build_gen(self, calc_surface_intensity_map):
        lambda_grid_aa = np.linspace(3000.0, 9000.0, 601)  # Angstrom, observer frame
        gen = make_fake_gen(
            band_names=['a', 'b'],
            obs_wavelength_nm=np.array([400.0, 800.0]),  # nm -> 4000, 8000 Angstrom
            source_redshift=0.5,  # nonzero, so rest-frame != observer-frame wavelength
        )
        gen.accretion_disk = types.SimpleNamespace(
            pixel_size=1.0,
            lum_dist=1.0e10,
            calculate_surface_intensity_map=calc_surface_intensity_map,
        )
        gen.lambda_sed_in_obs_frame = lambda_grid_aa
        gen.f_nu_sed = np.ones_like(lambda_grid_aa)  # flat template shape
        gen.fnu_interp = lambda x: 1.0
        gen.transmission = {
            'a': np.where((lambda_grid_aa > 3500) & (lambda_grid_aa < 4500), 1.0, 0.0),
            'b': np.where((lambda_grid_aa > 7500) & (lambda_grid_aa < 8500), 1.0, 0.0),
        }
        return gen

    def test_calls_disk_with_rest_frame_wavelength(self):
        # get_driving_signal already calls calculate_surface_intensity_map with a
        # rest-frame wavelength (via frame_conversion) -- static_disk_magnitude
        # must follow the same convention, not observer-frame.
        calls = []

        def fake_calc(wavelength_nm):
            calls.append(wavelength_nm)
            return types.SimpleNamespace(flux_array=np.array([[1e-10]]))

        gen = self._build_gen(fake_calc)
        sed.static_disk_magnitude(gen, band=0)

        assert len(calls) == 1
        # band 0 observer wavelength is 400.0 nm, source_redshift=0.5 ->
        # rest-frame = 400.0 / 1.5
        assert calls[0] == pytest.approx(400.0 / 1.5)

    def test_returns_finite_scalar(self):
        gen = self._build_gen(lambda wl: types.SimpleNamespace(flux_array=np.array([[1e-10]])))
        mag = sed.static_disk_magnitude(gen, band=0)
        assert np.isscalar(mag) or (hasattr(mag, "shape") and mag.shape == ())
        assert np.isfinite(mag)

    def test_matches_manual_convert_flux_arr_to_mag_call(self):
        # static_disk_magnitude should just be convert_flux_arr_to_mag applied
        # to the disk's static flux map wrapped as a single-snapshot array --
        # not a separately-derived calculation.
        static_flux = np.array([[1e-10, 2e-10], [3e-10, 4e-10]])
        gen = self._build_gen(lambda wl: types.SimpleNamespace(flux_array=static_flux))

        result = sed.static_disk_magnitude(gen, band=1)
        expected = sed.convert_flux_arr_to_mag(gen, np.expand_dims(static_flux, axis=0), band=1)[0]

        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# io_hdf5.py
# ---------------------------------------------------------------------------

class FakeGeneratorForIO:
    """Minimal stand-in exposing exactly what save_data / _write_cadence_group
    read or call."""

    def __init__(self):
        self.band_names = ['u', 'g']
        self.num_images = 1
        self.deflector_redshift = 0.5
        self.kappa_star = [0.1]
        self.kappa = [0.4]
        self.shear = [0.1]
        self.source_redshift = 1.5
        self.time_delays = [0.0]
        self.true_magnitudes = [20.0, 20.5]
        self.bh_mass = 8.0
        self.inclination_angle = 0
        self.eddington_ratio = 0.15
        self.observer_frame_log_tau = 2.0
        self.observer_frame_log_sf = -1.0
        self.telescope_diameter_m = 1.0
        self.reference_diameter_m = 2.4
        self.reference_zeropoint = 27.6
        self.exposure_time = 1200
        self.psf_fwhm_arcsec = 0.15
        self.closest_image_separation = 0.5
        self.include_low_snr = False

        times = {'u': np.array([0.0, 10.0, 20.0]), 'g': np.array([0.0, 10.0, 20.0])}
        mags = [np.array([20.0, 20.1, 20.2]), np.array([20.5, 20.4, 20.3])]

        self.lensed_mags_container = {0: mags}
        self.microlensed_mags_container = {0: mags}
        self.observation_list = times
        self.rubin_lensed_mags_container = {}
        self.rubin_microlensed_mags_container = {}
        self.rubin_observation_list = {}

    def add_noise(self, magnitudes, magnitude_error_bars):
        # Deterministic for testing: no actual noise draw.
        return np.asarray(magnitudes, dtype=float)


class TestLsstAndBlendingErrors:
    """
    io_hdf5._compute_lsst_and_blending_errors combines photerr's photometric
    error with the deblending error in quadrature -- this is the "ground-based,
    so blending error is always added" behavior for the Rubin cadence.
    """

    def _make_gen(self, closest_image_separation=0.5):
        gen = types.SimpleNamespace(closest_image_separation=closest_image_separation)
        gen.add_noise = lambda mags, errs: np.asarray(mags, dtype=float)  # no-op for determinism
        return gen

    def test_final_error_is_quadrature_sum_of_phot_and_blending(self):
        gen = self._make_gen()

        def fake_err_model(df):
            out = df.copy()
            out['u_err'] = 0.02  # constant, known photometric error
            return out

        times = np.array([0.0, 1.0, 2.0])
        mags = np.array([20.0, 20.5, 21.0])
        mags_out, final_errs, phot_errs, blending_errs, times_out = io_hdf5._compute_lsst_and_blending_errors(
            gen, fake_err_model, 'u', times, mags
        )

        np.testing.assert_allclose(phot_errs, 0.02)
        expected_final = np.sqrt(phot_errs ** 2 + blending_errs ** 2)
        np.testing.assert_allclose(final_errs, expected_final)
        # Blending error must actually contribute -- final > phot alone.
        assert np.all(final_errs > phot_errs)

    def test_raises_without_closest_image_separation(self):
        # Ground-based -> blending error is mandatory; no silent skip.
        gen = self._make_gen(closest_image_separation=None)

        def fake_err_model(df):
            out = df.copy()
            out['u_err'] = 0.02
            return out

        with pytest.raises(ValueError):
            io_hdf5._compute_lsst_and_blending_errors(gen, fake_err_model, 'u', np.array([0.0]), np.array([20.0]))

    def test_non_finite_rows_are_dropped_and_times_stay_aligned(self):
        gen = self._make_gen()

        def fake_err_model(df):
            out = df.copy()
            # Simulate photerr failing to produce an error for one row
            # (e.g. below its detection limit) -- non-finite error/mag.
            out['u_err'] = [0.02, np.nan, 0.03]
            return out

        times = np.array([10.0, 20.0, 30.0])
        mags = np.array([20.0, 20.5, 21.0])
        mags_out, final_errs, phot_errs, blending_errs, times_out = io_hdf5._compute_lsst_and_blending_errors(
            gen, fake_err_model, 'u', times, mags
        )

        assert len(mags_out) == len(final_errs) == len(times_out) == 2
        np.testing.assert_array_equal(times_out, [10.0, 30.0])


class TestSaveData:
    def test_writes_expected_hdf5_structure(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        gen = FakeGeneratorForIO()
        out_path = tmp_path / "test_output.h5"
        io_hdf5.save_data(gen, str(out_path))

        with h5py.File(out_path, 'r') as f:
            assert 'metadata' in f
            assert 'user_cadence' in f
            assert 'rubin_cadence' not in f  # empty containers -> not written

            assert list(f['metadata']['band_names'][:]) == [b'u', b'g']

            lensed = f['user_cadence']['lensed_mags']['image_1']
            np.testing.assert_allclose(lensed['u'][:], gen.lensed_mags_container[0][0])
            np.testing.assert_allclose(lensed['g'][:], gen.lensed_mags_container[0][1])
            assert 'u_err' in lensed
            assert 'g_err' in lensed

            ml = f['user_cadence']['microlensed_and_lensed_mags']['image_mags_1']
            assert 'u_pre_noise' in ml
            assert 'g_pre_noise' in ml

    def test_rubin_cadence_group_written_when_present(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        gen = FakeGeneratorForIO()
        rubin_times = {'u': np.array([1.0, 2.0]), 'g': np.array([1.0, 2.0])}
        rubin_mags = [np.array([21.0, 21.1]), np.array([21.5, 21.4])]
        gen.rubin_lensed_mags_container = {0: rubin_mags}
        gen.rubin_microlensed_mags_container = {0: rubin_mags}
        gen.rubin_observation_list = rubin_times

        out_path = tmp_path / "test_output_rubin.h5"
        io_hdf5.save_data(gen, str(out_path))

        with h5py.File(out_path, 'r') as f:
            assert 'rubin_cadence' in f
            # Rubin cadence must use the LSST error model + deblending error
            # (combined in quadrature), not the generic analytical model --
            # so _phot_err and _blending_err must both be present, and _err
            # must be their quadrature sum, not just the photometric error.
            ml_grp = f['rubin_cadence']['microlensed_and_lensed_mags']['image_mags_1']
            assert 'u_phot_err' in ml_grp
            assert 'u_blending_err' in ml_grp
            phot = ml_grp['u_phot_err'][:]
            blend = ml_grp['u_blending_err'][:]
            final = ml_grp['u_err'][:]
            np.testing.assert_allclose(final, np.sqrt(phot ** 2 + blend ** 2))
