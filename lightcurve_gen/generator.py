"""
LightCurveGenerator: generates lensed-quasar light curves using a
snapshot method, supporting dual cadences (user + Rubin) sampled from the
same underlying signal.

Built on AMOEBA disk/driving-signal setup and the main
snapshot-generation loop. Everything else (optics/noise, SED handling,
cadence parsing, microlensing, HDF5 output) lives in other files
(``optics``, ``sed``, ``cadence``, ``microlensing``, ``io_hdf5``) and is
called directly from here (passing ``self`` where the function needs class info).

If you need one of those functions from outside this class (e.g. to save
a generator's output, or compute a photometric error), call the module
function directly with the generator instance, e.g.:

    from lightcurve_gen import io_hdf5
    io_hdf5.save_data(lc_generator, "output.h5")

Please let us know if you have feedback/spot any bugs!
Contact: pv10@illinois.edu
"""

import contextlib
import io

import numpy as np
import tqdm
from scipy.interpolate import interp1d

from amoeba.Classes.accretion_disk import AccretionDisk
from amoeba.Util.util import create_maps, generate_signal_from_bending_power_law

from . import optics
from . import sed
from . import cadence
from . import microlensing


class LightCurveGenerator:
    def __init__(self, band_names, num_images, deflector_redshift, kappa_star, kappa, shear, source_redshift,
                 time_delays, lensed_magnitudes, true_magnitudes, bh_mass, inclination_angle, eddington_ratio,
                 observer_frame_log_tau, observer_frame_log_sf, include_low_snr=False,
                 passband_config=None, cadence_file=None, baseline_days=None,
                 telescope_diameter_m=1, reference_diameter_m=2.4,
                 reference_zeropoint=27.6, diffraction_wavelength_nm=606, exposure_time=1200,
                 closest_image_separation=None):
        """
        Generates realistic lensed quasar light curves using a snapshot method.
        Supports dual cadences (user + Rubin) sampled from the same underlying signal.
        """
        self.num_images = num_images
        print('self.num_images: ', self.num_images)
        self.kappa = kappa
        self.kappa_star = kappa_star
        self.shear = shear
        self.source_redshift = source_redshift
        self.deflector_redshift = deflector_redshift
        self.time_delays = time_delays
        self.band_names = band_names
        self.lensed_magnitudes = lensed_magnitudes
        self.true_magnitudes = true_magnitudes
        self.bh_mass = bh_mass
        self.inclination_angle = inclination_angle
        self.eddington_ratio = eddington_ratio
        self.cadence_file = cadence_file
        self.baseline_days = baseline_days
        self.passband_config = passband_config or {}
        self.exposure_time = exposure_time
        self.telescope_diameter_m = telescope_diameter_m
        self.reference_diameter_m = reference_diameter_m
        self.reference_zeropoint = reference_zeropoint
        self.diffraction_wavelength_nm = diffraction_wavelength_nm
        self.psf_fwhm_arcsec = optics.diffraction_limited_fwhm_arcsec(
            self.diffraction_wavelength_nm, self.telescope_diameter_m
        )
        self.closest_image_separation = closest_image_separation
        self.filter_data = sed.load_in_transmissions(self)
        self.obs_wavelength_nm = np.array([np.median(self.filter_data[b][:, 0]) for b in self.band_names])
        print('self.obs_wavelength_nm: ', self.obs_wavelength_nm)
        self.observer_frame_log_tau = observer_frame_log_tau
        self.observer_frame_log_sf = observer_frame_log_sf
        self.colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown']
        self.accretion_disk = self.set_accretion_disk_object()

        # baseline_days drives the total simulation length and track length
        t_max = np.max([self.baseline_days // 365, 1])  # years

        assert len(self.band_names) == self.lensed_magnitudes.shape[0], \
            "Length of bands must match the number of rows in magnitudes."
        assert self.num_images == len(self.kappa) == len(self.kappa_star) == len(self.shear) == len(self.time_delays), \
            "Provide properties for all images"

        self.total_time = int(365 * (t_max + 1 / 3) + np.max(time_delays))
        self.include_low_snr = include_low_snr
        self.ipms = microlensing.get_mag_map_ipms(self, half_length=25, num_pixels=5000)
        sed.get_quasar_sed_at_source_redshift(self, quasar_file='data/optical_nir_qso_sed_001.fits')
        self.mag_map_objects = dict(zip(
            range(self.num_images),
            [microlensing.generate_amoeba_mag_map(self, i) for i in range(self.num_images)]
        ))

        # The absolute (unlensed, non-variable) magnitude a band's light curve
        # fluctuates around, derived directly from the disk's own static
        # emission at each band -- self-consistent across ANY band (reference
        # survey filter or a custom telescope's own passband), with no
        # per-band external catalog value or filter-map anchor needed. This
        # only depends on band (not on image or time), so it's computed once.
        self.static_disk_mags = np.array([
            sed.static_disk_magnitude(self, band) for band in range(len(self.band_names))
        ])
        print('self.static_disk_mags (physically self-consistent, replaces true_magnitudes): ',
              self.static_disk_mags)
        print('self.true_magnitudes (as supplied -- kept for reference/comparison only): ',
              np.asarray(self.true_magnitudes))

        # Separate output containers for each cadence
        self.lensed_mags_container = {}
        self.microlensed_mags_container = {}
        self.rubin_lensed_mags_container = {}
        self.rubin_microlensed_mags_container = {}

        # Separate observation lists for each cadence
        self.observation_list = {}
        self.rubin_observation_list = {}

    # ------------------------------------------------------------------
    # AMOEBA disk setup and driving signal 
    # ------------------------------------------------------------------

    def set_accretion_disk_object(self, number_grav_radii=500, resolution=100, spin=0, temp_beta=0, corona_height=10):
        if self.bh_mass > 1e5:
            smbh_mass_exp = np.log10(self.bh_mass)
        else:
            smbh_mass_exp = self.bh_mass
        agn_dictionary = create_maps(smbh_mass_exp=smbh_mass_exp,
                                     redshift_source=self.source_redshift,
                                     number_grav_radii=number_grav_radii,
                                     inclination_angle=self.inclination_angle,
                                     resolution=resolution,
                                     spin=spin,
                                     eddington_ratio=self.eddington_ratio,
                                     temp_beta=temp_beta,
                                     corona_height=corona_height)
        return AccretionDisk(**agn_dictionary)

    def get_driving_signal(self):
        in_nm_rest = sed.frame_conversion(self, self.obs_wavelength_nm, 'observer_to_rest')
        rest_frame_log_tau = sed.frame_conversion(self, self.observer_frame_log_tau, 'observer_to_rest')
        if isinstance(rest_frame_log_tau, (list, np.ndarray)):
            log_tau_u = rest_frame_log_tau[0]
            log_sf_u = self.observer_frame_log_sf[0]
        else:
            log_tau_u = rest_frame_log_tau
            log_sf_u = self.observer_frame_log_sf
        tau_u = 10 ** log_tau_u
        sf_u = 10 ** log_sf_u
        u_band_static_flux = self.accretion_disk.calculate_surface_intensity_map(in_nm_rest[0]).flux_array.mean()
        my_driving_signal_times, my_driving_signal = generate_signal_from_bending_power_law(
            length_of_light_curve=self.total_time,
            time_resolution=1,
            log_breakpoint_frequency=np.log(1 / (2 * np.pi * tau_u)),
            low_frequency_slope=0,
            high_frequency_slope=2,
            mean_magnitude=u_band_static_flux,
            standard_deviation=(sf_u / np.sqrt(2)) * u_band_static_flux,
            normal_magnitude_variance=True,
            zero_point_mag=0,
        )
        return my_driving_signal_times, my_driving_signal

    def add_noise(self, magnitudes, magnitude_error_bars):
        epsilon = np.random.normal(0, magnitude_error_bars)
        return magnitudes + epsilon

    # ------------------------------------------------------------------
    # Main snapshot-generation loop -- the class's core orchestration,
    # tying together all the modules above.
    # ------------------------------------------------------------------

    def generate_light_curves_from_snapshots(self, sampling_freq=3, specific_images=None, specific_bands=None,
                                              specific_mag_tracks=None, observation_mode='user_cadence',
                                              include_rubin=False, rubin_only=False):
        """
        Generate light curves sampled at both the user cadence and (optionally) the Rubin cadence.
        Both cadences share the same driving signal and microlensing track.

        Parameters
        ----------
        include_rubin : bool
            If True, also generate Rubin-cadence light curves by sampling the Rubin scheduler
            at a random sky position within the survey footprint.
        rubin_only : bool
            If True, skip the primary/user cadence entirely and only generate Rubin-cadence
            light curves. Implies ``include_rubin=True``.
        """
        include_rubin = include_rubin or rubin_only
        specific_images = np.arange(self.num_images) if specific_images is None else specific_images
        specific_bands = np.arange(len(self.band_names)) if specific_bands is None else specific_bands

        # --- User cadence (skipped entirely in rubin_only mode) ---
        if rubin_only:
            user_observation_times = {band: np.array([], dtype=float) for band in self.band_names}
        else:
            user_observation_times = cadence.get_observation_times(
                self, observation_mode, cadence_file=self.cadence_file, sampling_freq=sampling_freq
            )
        self.observation_list = user_observation_times

        # --- Rubin cadence: draw a random sky position and query the scheduler ---
        if include_rubin:
            ra_points, dec_points = cadence.get_random_rubin_ra_dec(N=1)
            rubin_observation_times = cadence.get_observation_times(
                self, 'rubin_cadence', ra_points=ra_points, dec_points=dec_points
            )
            self.rubin_observation_list = rubin_observation_times
            print(f"Rubin sky position: RA={ra_points[0].deg:.3f}, Dec={dec_points[0].deg:.3f}")
            for b, t in rubin_observation_times.items():
                print(f"  Rubin band {b}: {len(t)} observations over {self.baseline_days} days")
        else:
            rubin_observation_times = {}

        # --- Single driving signal shared across all cadences ---
        driving_signal_times, driving_signal = self.get_driving_signal()
        baseline_years = max(self.baseline_days / 365.0, 1.0)

        for image_ind in specific_images:
            mag_map = microlensing.generate_amoeba_mag_map(self, image_ind)
            mags_per_band, ml_mags_per_band = [], []
            rubin_mags_per_band, rubin_ml_mags_per_band = [], []

            image_track = None  # (x_pos, y_pos), set on the first band processed

            for band in tqdm.tqdm(specific_bands):
                band_name = self.band_names[band]

                # Union of all timestamps this band will need (user + rubin), so we
                # generate enough snapshots to cover both cadences in one pass.
                user_timestamps = user_observation_times[band_name]
                rubin_timestamps = rubin_observation_times[band_name] if include_rubin else np.array([])

                all_timestamps = np.union1d(user_timestamps, rubin_timestamps)
                # timestamps = self.observation_list[band_name]

                dt = float(self.time_delays[image_ind])
                driving_interp = interp1d(
                    driving_signal_times, driving_signal,
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                shifted_times = np.arange(0, driving_signal_times[-1] - dt, 1.0) + dt
                delayed_signal = driving_interp(shifted_times)
                if len(all_timestamps) == 0:
                    print(f"Warning: no observation times for band {band_name}, skipping.")
                    continue

                my_observed_wavelength = self.obs_wavelength_nm[band]

                print("THESE ARE THE APPLIED TIME DELAYS:")
                print(f'Image {image_ind} starts at {self.time_delays[image_ind]} days.')

                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    my_snapshots, _ = self.accretion_disk.generate_snapshots(
                        my_observed_wavelength,
                        all_timestamps / (1 + self.source_redshift),
                        delayed_signal,
                        driving_signal_fractional_strength=1,
                    )
                    snapshots_array = np.array([i.flux_array for i in my_snapshots])
                    convolved_map = mag_map.convolve_with_flux_projection(
                        my_snapshots[0],
                        relative_orientation=0
                    )

                warning_text = buf.getvalue()
                print(warning_text, end="")

                if (
                    "warning, driving signal is not long enough to support all snapshots. looping signal"
                    in warning_text.lower()
                    or "magnification map not large enough to perform convolution" in warning_text.lower()
                ):
                    raise SystemExit("Fatal warning encountered. Exiting.")

                # Extract the microlensing track once per image (on the first band
                # processed), then reuse the same (x_pos, y_pos) for every later band.
                if image_track is None:
                    image_track = microlensing._extract_microlensing_track(
                        self, convolved_map, specific_mag_tracks, image_ind
                    )
                x_pos, y_pos = image_track

                pixel_ratio = self.accretion_disk.pixel_size / mag_map.pixel_size

                # ---- Helper: get mags for a specific subset of timestamps ----
                def process_timestamps(ts):
                    # Find which rows of all_timestamps correspond to ts
                    idx = np.searchsorted(all_timestamps, ts)
                    snaps_subset = snapshots_array[idx]
                    mag_maps_subset, _ = microlensing._sample_magnifications_at_times(
                        x_pos, y_pos, ts, baseline_years, mag_map, pixel_ratio,
                        my_snapshots[0].flux_array
                    )
                    micro_snaps = microlensing._build_microlensed_snapshots(snaps_subset, mag_maps_subset, pixel_ratio)
                    true_m = sed.convert_flux_arr_to_mag(self, snaps_subset, band)
                    micro_plus_true_m = sed.convert_flux_arr_to_mag(self, micro_snaps, band)
                    micro_m = micro_plus_true_m - true_m
                    # Normalize: subtract the simulated mean, add the physically
                    # self-consistent static-disk magnitude for this band (not a
                    # user-supplied catalog value or filter-map anchor).
                    true_m -= np.mean(true_m)
                    true_m += self.static_disk_mags[band]
                    agn_m = true_m - 2.5 * np.log10(np.abs(mag_map.macro_magnification))
                    ml_agn_m = true_m + micro_m - 2.5 * np.log10(np.abs(mag_map.macro_magnification))
                    return agn_m, ml_agn_m

                # --- User cadence (skipped in rubin_only mode) ---
                if not rubin_only:
                    agn_disk_mags, ml_agn_disk_mags = process_timestamps(user_timestamps)
                    mags_per_band.append(agn_disk_mags)
                    ml_mags_per_band.append(ml_agn_disk_mags)

                # --- Rubin cadence ---
                if include_rubin and len(rubin_timestamps) > 0:
                    rubin_agn_mags, rubin_ml_mags = process_timestamps(rubin_timestamps)
                    rubin_mags_per_band.append(rubin_agn_mags)
                    rubin_ml_mags_per_band.append(rubin_ml_mags)

            if not rubin_only:
                self.lensed_mags_container[image_ind] = mags_per_band
                self.microlensed_mags_container[image_ind] = ml_mags_per_band
            if include_rubin:
                self.rubin_lensed_mags_container[image_ind] = rubin_mags_per_band
                self.rubin_microlensed_mags_container[image_ind] = rubin_ml_mags_per_band
