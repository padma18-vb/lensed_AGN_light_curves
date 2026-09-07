#!/usr/bin/env bash
#
# run_light_curve_gen.sh
#
# Thin wrapper around `python -m lightcurve_gen.cli`. Edit the defaults
# below (or override any of them with an environment variable, e.g.
#   LENS_IDX=17 ./run_light_curve_gen.sh
# ) then run from the repo root:
#   ./run_light_curve_gen.sh
#
# Any extra arguments you pass on the command line are forwarded as-is to
# lightcurve_gen.cli, e.g.:
#   ./run_light_curve_gen.sh --include_rubin --include_noisy_data

set -euo pipefail

# cd to the repo root (the directory this script lives in) so `python -m
# lightcurve_gen.cli` can find the package regardless of where you call
# this script from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export RUBIN_SIM_DATA_DIR=/pscratch/sd/v/vpadma/rubin_sim_data
echo "RUBIN_SIM_DATA_DIR=${RUBIN_SIM_DATA_DIR}"


# ---- Defaults -- override with environment variables as needed ----------
CSV_FILE="${CSV_FILE:-data/joined_full_sample_all_bands_detected.csv}"
LENS_IDX="${LENS_IDX:-2}"
DATASET_NAME="${DATASET_NAME:-example_run}"
OBSERVATION_MODE="${OBSERVATION_MODE:-regular_cadence}"
PASSBAND_CONFIG="${PASSBAND_CONFIG:-configs/passband_config_3_bands.json}"
BASELINE_DAYS="${BASELINE_DAYS:-730}"
TELESCOPE_DIAMETER_M="${TELESCOPE_DIAMETER_M:-1.0}"
REFERENCE_DIAMETER_M="${REFERENCE_DIAMETER_M:-2.4}"
DIFFRACTION_WAVELENGTH_NM="${DIFFRACTION_WAVELENGTH_NM:-606}"
EXPOSURE_TIME="${EXPOSURE_TIME:-1200}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

echo "Running lightcurve_gen with:"
echo "  csv_file:      $CSV_FILE"
echo "  lens_idx:      $LENS_IDX"
echo "  dataset_name:  $DATASET_NAME"
echo "  obs_mode:      $OBSERVATION_MODE"
echo "  passband_cfg:  $PASSBAND_CONFIG"
echo "  output_dir:    $OUTPUT_DIR"
echo ""

python -m lightcurve_gen.cli \
    --csv_file "$CSV_FILE" \
    --lens_idx "$LENS_IDX" \
    --dataset_name "$DATASET_NAME" \
    --observation_mode "$OBSERVATION_MODE" \
    --passband_config "$PASSBAND_CONFIG" \
    --baseline_days "$BASELINE_DAYS" \
    --telescope_diameter_m "$TELESCOPE_DIAMETER_M" \
    --reference_diameter_m "$REFERENCE_DIAMETER_M" \
    --diffraction_wavelength_nm "$DIFFRACTION_WAVELENGTH_NM" \
    --exposure_time "$EXPOSURE_TIME" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
