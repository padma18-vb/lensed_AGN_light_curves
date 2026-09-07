# lensed_agn_light_curves

Generates lensed-AGN light curves for an arbitrary telescope
(passband set, aperture, exposure time), with an optional simultaneous
Rubin/LSST cadence. Uses a variable flux map x microlensing map method built on
[AMOEBA](https://github.com/Henry-Best-01/Amoeba) (accretion-disk simulation) and [microlensing](https://github.com/weisluke/microlensing) magnification maps, [photErr](https://github.com/jfcrenshaw/photerr) for Rubin photometric error, and [rubin-sim, installation required](https://rubin-sim.lsst.io/installation.html) for Rubin cadence.
## Repository structure

```
lensed_agn_light_curves/
├── lightcurve_gen/           # the package
│   ├── optics.py              # observation model
│   ├── sed.py                 # passbands, quasar SED, flux -> AB mag
│   ├── cadence.py             # processes observation times based on what telescope/mode you ask for
│   ├── microlensing.py        # loads in magnification maps, track extraction/sampling
│   ├── io_hdf5.py              # HDF5 output
│   ├── generator.py           # main part that assembles code
│   └── cli.py                  # command-line entry point
├── configs/
│   └── passband_config_3_bands.json   # example passband config
├── data/                       # your lens-catalog CSV goes here
├── notebooks/
│   └── visualize_light_curve.ipynb    # plot an already-generated .h5 file
├── tests/
│   ├── conftest.py             # claude-generated tests
│   └── test_lightcurve_gen.py
├── run_light_curve_gen.sh      # wrapper to launch in a GPU shell
├── requirements.txt
├── requirements-dev.txt
└── LICENSE
```

## Installation
Create and activate a new environment using either conda or Python's built-in
`venv` module.

### Option 1: conda

```bash
conda create -n lensed-agn-light-curves python=3.11
conda activate lensed-agn-light-curves
pip install -r requirements.txt
```

### Option 2: Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NOTE: I have hashed out the rubin sim installation in the requirements.txt. I would suggest you directly follow the instructions for rubin-sim installation [here](https://rubin-sim.lsst.io/installation.html).
Repoduced here:
```
pip install rubin-sim
scheduler_download_data
rs_download_data
```
ALSO: in `run_light_curve_gen.sh` make sure to point where the rubin sim data is stored.

## making light curves
Currently, the pipeline does need a `csv` file that has a bunch of physics corresponding to the the microlensing, the accretion disk physics (e.g black hole mass). This output is easily generated through [SLSim](https://github.com/LSST-strong-lensing/slsim). We have already included a catalog of lenses that we expect to find with Rubin (sampled over a 21000sq deg region, so slighly oversampled). 

Some values are hardcoded currently, like the inclination angle of the disk, the corona height the accretion disk model used etc. This can be easily modified by a user, and in a future update, we can make the light curve generator a lot more flexible as needed. We can also just include these values in the catalog and read them in as we do for other parameters, once we decide what distribution we are calling them from.

Using `cli`:

```bash
python -m lightcurve_gen.cli \
    --csv_file data/lens_catalog.csv \
    --lens_idx 0 \
    --dataset_name example_run \
    --observation_mode regular_cadence \
    --passband_config configs/passband_config_3_bands.json \
    --baseline_days 730 \
    --output_dir outputs
```

or use the `bash` wrapper (make sure to open the wrapper and change the telescope configs etc as you want them! Currently some defaults are set just so users can easily run the code )

```bash
./run_light_curve_gen.sh
# or, e.g.:
LENS_IDX=5 DATASET_NAME=lens5_run ./run_light_curve_gen.sh --include_rubin
```

Output is written to `<output_dir>/custom_telescope/<MMDDYYYY>/<dataset_name><lens_idx>.h5`.

To generate light curves for a whole sample, ~ 1000s of light curves, I would just launch a 10 hour GPU job, and run 1000 objects (if only using one core).
## Visualizing output

Open `notebooks/visualize_light_curve.ipynb`, point `H5_PATH` at a
generated `.h5` file, and run all cells. It plots lensed-only and
microlensed+lensed magnitudes (with error bars) per image and band, and
will also plot the Rubin cadence if the file has one (i.e. was generated
with `--include_rubin`). Note that you can pass in a user cadence + telescope and ask for --include_rubin, in case you want to compare.

