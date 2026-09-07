"""
Pytest configuration.

``lightcurve_gen`` imports several domain-specific research packages at
module load time: ``amoeba``, ``microlensing`` (the IPM package),
``rubin_sim_pipeline``, and ``photerr``. These aren't on PyPI and won't be
installed in most CI environments, but most of the package's logic
(cadence generation, error models, baseline cuts, HDF5 output) doesn't
actually need the real physics from those packages to be exercised.

This file installs minimal stand-in modules for the ones that are only
*imported*, not used, by the functions these tests exercise, so the test
suite can run without the real research environment. Anything that needs
the actual AMOEBA disk/microlensing physics (e.g. ``convert_flux_arr_to_mag``
end-to-end, ``generate_light_curves_from_snapshots``) is exercised with a
synthetic stand-in accretion disk in the relevant test, or is out of scope
here and left to integration testing in the real environment.
"""

import sys
import types


def _stub_module(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_stubs():
    _stub_module('amoeba')
    _stub_module('amoeba.Classes')
    _stub_module('amoeba.Classes.accretion_disk', AccretionDisk=object)
    _stub_module('amoeba.Classes.magnification_map', MagnificationMap=object)
    _stub_module('amoeba.Util')
    _stub_module(
        'amoeba.Util.util',
        create_maps=lambda **kwargs: {},
        generate_signal_from_bending_power_law=lambda **kwargs: (None, None),
        extract_path_on_microlensing_map=lambda *args, **kwargs: None,
        pull_subarray_from_grid=lambda *args, **kwargs: None,
    )
    _stub_module('microlensing')
    _stub_module('microlensing.IPM')
    _stub_module('microlensing.IPM.ipm', IPM=object)
    _stub_module('rubin_sim_pipeline', get_rubin_cadence=lambda *args, **kwargs: None)

    class _FakeLsstErrorModel:
        """
        Stands in for photerr.LsstErrorModel: callable on a DataFrame with a
        'times' column plus one magnitude column, returns that DataFrame with
        a '{band}_err' column added. Real enough for tests exercising the
        surrounding pipeline (row filtering, combining with deblending error,
        dataset naming) without needing the real photometric error physics.
        """

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, df):
            band = [c for c in df.columns if c != 'times'][0]
            out = df.copy()
            out[f'{band}_err'] = 0.01
            return out

    _stub_module('photerr', LsstErrorModel=_FakeLsstErrorModel)


_install_stubs()
