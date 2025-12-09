from batch_size_studies.definitions import LossType, OptimizerType, Parameterization, RunKey
from batch_size_studies.experiment_types.synthetic import SyntheticExperimentFixedData
from batch_size_studies.spectral.cache import SpectralCache


def _make_config():
    return SyntheticExperimentFixedData(
        D=4,
        P=8,
        N=4,
        K=2,
        num_epochs=1,
        gamma=1.0,
        L=2,
        parameterization=Parameterization.SP,
        optimizer=OptimizerType.SGD,
        loss_type=LossType.MSE,
    )


def test_spectral_cache_persists_entries(tmp_path):
    config = _make_config()
    cache = SpectralCache(config, directory=str(tmp_path), spectral_dir=str(tmp_path / "spectral"))

    run_key = RunKey(batch_size=16, eta=0.1)
    cache.store_step(run_key, 0, [1.0, 0.5], 1.5)
    cache.store_step(run_key, 5, [0.9], 1.1)

    cache2 = SpectralCache(config, directory=str(tmp_path), spectral_dir=str(tmp_path / "spectral"))
    data = cache2.get_run_dict(run_key)
    assert data[0]["eigenvalues"] == [1.0, 0.5]
    assert data[5]["trace"] == 1.1
