import pytest

from batch_size_studies.definitions import Parameterization, RunKey


def test_parameterization_enum_values():
    assert Parameterization.SP.value == "SP"
    assert Parameterization.MUP.value == "muP"


def test_parameterization_enum_members():
    # Check that valid members can be accessed
    assert Parameterization["SP"] == Parameterization.SP
    assert Parameterization("muP") == Parameterization.MUP

    # Check that accessing a non-existent member raises an error
    with pytest.raises(KeyError):
        _ = Parameterization["sp"]  # Should be case-sensitive

    with pytest.raises(ValueError):
        _ = Parameterization("invalid_scale")


def test_runkey_creation_and_validation():
    # This should work without any errors.
    try:
        key = RunKey(batch_size=32, eta=0.1)
        assert key.batch_size == 32
        assert key.eta == 0.1
    except TypeError:
        pytest.fail("RunKey raised an unexpected TypeError with correct input.")

    # This should fail because batch_size is a float, not an int.
    with pytest.raises(TypeError, match="RunKey 'batch_size' must be an integer"):
        RunKey(batch_size=32.0, eta=0.1)

    # This should fail because eta is an integer, not a float.
    with pytest.raises(TypeError, match="RunKey 'eta' must be a float"):
        RunKey(batch_size=32, eta=1)
