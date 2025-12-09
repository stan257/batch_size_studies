def test_custom_unpickler_remaps_legacy_classes():
    import io

    from batch_size_studies.storage_utils import _EXPERIMENT_CLASS_REMAP, _LEGACY_EXPERIMENT_MODULE, CustomUnpickler

    for legacy_class, new_module in _EXPERIMENT_CLASS_REMAP.items():
        unpickler = CustomUnpickler(io.BytesIO(b""))
        cls = unpickler.find_class(_LEGACY_EXPERIMENT_MODULE, legacy_class)
        assert cls.__module__.startswith(new_module)
