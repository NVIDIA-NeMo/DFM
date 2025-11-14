from dfm.src.megatron.model.wan.inference import SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES


def test_size_configs_structure_and_values():
    assert isinstance(SIZE_CONFIGS, dict)
    for key, val in SIZE_CONFIGS.items():
        assert isinstance(key, str)
        assert isinstance(val, tuple) and len(val) == 2
        w, h = val
        assert isinstance(w, int) and isinstance(h, int)
        assert w > 0 and h > 0


def test_max_area_configs_consistency():
    for size_key, area in MAX_AREA_CONFIGS.items():
        w, h = SIZE_CONFIGS[size_key]
        assert area == w * h


def test_supported_sizes_lists():
    assert "t2v-14B" in SUPPORTED_SIZES
    assert "t2v-1.3B" in SUPPORTED_SIZES
    for model_key, sizes in SUPPORTED_SIZES.items():
        assert isinstance(sizes, tuple)
        for s in sizes:
            assert s in SIZE_CONFIGS



