from dfm.src.megatron.model.wan.wan_layer_spec import get_wan_block_with_transformer_engine_spec


def test_get_wan_block_with_transformer_engine_spec_basic():
    spec = get_wan_block_with_transformer_engine_spec()
    # Basic structure checks
    assert hasattr(spec, "module")
    assert hasattr(spec, "submodules")
    sub = spec.submodules
    # Expected submodule fields exist
    for name in ["norm1", "norm2", "norm3", "full_self_attention", "cross_attention", "mlp"]:
        assert hasattr(sub, name), f"Missing submodule {name}"


