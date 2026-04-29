"""Tests for HuggingFace Transformers integration."""
import pytest
import torch


def test_asam_hf_config_creation():
    """ASAMHFConfig can be created with default values."""
    from asam.modeling_asam import ASAMHFConfig

    config = ASAMHFConfig(dim=256, num_heads=4, num_labels=2)
    assert config.model_type == "asam"
    assert config.dim == 256
    assert config.num_heads == 4


def test_asam_hf_config_defaults():
    """ASAMHFConfig has sensible defaults."""
    from asam.modeling_asam import ASAMHFConfig

    config = ASAMHFConfig()
    assert config.dim == 512
    assert config.pattern_type == "hierarchical"
    assert config.use_adaptive_gate is True


def test_asam_hf_config_serialization():
    """ASAMHFConfig can save/load JSON."""
    from asam.modeling_asam import ASAMHFConfig
    import tempfile, os

    config = ASAMHFConfig(dim=256, num_heads=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_pretrained(tmpdir)
        loaded = ASAMHFConfig.from_pretrained(tmpdir)
        assert loaded.dim == 256
        assert loaded.num_heads == 4


def test_asam_hf_config_to_asam_config():
    """to_asam_config() produces correct ASAMConfig."""
    from asam.modeling_asam import ASAMHFConfig
    from asam.asam_layer import ASAMConfig

    hf_config = ASAMHFConfig(dim=256, num_heads=4)
    asam_config = hf_config.to_asam_config()
    assert isinstance(asam_config, ASAMConfig)
    assert asam_config.dim == 256
    assert asam_config.num_heads == 4
    assert asam_config.pattern_type == "hierarchical"


def test_asam_hf_model_creation():
    """ASAMHFModel can be created and run forward."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFModel

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000)
    model = ASAMHFModel(config)
    input_ids = torch.randint(0, 1000, (2, 128))
    output = model(input_ids)
    assert output.last_hidden_state.shape == (2, 128, 64)


def test_asam_hf_model_save_load():
    """ASAMHFModel can save and load with HF methods."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFModel
    import tempfile, os

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000)
    model = ASAMHFModel(config)
    input_ids = torch.randint(0, 1000, (2, 128))

    with torch.no_grad():
        before = model(input_ids).last_hidden_state

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = ASAMHFModel.from_pretrained(tmpdir)

    with torch.no_grad():
        after = loaded(input_ids).last_hidden_state

    assert torch.allclose(before, after, atol=1e-6)


def test_asam_hf_classification():
    """ASAMHFForSequenceClassification produces valid logits."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000, num_labels=3)
    model = ASAMHFForSequenceClassification(config)
    input_ids = torch.randint(0, 1000, (2, 128))
    labels = torch.randint(0, 3, (2,))

    output = model(input_ids, labels=labels)
    assert output.logits.shape == (2, 3)
    assert output.loss is not None
    assert output.loss.item() > 0


def test_asam_hf_classification_save_load():
    """Classification model round-trips through save/load."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification
    import tempfile

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000, num_labels=2)
    model = ASAMHFForSequenceClassification(config)
    input_ids = torch.randint(0, 1000, (2, 128))

    with torch.no_grad():
        before = model(input_ids).logits

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = ASAMHFForSequenceClassification.from_pretrained(tmpdir)

    with torch.no_grad():
        after = loaded(input_ids).logits

    assert torch.allclose(before, after, atol=1e-6)
