"""Local binder optimization smoke test without pretrained weights."""

import pytest
import torch

from esm.models.esmfold2 import binder_design


@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_binder_logits_receive_gradients(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")
    gradients = []
    normalize = binder_design.normalized_gradient_tensor

    def record_gradient(gradient, mask):
        gradients.append(gradient.abs().sum().item())
        return normalize(gradient, mask)

    def fake_fold(model, target_seq, target_one_hot, design, **kwargs):
        assert design.device.type == device
        return {
            "distogram_logits": design,
            "seq_list": [target_seq + "|" + "M" * design.shape[1]],
            "iptm": [0.5] if kwargs["calculate_confidence"] else None,
        }

    monkeypatch.setattr(binder_design, "STEPS", 1)
    monkeypatch.setattr(binder_design, "fold_and_get_distogram", fake_fold)
    monkeypatch.setattr(
        binder_design,
        "compute_structure_losses",
        lambda design, *_args, **_kwargs: {"total_loss": design[..., 0].sum(dim=1)},
    )
    monkeypatch.setattr(
        binder_design,
        "compute_esmc_pseudoperplexity_nll",
        lambda **kwargs: (kwargs["binder_design"][..., 1].sum(dim=1)),
    )
    monkeypatch.setattr(binder_design, "normalized_gradient_tensor", record_gradient)

    sequences, trajectory, _ = binder_design.design_binder(
        inversion_models={"stub": object()},
        hf_critic_models={},
        esmc_model=object(),
        scaling_critic_names=None,
        target_name="custom",
        target_sequence="M",
        binder_name="custom",
        binder_sequence="#" * 60,
        is_antibody=False,
        seed=5,
        device=device,
    )

    assert sequences == ["M|" + "M" * 60]
    assert len(trajectory) == 1
    assert len(gradients) == 2 and all(gradient > 0 for gradient in gradients)


def test_local_models_share_frozen_esmc(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    loaded = []

    class FakeFold(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.esmc = None
            self.config = type("Config", (), {"esmc_id": "biohub/ESMC-6B"})()

        def load_esmc(self, path):
            loaded.append(("backbone", path))
            self.esmc = torch.nn.Linear(2, 2)

        def configure_lm_dropout(self, *_args, **_kwargs):
            pass

        def set_kernel_backend(self, _backend):
            pass

    class FakeLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.esmc = torch.nn.Linear(2, 2)

    def load_fold(path, *, load_esmc, local_files_only):
        assert not load_esmc and local_files_only
        loaded.append(("fold", path))
        return FakeFold()

    def load_lm(path, *, device, dtype, local_files_only):
        assert device == "cpu" and local_files_only
        loaded.append(("lm", path))
        return FakeLM()

    monkeypatch.setattr(
        binder_design.EsmFold2ExperimentalModel, "from_pretrained", load_fold
    )
    monkeypatch.setattr(binder_design.EsmcForMaskedLM, "from_pretrained", load_lm)
    binder_design._ESMC_CACHE.clear()
    engine = binder_design.ESMFold2Design()
    fast = tmp_path / "fast"
    critic = tmp_path / "critic"
    esmc = tmp_path / "esmc"
    for directory in (fast, critic, esmc):
        directory.mkdir()

    engine.load(fast, critic, esmc, device="cpu")

    assert loaded == [
        ("fold", str(fast)),
        ("backbone", str(esmc)),
        ("fold", str(critic)),
        ("lm", str(esmc)),
    ]
    assert (
        engine.esmc_model.esmc
        is engine.inversion_models["ESMFold2-Experimental-Fast"].esmc
    )
    assert not any(
        parameter.requires_grad for parameter in engine.esmc_model.parameters()
    )
    binder_design._ESMC_CACHE.clear()
