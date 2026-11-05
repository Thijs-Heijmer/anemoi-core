# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.training.losses.scalers import NaNMaskScaler
from anemoi.training.losses.scalers import SpectralDimensionScaler
from anemoi.training.losses.spectral import SpectralCVLoss
from anemoi.training.utils.enums import TensorDim


def _loss(bin_weights: list[float] | None = None) -> SpectralCVLoss:
    return SpectralCVLoss(
        x_dim=20,
        y_dim=16,
        min_freq=0.08,
        max_freq=0.5,
        radial_edge_count=6,
        angle_edge_count=30,
        bin_weights=bin_weights,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    pred = torch.randn(2, 2, 2, 320, 3, generator=generator)
    target = torch.randn(2, 2, 1, 320, 3, generator=generator)
    return pred, target


def _add_radial_scaler(loss: SpectralCVLoss) -> None:
    scaler = SpectralDimensionScaler(n_spectral_modes=1, spectral_dims=loss.n_radial)
    loss.add_scaler(*scaler.get_scaling(), name="cv_radial_weights")


def _add_scalers(loss: SpectralCVLoss) -> None:
    _add_radial_scaler(loss)
    loss.add_scaler(TensorDim.VARIABLE, torch.tensor([0.2, 1.0, 1.2]), name="pressure_level")
    loss.add_scaler(TensorDim.VARIABLE, torch.tensor([6.0, 0.5, 2.0]), name="general_variable")
    loss.add_scaler(TensorDim.TIME, torch.tensor([0.25, 0.75]), name="time_steps")
    loss.add_scaler(*NaNMaskScaler().get_scaling(), name="nan_mask_weights")


@pytest.mark.parametrize("bin_weights", [None, [0.0, 0.5, 2.0, 1.0, 3.0]])
def test_unit_radial_scaler_preserves_sum(bin_weights: list[float] | None) -> None:
    pred, target = _inputs()
    loss = _loss(bin_weights)
    unscaled = loss(pred, target)
    _add_radial_scaler(loss)
    loss.add_scaler(*NaNMaskScaler().get_scaling(), name="nan_mask_weights")
    torch.testing.assert_close(loss(pred, target), unscaled)


def test_bin_weights_applied_once() -> None:
    pred, target = _inputs()
    weights = torch.tensor([0.0, 0.5, 2.0, 1.0, 3.0])
    # Isolate each band, then form an independent weighted sum of its loss.
    band_losses = torch.stack([_loss(row.tolist())(pred, target) for row in torch.eye(5)])
    loss = _loss(weights.tolist())
    _add_radial_scaler(loss)
    torch.testing.assert_close(loss(pred, target), (band_losses * weights).sum())


@pytest.mark.parametrize("squash_mode", ["avg", "sum"])
def test_variable_time_scaling_and_selection(squash_mode: str) -> None:
    pred, target = _inputs()
    loss = _loss([0.0, 0.5, 2.0, 1.0, 3.0])
    # Unscaled per-time/per-variable errors supply an independent reduction reference.
    per_time = torch.stack([loss(pred[:, t : t + 1], target[:, t : t + 1], squash=False) for t in range(2)])
    expected_variables = (per_time * torch.tensor([0.25, 0.75])[:, None]).sum(0)
    expected_variables *= torch.tensor([0.2, 1.0, 1.2]) * torch.tensor([6.0, 0.5, 2.0])
    _add_scalers(loss)

    torch.testing.assert_close(loss(pred, target, squash=False), expected_variables)
    expected = expected_variables.mean() if squash_mode == "avg" else expected_variables.sum()
    torch.testing.assert_close(loss(pred, target, squash_mode=squash_mode), expected)

    indices = torch.tensor([2, 0])
    selected = loss(pred, target, squash=False, scaler_indices=(..., indices))
    torch.testing.assert_close(selected, expected_variables[indices])


@pytest.mark.parametrize("without_scalers", [["general_variable"], [TensorDim.VARIABLE]])
def test_scaler_exclusion(without_scalers: list) -> None:
    pred, target = _inputs()
    loss = _loss()
    per_time = torch.stack([loss(pred[:, t : t + 1], target[:, t : t + 1], squash=False) for t in range(2)])
    expected = (per_time * torch.tensor([0.25, 0.75])[:, None]).sum(0)
    if without_scalers == ["general_variable"]:
        expected *= torch.tensor([0.2, 1.0, 1.2])
    _add_scalers(loss)
    torch.testing.assert_close(loss(pred, target, without_scalers=without_scalers), expected.mean())


@pytest.mark.parametrize("grid_size", [5, 320])
def test_spatial_nan_mask_rejected(grid_size: int) -> None:
    pred, target = _inputs()
    loss = _loss()
    # Also reject a spatial mask that accidentally matches the radial-band count.
    loss.add_scaler(
        (TensorDim.BATCH_SIZE, TensorDim.GRID, TensorDim.VARIABLE),
        torch.ones(2, grid_size, 3),
        name="nan_mask_weights",
    )
    with pytest.raises(ValueError, match="SpectralCVLoss does not support spatial nan_mask_weights"):
        loss(pred, target)


def test_incorrect_radial_scaler_size_rejected() -> None:
    pred, target = _inputs()
    loss = _loss()
    loss.add_scaler(TensorDim.GRID, torch.ones(6), name="cv_radial_weights")
    with pytest.raises(AssertionError, match=r"cv_radial_weights.*size 6.*size 5"):
        loss(pred, target)


@pytest.mark.parametrize("input_name", ["prediction", "target"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_fft_input_rejected(input_name: str, value: float) -> None:
    pred, target = _inputs()
    tensor = pred if input_name == "prediction" else target
    tensor[0, 0, 0, 0, 0] = value
    loss = _loss()
    _add_scalers(loss)
    with pytest.raises(ValueError, match=f"Selected {input_name} contains NaN or Inf"):
        loss(pred, target)


def test_scaled_gradients_are_finite() -> None:
    pred, target = _inputs()
    pred.requires_grad_()
    loss = _loss([0.0, 0.5, 2.0, 1.0, 3.0])
    _add_scalers(loss)
    result = loss(pred, target)
    result.backward()
    assert torch.isfinite(result)
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def _distributed_cv_worker(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        pred, target = _inputs()
        pred.requires_grad_()
        loss = _loss([0.0, 0.5, 2.0, 1.0, 3.0])
        _add_scalers(loss)
        # Nonuniform radial scalers expose incorrect slicing on the second rank.
        loss.update_scaler("cv_radial_weights", torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5]))
        expected = loss(pred, target)
        expected.backward()

        # Uneven spatial (137/183), variable (2/1), and radial-band (3/2) partitions.
        grid_sizes = [137, 183]
        start = sum(grid_sizes[:rank])
        shard_slice = slice(start, start + grid_sizes[rank])
        pred_shard = pred.detach()[..., shard_slice, :].clone().requires_grad_()
        target_shard = target[..., shard_slice, :].clone()
        result = loss(
            pred_shard,
            target_shard,
            grid_shard_slice=shard_slice,
            grid_shard_sizes=grid_sizes,
            group=dist.group.WORLD,
        )
        result.backward()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(pred_shard.grad, pred.grad[..., shard_slice, :], atol=2e-6, rtol=2e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
def test_sharded_cv_matches_unsharded(tmp_path: Path) -> None:
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if version < (2, 6):
        pytest.skip("Anemoi Gloo all-to-all requires PyTorch >= 2.6")
    mp.spawn(_distributed_cv_worker, args=(str(tmp_path / "cv_dist_init"),), nprocs=2, join=True)
