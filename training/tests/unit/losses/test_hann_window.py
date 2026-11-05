# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import einops
import pytest
import torch

from anemoi.training.utils.enums import TensorDim


@pytest.mark.parametrize("ensemble_size", [1, 2])
def test_flattened_hann_window_matches_2d(ensemble_size: int) -> None:
    """Flattening preserves per-field centering and spatial Hann windowing."""
    y_dim, x_dim = 5, 7
    generator = torch.Generator(device="cpu").manual_seed(0)
    values = torch.randn(
        (2, 2, ensemble_size, y_dim, x_dim, 3),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    original = values.clone()
    window_2d = (
        torch.hann_window(y_dim, dtype=values.dtype, device=values.device)[:, None]
        * torch.hann_window(x_dim, dtype=values.dtype, device=values.device)[None, :]
    )

    expected = (values - values.mean(dim=(-3, -2), keepdim=True)) * window_2d[..., None]

    flattened_values = einops.rearrange(values, "b t e y x v -> b t e (y x) v")
    flattened_window = einops.rearrange(window_2d, "y x -> (y x)", y=y_dim, x=x_dim)
    actual_flat = (
        flattened_values - flattened_values.mean(dim=TensorDim.GRID, keepdim=True)
    ) * flattened_window[:, None]

    assert actual_flat.shape == (2, 2, ensemble_size, y_dim * x_dim, 3)
    actual = actual_flat.reshape(values.shape)
    assert actual.shape == expected.shape == values.shape
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
