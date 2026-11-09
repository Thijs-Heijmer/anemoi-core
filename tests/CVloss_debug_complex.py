from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr
from omegaconf import OmegaConf

from anemoi.training.losses import CRPS, SpectralCRPSLoss, SpectralCVLoss
from anemoi.training.utils.enums import TensorDim


root = Path(__file__).resolve().parents[2]
config_path = root / "bris-configs/configs/alpaca/anemoi-training/coherent-lama_w0.02_standalone.yaml"
forecast_path = root / "bris-inference/outputs/malu/calm-coriolis_nordic_20251110T120000Z_to_20251114T180000Z_m26.nc"
target_path = Path("/lustre/storeB/project/nwp/bris/datasets/meps-2p5km-202501-202604-6h-v7.zarr")
variables = ["2t", "t_850", "q_850", "z_500"]
n_samples = 4
device = "cuda" if torch.cuda.is_available() else "cpu"

# Explicit mappings for this small diagnostic subset; pressure coordinates are in hPa.
forecast_fields = {
    "2t": ("air_temperature_2m", None),
    "t_850": ("air_temperature_pl", 850),
    "q_850": ("specific_humidity_pl", 850),
    "z_500": ("geopotential_pl", 500),
}
config = OmegaConf.load(config_path)
loss_config = config.training.training_loss.datasets.data
scalers = config.training.scalers.datasets.data
normalizer = config.data.datasets.data.processors.normalizer.config
training = zarr.open_group(config.system.input.dataset, mode="r")
targets = zarr.open_group(str(target_path), mode="r")
target_dates = targets["dates"][:].astype("datetime64[ns]")
trim = config.dataloader.dataset.cutout[0].trim_edge
grid_slice = slice(trim, -trim) if trim else slice(None)
field_shape = tuple(targets.attrs["field_shape"])
ny, nx = loss_config.losses[1].y_dim, loss_config.losses[1].x_dim

losses = []
for loss_class, settings in zip((CRPS, SpectralCRPSLoss, SpectralCVLoss), loss_config.losses):
    kwargs = OmegaConf.to_container(settings, resolve=True)
    kwargs.pop("_target_")
    kwargs.pop("scalers")
    losses.append(loss_class(**kwargs))

# BaseLoss sums over grid/modes/bands; the grid scalers set the desired normalization.
losses[0].add_scaler(TensorDim.GRID, torch.full((ny * nx,), 1 / (ny * nx)), name="node_weights")
for loss, settings, name in zip(
    losses[1:],
    (scalers.spectral_weights, scalers.cv_radial_weights),
    ("spectral_weights", "cv_radial_weights"),
):
    size = settings.get("spectral_dims", settings.n_spectral_modes)
    loss.add_scaler(TensorDim.GRID, torch.full((size,), 1 / settings.n_spectral_modes), name=name)
for loss in losses:
    loss.add_scaler(TensorDim.VARIABLE, torch.ones(1), name="variable_weights")
    loss.to(device)

combined_weights = np.asarray(loss_config.loss_weights, dtype=float)
scores = []
with xr.open_dataset(forecast_path) as forecast, torch.no_grad():
    for coord, key in (("latitude", "latitudes"), ("longitude", "longitudes")):
        target_grid = targets[key][:].reshape(field_shape)[grid_slice, grid_slice]
        if target_grid.shape != (ny, nx):
            raise ValueError(f"Unexpected trimmed grid shape: {target_grid.shape}")
        np.testing.assert_allclose(target_grid, forecast[coord].values, rtol=0, atol=1e-6)

    times = forecast.time.values[forecast.time.values > forecast.forecast_reference_time.values][:n_samples]
    if n_samples < 1 or len(times) != n_samples:
        raise ValueError(f"Requested {n_samples} forecast times, found {len(times)}")
    print("Regional, selected-variable estimate of training loss balance", flush=True)
    print(f"Times: {times}; members: {forecast.sizes['ensemble_member']}; device: {device}", flush=True)
    print(f"Variables: {variables}\nTargets: {target_path}", flush=True)
    print(f"Normalization: config rules, regional training statistics from {config.system.input.dataset}", flush=True)
    print("CRPS uses uniform regional weights summing to one; global grid contribution is omitted.", flush=True)
    print("Targets are outside the training period. Variable/pressure weights are included in all reported losses.", flush=True)

    for time in times:
        matches = np.flatnonzero(target_dates == time)
        if len(matches) != 1:
            raise ValueError(f"Expected one target at {time}, found {len(matches)}")
        # The Zarr chunks contain all variables for one time; read each chunk only once.
        target_fields = targets["data"][int(matches[0]), :, 0, :]
        for variable in variables:
            field, pressure = forecast_fields[variable]
            prediction_field = forecast[field].sel(time=time)
            if pressure is not None:
                prediction_field = prediction_field.sel(pressure=pressure)
            else:
                prediction_field = prediction_field.squeeze("height1", drop=True)
            prediction_values = prediction_field.transpose("ensemble_member", "y", "x").values
            target_index = targets.attrs["variables"].index(variable)
            target_values = target_fields[target_index].reshape(field_shape)[grid_slice, grid_slice]

            # Shape: batch, time, ensemble, flattened grid, variable.
            prediction = torch.as_tensor(prediction_values.copy(), dtype=torch.float32, device=device).reshape(1, 1, -1, ny * nx, 1)
            target = torch.as_tensor(target_values.copy(), dtype=torch.float32, device=device).reshape(1, 1, 1, ny * nx, 1)
            stats_index = training.attrs["variables"].index(variable)
            method = normalizer.default
            for candidate in ("mean-std", "std", "none", "max", "min-max"):
                if variable in (normalizer.get(candidate) or []):
                    method = candidate
            if method == "mean-std":
                offset, scale = training["mean"][stats_index], training["stdev"][stats_index]
            elif method == "std":
                offset, scale = 0, training["stdev"][stats_index]
            elif method == "none":
                offset, scale = 0, 1
            else:
                raise ValueError(f"Unsupported normalization for this diagnostic: {variable}: {method}")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid normalization scale for {variable}: {scale}")
            prediction = (prediction - float(offset)) / float(scale)
            target = (target - float(offset)) / float(scale)
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError(f"Nonfinite input for {variable} at {time}")

            base_variable = variable.split("_")[0]
            variable_weight = scalers.general_variable.weights.get(base_variable, scalers.general_variable.weights.default)
            if pressure is not None:
                variable_weight *= max(scalers.pressure_level.y_intercept, pressure * scalers.pressure_level.slope)
            values = []
            for loss in losses:
                loss.update_scaler("variable_weights", torch.tensor([variable_weight], device=device))
                value = loss(prediction, target).item()
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite {loss.name} for {variable} at {time}")
                values.append(value)
            scores.append(values)
        print(f"Finished {time}", flush=True)

means = np.asarray(scores).reshape(len(times), len(variables), 3).mean(axis=0)
overall = means.mean(axis=0)
print("\nMean losses before / after combined-loss weights:")
print(f"{'Variable':<12} {'CRPS':>25} {'Spectral CRPS':>25} {'Spectral CV':>25}")
for label, values in zip([*variables, "Overall"], [*means, overall]):
    print(f"{label:<12}" + "".join(f"{raw:12.6g} / {weighted:<10.6g}" for raw, weighted in zip(values, values * combined_weights)))
print(f"Combined-loss weights: {combined_weights}")
if overall[2] == 0:
    print("CV multiplier matching weighted spectral CRPS: undefined (mean CV is zero)")
else:
    print(f"CV multiplier matching weighted spectral CRPS: {combined_weights[1] * overall[1] / overall[2]:.6g}")
