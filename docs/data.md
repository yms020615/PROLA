# Data

Paths in each YAML configuration are relative to `--data-root`. Arrays contain physical values; the model applies Aurora normalization. Latitude runs north to south and longitude increases. Date intervals are start-inclusive and end-exclusive; input and target frames stay within the same split.

## PSWD

Use ERA5/WeatherBench 2 data arranged as:

```text
data/pswd/
  5.625/<variable>/<variable>_<year>_5.625deg.nc
  1.5_1h_nc/<variable>/<variable>_<year>_1.5deg.nc
  static.nc
  climatology.npz
```

Inputs are sampled every 6h and targets every hour. The reader crops the target latitude dimension from 121 to 120 rows for patch size 4.

| Directory | NetCDF variable |
| --- | --- |
| `2m_temperature` | `t2m` |
| `10m_u_component_of_wind` | `u10` |
| `10m_v_component_of_wind` | `v10` |
| `temperature` | `t` |
| `u_component_of_wind` | `u` |
| `v_component_of_wind` | `v` |
| `specific_humidity` | `q` |
| `geopotential` | `z` |

Coordinates are `time`, `latitude`, `longitude`, and `level` for upper-air fields. Levels are 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, and 1000 hPa.

`static.nc` contains `lsm`, `z`, and `slt` with geographic coordinates. `climatology.npz` contains training-period temporal-mean maps under keys `2t`, `10u`, `10v`, `z500`, and `t850`, on the cropped target grid. It is needed only for evaluation.

Train on 2007–2016, validate on 2017, and test on 2018.

## ERA5-to-CERRA

```text
data/cerra/
  era5/<variable>/<variable>_<year>_5.625deg.nc
  cerra/cerra_central_europe_<YYYYMM>.h5
  static.nc
```

ERA5 files use the layout above. Each regional CERRA shard contains `time` in Unix seconds, two-dimensional `latitude` and `longitude`, and `fields` shaped `[time, 13, 128, 128]` at 3h intervals.

The channel order is T2M, 10m wind speed, 10m wind direction, followed by Z, T, U, V, and relative humidity, each at 500 and 850 hPa. Store this order as JSON in the HDF5 `channels` attribute using names `2t`, `10si`, `10wdir`, `z`, `t`, `u`, `v`, and `r`; surface levels are 0. The `variable_metadata` JSON attribute records surface direction `units="Degree true"` and upper-air `uvRelativeToGrid` flags by channel index.

The reader remaps ERA5 to the regional coordinates, converts humidity, and rotates grid-relative winds when required. Supply geopotential in m²/s², temperature in K, wind in m/s, and relative humidity in percent.

Train on all months of 2012–2016, validate on 2017, and test on 2018.

## Regional precipitation

```text
data/precipitation/
  frames/<YYYYMMDDHH>.npy
  frames/metadata.npz
  climatology/total_precipitation_6hr.nc
  climatology/total_precipitation_6hr_seeps_threshold.nc
  climatology/total_precipitation_6hr_seeps_dry_fraction.nc
```

Each 12-hourly frame has shape `[5, 14, 40, 40]`. Index 0 on the second axis stores the surface fields `(2t, 10u, 10v, msl, tp)`; indices 1–13 store `(t, u, v, r, z)` at the pressure levels listed above. Precipitation is the preceding 6h accumulation in meters. The region spans 0–58.5°N and 75–133.5°E at 1.5° resolution.

`metadata.npz` contains `lat`, `lon`, `static_lsm`, `static_z`, and `static_slt`. It can also include `surf_vars`, `upper_vars`, and `levels` for validation.

Each climatology file contains a variable with the same name as its filename stem and dimensions `dayofyear`, `hour`, `latitude`, and `longitude`. The mean and wet threshold are indexed by valid time; dry fraction is averaged across day and hour. Evaluation reports precipitation RMSE in cm.

Train on 2007–2016, validate on 2017, and test on 2018. Climatology files are needed only for evaluation.
