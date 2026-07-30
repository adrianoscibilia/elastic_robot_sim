# KUKA KR300 R2500 ultra SE identification benchmark

## Upstream

- Landing page: <https://doi.org/10.26204/DATA/5>
- Direct file base: <https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/>
- Benchmark page: <https://www.nonlinearbenchmark.org/benchmarks/industrial-robot>
- Licence stated by the DOI landing page: **MIT License, CC BY-SA 4.0**.
- Citation: J. Weigand, J. Götz, J. Ulmen, and M. Ruskowski, *Dataset and
  Baseline for an Industrial Robot Identification Benchmark* (2022).

## Local acquisition (2026-07-30)

The following upstream archives were downloaded into `raw/` (ignored by Git):

| File | Bytes | MD5 |
| --- | ---: | --- |
| `Robot_Identification_Benchmark_Without_Raw_Data.rar` | 12,717,003 | `6fffa993f60d65f85100bad053888df2` |
| `Robot_Identification_Benchmark_With_Raw_Data.rar` | 227,384,738 | `4a9d3adbbca1c9b68577d6db63d58c58` |

`Robot_Identification_Benchmark_Description.pdf` was also downloaded beside
this document.  The full archive's `raw_data/` directory has been extracted
locally and contains all 12 source recordings.  Re-fetch, if necessary:

```powershell
curl.exe -fL --retry 3 --output raw/Robot_Identification_Benchmark_With_Raw_Data.rar \
  https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_With_Raw_Data.rar
tar.exe -xf raw/Robot_Identification_Benchmark_With_Raw_Data.rar -C raw raw_data
```

## Verified MATLAB schemas

All MATLAB matrices have channel-major shape `(6, N)` and must be transposed
by a row-oriented dataset loader.

### Raw recordings — calibration source of truth

Each `raw/raw_data/recording_*.mat` has `N = 90,881`, `dt = 0.004 s`, angles
in degrees, angular velocities in degrees/s, and torques in Nm:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `q_mot_meas` | `(6, 90881)` | motor-side real-axis positions |
| `q_se_meas` | `(6, 90881)` | link/output-side secondary-encoder positions |
| `qd_mot_meas` | `(6, 90881)` | motor-side velocities |
| `qd_se_meas` | `(6, 90881)` | link/output-side velocities |
| `tau_meas` | `(6, 90881)` | real motor torque |
| `tau_fb_meas` | `(6, 90881)` | position-control feedback torque |
| `q_ref`, `qd_ref`, `tau_ref_ff` | `(6, 90881)` each | executed references/feed-forward torque |
| `time` | `(1, 90881)` | seconds |

This is the preferred source for elastic-transmission identification: replay
`tau_meas` at the simulator motor joint and fit motor and secondary/link-side
states, after converting all angles and rates to SI units.

### Prepared forward-identification file

`forward_identification_without_raw_data.mat` contains `time_train (1,39988)`,
`time_test (1,3636)`, `y_train/y_test` positions `(6,N)` in degrees and
`u_train/u_test` motor torques `(6,N)` in Nm.  Its sample time is about
`0.100002500812 s`; it does **not** contain velocity or motor/link-side
encoder separation.

### Prepared inverse-identification file

`inverse_identification_without_raw_data.mat` contains output motor torque
`y_train/y_test (6,N)` in Nm and input `u_train/u_test (18,N)`, ordered as
six position, six velocity, then six acceleration channels in deg, deg/s and
deg/s².  It is an inverse-dynamics dataset, not a torque-replay source.

There is no `KukaDirectDynamics.mat` file in this official benchmark.  That
name and the assumed `(T,35)` schema in the external notebook are incompatible
with this verified 6-DoF resource.
