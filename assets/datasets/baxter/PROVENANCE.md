# Baxter dynamics dataset

## Upstream

- Canonical data record: <https://zenodo.org/records/17035193>
- API: <https://zenodo.org/api/records/17035193>
- DOI: <https://doi.org/10.5281/zenodo.17035193>
- Licence: **CC BY 4.0** (`cc-by-4.0`, as returned by the Zenodo API).
- Companion project: <https://github.com/EduardoRosLab/Baxter_Dynamic_Model>

## Local acquisition (2026-07-30)

One small, representative left-arm trajectory was downloaded to verify the
schema, instead of fetching the entire approximately 899 MB collection:

| File | Bytes | MD5 |
| --- | ---: | --- |
| `raw/left_circle_p-15_t105.csv` | 2,675,097 | `cb40dd7cf7f1a4833cf9e47f25490d0a` |

The record exposes the remaining trajectories as individual CSV downloads. A
bulk downloader should enumerate `files` from the API rather than hard-code a
file list.

## Verified schema

The data is **CSV, not MATLAB**. The verified sample has 14,907 observations,
one whitespace-delimited header, and 21 numeric columns in this exact order:

```text
ang_s0 ang_s1 ang_e0 ang_e1 ang_w0 ang_w1 ang_w2
vel_s0 vel_s1 vel_e0 vel_e1 vel_w0 vel_w1 vel_w2
torq_s0 torq_s1 torq_e0 torq_e1 torq_w0 torq_w1 torq_w2
```

The fields are left-arm joint position, velocity and applied torque. The
record documentation identifies the same `s0` through `w2` ordering. It has
no timestamp, no explicit `q_next/dq_next`, and no separate motor/link-side
encoders. A loader therefore must make the sample period a required dataset
configuration (not silently invent it) and construct transition targets only
within each individual file.

The recorded right-arm files use the same column layout. Dataset file names
follow `{arm}_{trajectory}_p{phi}_t{theta}.csv`.

## Local canonical export

`canonical/baxter_left_circle_p-15_t105.csv` is generated locally from the
raw sample with `sample_time_s=0.002`.  It uses the calibration-ready columns
`time`, `q0..q6`, `dq0..dq6`, and `tau0..tau6`; the raw source remains the
provenance-preserving ground truth.
