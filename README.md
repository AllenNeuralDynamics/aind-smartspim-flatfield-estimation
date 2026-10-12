# aind-smartspim-flatfield-estimation

![support](https://img.shields.io/badge/support-supported-brightgreen)

Retrospective flatfield estimation for SmartSPIM light-sheet microscopy data. When the microscope does not supply a measured flatfield, this capsule estimates one from the acquired tiles.

The estimation itself is done by the [aind-flatfield-correction](https://github.com/AllenNeuralDynamics/aind-flatfield-correction) package (BaSiCPy based). This repository only does the SmartSPIM specific work: it reads the capsule inputs, calls the package for each channel, writes the flatfield files that the SmartSPIM destripe capsule reads, and writes the AIND processing metadata.

A separate flatfield is computed for each channel (laser). The same flatfield is written for each brain side.

---

## How it works

```
preprocess_<channel>.json ──► channel folder (s3://.../<channel> or local)
                                   │
derivatives/DarkMaster_cropped.tif ┤
                                   ▼
        aind-flatfield-correction (joint fit + parameter search)
        - tiles at pyramid level 2, darkfield subtracted before the fit
        - search for smoothness_flatfield (mandatory)
        - plausibility check, validation figures
                                   │
                                   ▼
        flatfield_<channel>_level0.npy + flatfield_<channel>.json
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
   estimated_flat_laser_<channel>_side_<s>.tif   metadata/processing.json
```

The flatfield is estimated on the raw (not destriped) tiles. The destripe capsule applies it after destriping.

### When the capsule fails

The capsule writes all its outputs first. Then it raises an error, so you can inspect the results. It raises:

| Error | Cause |
|---|---|
| `SuspiciousFlatfieldError` | The package plausibility check failed for at least one channel (flat is a no-op, has non-finite values, or is out of range). The reasons are in `processing.json` notes and in the package sidecar json. |
| `SearchSkippedError` | The package did not run the parameter search (`over_budget` or `all_candidates_failed`). The search is mandatory. |

Every channel runs before the error is raised.

---

## Repository layout

```
aind-smartspim-flatfield-estimation/
├── code/
│   ├── run_capsule.py                          # Entry point (Code Ocean capsule)
│   ├── aind_smartspim_flatfield_estimation/
│   │   ├── estimation.py                       # Package arguments, sidecar report, flatfield tifs, errors
│   │   └── utils.py                            # Metadata, folders, resource monitor, processing.json
│   └── tests/
│       ├── test_estimation.py
│       ├── test_run_capsule.py
│       └── test_utils.py
└── environment/
    └── Dockerfile_local                        # Container definition
```

---

## Requirements

Python 3.10 or 3.11. The estimation dependencies (BaSiCPy, jax, numpy, zarr, dask, boto3) are pinned by `aind-flatfield-correction`. This repository adds:

| Package | Version |
|---------|---------|
| aind-flatfield-correction | pinned git commit (see `pyproject.toml`) |
| aind-data-schema | 2.8.0 |
| aind-data-schema-models | >=5.7.1,<6 |
| log-schema | 0.2.10.dev1 |

---

## Installation

**User install** (run from the repository root):

```bash
pip install -e .
```

**Development install** (includes linters, test runners, and documentation tools):

```bash
pip install -e .[dev]
```

> **Note:** BaSiCPy requires JAX. On machines without AVX support, build `jaxlib` from source or use the Docker image provided in `environment/`.

---

## Running the pipeline

### Code Ocean capsule

The pipeline is designed to run as a Code Ocean capsule. Place the following files in the capsule's `/data` folder:

| File | Description |
|------|-------------|
| `metadata.json` | Tile configuration (X/Y positions, laser sides) |
| `data_description.json` | Dataset metadata |
| `derivatives/DarkMaster_cropped.tif` | Microscope darkfield. The destripe capsule subtracts the same file. |
| `preprocess_<channel>.json` | Per-channel config with `input_data` and `channel` keys |

Then execute:

```bash
python run_capsule.py
```

Results are written to `/results/`:

```
results/
├── estimated_flat_laser_<channel>_side_0.tif     # read by the destripe capsule
├── estimated_flat_laser_<channel>_side_1.tif
├── laser_tiles.json                              # tile-to-side mapping
├── flatfield_estimation/<channel>/               # package products
│   ├── flatfield_<channel>.npy / _level0.npy
│   ├── darkfield_<channel>.npy / _level0.npy
│   ├── flatfield_<channel>.json                  # parameters, search report, guard verdict, provenance
│   ├── metadata/estimation.log
│   └── <channel>/*.png                           # validation figures (always written)
└── metadata/
    └── processing.json                           # AIND provenance record
```

Review the validation figures before you use a flatfield on a whole dataset.

### Local / S3 usage

`preprocess_<channel>.json` supports both local paths and S3 URIs:

```json
{
    "input_data": "s3://my-bucket/my-dataset",
    "channel": "Ex_488_Em_525"
}
```

For local data, set `input_data` to an absolute directory path containing `<col>_<row>.zarr` folders.

> **Note:** The package lists S3 tiles with an unsigned client. S3 input must be in a public bucket.

---

## Key parameters

These are constants in `estimation.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `PYRAMID_LEVEL` | `2` | Zarr multiscale level used for the fit |
| `TILE_PATTERN` | `\.zarr$` | Entries in the channel folder that are tiles |
| `SEARCH_BUDGET_MINUTES` | `480` | `--max-search-minutes`. The package skips the search if the projected cost is more than half of this. |

The package defaults are used for everything else (joint fit, `ladmap` solver, 2000 planes in total across all tiles). See the package README for its options.

---

## Public API

### `estimation.py`

```python
build_estimate_argv(channel_path, darkfield_path, output_folder, output_name, pyramid_level=2, validate=True)
```
Builds the package command-line arguments. `--validate` is on by default. `--skip-search` is never added.

```python
run_estimate(argv)
```
Runs the package estimation.

```python
read_estimation_report(sidecar_path)
```
Reads the package sidecar and returns the plausibility verdict and the search verdict.

```python
write_side_flats(flatfield_path, sides, channel_name, results_folder)
```
Writes `estimated_flat_laser_<channel>_side_<side>.tif` (float32) for each side.

### `utils.py`

```python
get_col_rows_per_laser(metadata_json_path)
```
Parses a SmartSPIM metadata JSON and returns `{"0": [col_row, ...], "1": [...]}` mapping each laser side to its tile identifiers.

```python
ResourceMonitor, generate_processing, read_json_as_dict, create_folder
```
Helpers for the AIND processing record and file handling.

---

## Contributing

### Linters and testing

Run all checks from the `code/` directory:

```bash
# Unit tests with coverage
coverage run -m unittest discover && coverage report

# Docstring coverage
interrogate .

# Style checks
flake8 .

# Auto-format
black .
isort .
```

> The tests do not run the package estimation. They replace it with a fake that writes the same files. The suite needs `aind-data-schema`, `aind-data-schema-models`, `numpy`, `tifffile`, `psutil` and the `aind_flatfield_correction.core.basicpy.config` module. `log_schema` is stubbed.

### Pull requests

For internal contributors: create a branch. For external contributors: fork the repository and open a pull request from the fork.

Commit messages follow [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style:

```
<type>(<scope>): <short summary>
```

| Type | When to use |
|------|-------------|
| `build` | Build tools or external dependencies |
| `ci` | CI configuration |
| `docs` | Documentation only |
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or correcting tests |

### Generating documentation

```bash
# Generate RST source files
sphinx-apidoc -o doc_template/source/ src

# Build HTML
sphinx-build -b html doc_template/source/ doc_template/build/html
```

See the [Sphinx installation guide](https://www.sphinx-doc.org/en/master/usage/installation.html) for setup instructions.

---

## License

[MIT](LICENSE) — Allen Institute for Neural Dynamics
