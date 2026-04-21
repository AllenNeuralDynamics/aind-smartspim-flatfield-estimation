# aind-smartspim-flatfield-estimation

Retrospective flatfield estimation for SmartSPIM light-sheet microscopy data. When the microscope does not supply a measured flatfield, this pipeline estimates one from the acquired tiles using [BaSiCPy](https://github.com/peng-lab/BaSiCPy) and applies the correction to large-scale Zarr datasets.

A separate flatfield is computed per laser channel because each laser can have a different illumination profile at acquisition time. For datasets with two illumination sides (left / right hemisphere), a per-hemisphere flatfield is also computed.

---

## How it works

```
Zarr tiles on disk or S3
        │
        ▼
  Pick representative Z-slices (20 %–80 % of stack, configurable %)
        │
        ▼
  BaSiCPy fit  →  flatfield + darkfield + baseline  (per slide)
        │
        ▼
  unify across slides  →  single flatfield per laser
        │
        ▼
  Bicubic upsample to full resolution
        │
        ▼
  Save estimated_flat_laser_<channel>_side_<side>.tif
```

**Optional stripe removal** (`filter_stripes`) can be applied before flatfield correction. It uses wavelet decomposition + FFT Gaussian filtering to suppress horizontal banding artifacts common in light-sheet data.

---

## Repository layout

```
aind-smartspim-flatfield-estimation/
├── code/
│   ├── run_capsule.py                          # Entry point (Code Ocean capsule)
│   ├── aind_smartspim_flatfield_estimation/
│   │   ├── flatfield_estimation.py             # BaSiCPy fitting, correction, unification
│   │   ├── filtering.py                        # Stripe removal, normalization, hemisphere routing
│   │   └── utils.py                            # I/O helpers, slice picking, S3 utilities
│   └── tests/
│       ├── test_flatfield_estimation.py
│       ├── test_filtering.py
│       └── test_utils.py
└── environment/
    └── Dockerfile                              # Production container definition
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.9 |
| numpy | 1.24.2 |
| BaSiCPy | 1.1.0 |
| jax / jaxlib | 0.4.23 |
| dask[distributed] | latest |
| zarr | 2.18.2 |
| PyWavelets | 1.6.0 |
| natsort | 8.4.0 |
| aind-data-schema | 1.0.0 |
| boto3 / s3fs | latest |

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
| `preprocess_<channel>.json` | Per-channel config with `input_data` and `channel` keys |

Then execute:

```bash
python run_capsule.py
```

Results are written to `/results/`:

```
results/
├── estimated_flat_laser_<channel>_side_0.tif
├── estimated_flat_laser_<channel>_side_1.tif
├── laser_tiles.json          # tile-to-side mapping
└── metadata/
    └── processing.json       # AIND provenance record
```

### Local / S3 usage

`preprocess_<channel>.json` supports both local paths and S3 URIs:

```json
{
    "input_data": "s3://my-bucket/my-dataset",
    "channel": "Ex_488_Em_525"
}
```

For local data, set `input_data` to an absolute directory path containing `<col>_<row>.zarr` folders.

---

## Key parameters

These are set in `run_capsule.py` and can be tuned per dataset:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_step_percentage` | `0.3` | Fraction of Z-planes sampled for fitting (avoids blank top/bottom slices) |
| `SCALE` | `2` | Zarr multiscale level used during fitting (lower = faster, coarser) |
| `smoothness_flatfield` | `1.0` | BaSiCPy flatfield smoothness regularisation |
| `smoothness_darkfield` | `20` | BaSiCPy darkfield smoothness regularisation |
| `max_reweight_iterations` | `35` | BaSiCPy optimisation iterations |
| `get_darkfield` | `False` | Whether BaSiCPy estimates a darkfield component |

---

## Public API

### `flatfield_estimation.py`

```python
shading_correction(slides, shading_parameters, mask=None)
```
Fits BaSiCPy on a list of 2-D image slices. Returns `{"flatfield", "darkfield", "baseline"}`.

```python
flatfield_correction(image_tiles, flatfield, darkfield, baseline=None)
```
Applies darkfield subtraction and flatfield division to a batch of tiles. Zero-valued flatfield pixels are handled safely (output clamped to 0). Returns `uint16` array.

### `filtering.py`

```python
log_space_fft_filtering(input_image, wavelet="db3", level=0, sigma=64, max_threshold=4)
```
Removes horizontal stripes via log-space wavelet decomposition + FFT Gaussian filtering. Correctly inverts the log transform as `exp(y) - 1`.

```python
normalize_image(images)
```
Normalises a batch of images to `[1.0, 2.0]`. Returns `float32`. Handles constant-valued batches without division-by-zero.

```python
filter_stripes(image, input_tile_path, no_cells_config, cells_config, shadow_correction=None)
```
Detects whether a tile contains cells and applies the appropriate stripe-removal configuration. Optionally applies retrospective or prospective flatfield correction.

### `utils.py`

```python
pick_slices(image_stack, percentage, read_lazy=True)
```
Samples a fixed percentage of Z-slices from the central 60 % of the stack. Returns slices as a Dask array (`read_lazy=True`) or NumPy array (`read_lazy=False`).

```python
get_col_rows_per_laser(metadata_json_path)
```
Parses a SmartSPIM metadata JSON and returns `{"0": [col_row, ...], "1": [...]}` mapping each laser side to its tile identifiers.

```python
get_slicer_per_side(tiles_per_laser, channel_path, indices, scale=2)
```
Loads image slices for the requested Z-indices and groups them by laser side.

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

> Tests stub out optional heavy dependencies (`basicpy`, `dask`, `psutil`, etc.) via `sys.modules` so the suite runs in environments that only have NumPy, SciPy, PyWavelets, and scikit-image.

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
