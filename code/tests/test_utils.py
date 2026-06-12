"""Test module for utils"""

import os
import platform
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

sys.path.append("../")

# ---------------------------------------------------------------------------
# Stub optional heavy dependencies that may not be installed in all envs.
# These must be in sys.modules BEFORE utils.py is imported so that top-level
# imports inside utils.py succeed.
# ---------------------------------------------------------------------------
class _DaskProxy(np.ndarray):
    """numpy ndarray subclass with a .compute() method for dask compatibility."""

    def compute(self):
        return np.asarray(self)


def _setup_missing_modules():
    # dask / dask.array
    # NOTE: `import dask.array as da` resolves da via getattr(sys.modules["dask"], "array"),
    # NOT directly from sys.modules["dask.array"].  The parent mock's .array attribute
    # must therefore point at our configured sub-mock.
    try:
        import dask.array  # noqa: F401
    except (ImportError, RuntimeError):
        _da = MagicMock()
        _da.stack = lambda arrays, **kw: np.stack(arrays)
        _da.squeeze = lambda x, **kw: np.squeeze(np.asarray(x)).view(_DaskProxy)
        _da.zeros = lambda shape, **kw: np.zeros(shape).view(_DaskProxy)
        _da.from_zarr = MagicMock(
            return_value=np.zeros((1, 1, 1, 20, 20)).view(_DaskProxy)
        )
        _dask = MagicMock()
        _dask.array = _da  # must be set so `import dask.array as da` resolves correctly
        sys.modules.setdefault("dask", _dask)
        sys.modules.setdefault("dask.array", _da)

    # psutil
    try:
        import psutil  # noqa: F401
    except ImportError:
        _psutil = MagicMock()
        _psutil.cpu_count = MagicMock(return_value=4)
        sys.modules.setdefault("psutil", _psutil)

    # natsort
    try:
        import natsort  # noqa: F401
    except ImportError:
        _natsort = MagicMock()
        _natsort.natsorted = sorted  # fall back to plain sort for tests
        sys.modules.setdefault("natsort", _natsort)

    # boto3
    try:
        import boto3  # noqa: F401
    except ImportError:
        sys.modules.setdefault("boto3", MagicMock())

    # aind_data_schema and sub-modules
    global AIND_DATA_SCHEMA_AVAILABLE
    AIND_DATA_SCHEMA_AVAILABLE = True
    for _mod in (
        "aind_data_schema",
        "aind_data_schema.core",
        "aind_data_schema.core.processing",
        "aind_data_schema.components",
        "aind_data_schema.components.identifiers",
    ):
        try:
            __import__(_mod)
        except ImportError:
            sys.modules.setdefault(_mod, MagicMock())
            AIND_DATA_SCHEMA_AVAILABLE = False


AIND_DATA_SCHEMA_AVAILABLE = True


_setup_missing_modules()

# Re-import dask.array after stubs are in place
import dask.array as da  # noqa: E402

from aind_smartspim_flatfield_estimation.utils import (  # noqa: E402
    ResourceMonitor,
    create_folder,
    generate_processing,
    get_brain_slices,
    get_code_ocean_cpu_limit,
    get_col_rows_per_laser,
    get_slicer_per_side,
    pick_slices,
    read_json_as_dict,
)

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"

JSON_FILE_PATH = RESOURCES_DIR / "local_json.json"
METADATA_FILE_PATH = RESOURCES_DIR / "metadata.json"


class TestUtilities(unittest.TestCase):
    """
    Test utilities
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        # Folder to test the zarr writing from PNGs
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

    @patch("os.environ.get")
    @patch("psutil.cpu_count")
    def test_get_code_ocean_cpu_limit(self, mock_cpu_count, mock_env_get):
        """
        Tests we get the code ocean CPU limits if
        it's a code ocean instance
        """
        mock_env_get.side_effect = lambda x: "4" if x == "CO_CPUS" else None
        mock_cpu_count.return_value = 8

        result = get_code_ocean_cpu_limit()
        self.assertEqual(result, 4)
        self.assertIsInstance(result, int)

        mock_env_get.side_effect = lambda x: None
        with patch("builtins.open", mock_open(read_data="100000")) as mock_file:
            self.assertEqual(get_code_ocean_cpu_limit(), 1)

        mock_file.side_effect = FileNotFoundError
        self.assertEqual(get_code_ocean_cpu_limit(), 8)

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "job_id"}, clear=True)
    def test_get_code_ocean_cpu_limit_aws_batch(self):
        """
        Tests the case where it's a pipeline execution
        """
        self.assertEqual(get_code_ocean_cpu_limit(), 1)

    def test_pick_slices(self):
        """
        Tests we pick slices for the datasets
        """
        image_stack = np.random.rand(10, 20, 20)
        percentage = 0.5
        picked_slices, slices = pick_slices(image_stack, percentage)

        z_dim = image_stack.shape[0]
        start_slice = int(z_dim * 0.2)
        end_slice = z_dim - start_slice + 1
        result_slices = list(range(start_slice, end_slice, 2))

        self.assertEqual(len(slices), len(result_slices))
        self.assertEqual(picked_slices.shape[0], len(result_slices))

    @patch("os.path.exists", return_value=False)
    def test_get_col_rows_per_laser_file_not_found(self, mock_exists):
        """
        Tests getting columns and rows when the metadata
        file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            get_col_rows_per_laser(Path("mock_metadata.json"))

    @patch("aind_smartspim_flatfield_estimation.utils.get_brain_slices")
    @patch("aind_smartspim_flatfield_estimation.utils.Path.glob")
    def test_get_slices_per_side(self, mock_glob, mock_get_brain_slices):
        """
        Get slices per laser side
        """
        # Mock the tiles_per_laser dictionary
        tiles_per_laser = {
            "0": ["1_1", "2_2"],
            "1": ["3_3", "4_4"],
        }

        # Mock the channel_path
        channel_path = "/mock/path/to/channel"

        # Mock indices
        indices = [0, 1]

        # Mock Zarr dataset structure
        mock_folders = [
            MagicMock(suffix=".zarr", stem="1_1"),
            MagicMock(suffix=".zarr", stem="2_2"),
            MagicMock(suffix=".zarr", stem="3_3"),
            MagicMock(suffix=".zarr", stem="4_4"),
        ]
        mock_glob.return_value = mock_folders

        # Mock get_brain_slices return value
        mock_slices = np.random.rand(2, 10, 10)  # Random 3D arrays
        mock_names = ["1_1.zarr", "2_2.zarr"]
        mock_get_brain_slices.side_effect = [
            (mock_slices, mock_names),
            (mock_slices, mock_names),
        ]

        # Call the function
        result = get_slicer_per_side(tiles_per_laser, channel_path, indices)

        # Assertions
        self.assertIn("0", result)
        self.assertIn("1", result)

        self.assertEqual(result["0"].shape, (4, 10, 10))

        # Check if the mocked functions were called correctly
        mock_glob.assert_called_once_with("*.zarr")
        self.assertEqual(mock_get_brain_slices.call_count, len(indices))

    def test_get_col_rows_per_laser(self):
        """
        Tests we get columns and rows per laser.
        """
        result = get_col_rows_per_laser(METADATA_FILE_PATH)
        expected_result = {"0": ["439030_262420"], "1": []}

        self.assertEqual(result, expected_result)

    def test_read_json_as_dict(self):
        """
        Tests successful reading of a dictionary
        """
        expected_result = {"some_key": "some_value"}
        result = read_json_as_dict(JSON_FILE_PATH)
        self.assertEqual(expected_result, result)

    @patch("dask.array.from_zarr", return_value=da.zeros((1, 1, 1, 20, 20)))
    def test_get_brain_slices(self, mock_from_zarr):
        """
        Tests that we get lazy brain slices
        """
        dataset_path = Path("/mock_path")
        cols = ["1", "2"]
        rows = ["A", "B"]
        slide_idx = 0

        imgs, names = get_brain_slices(dataset_path, cols, rows, slide_idx)
        self.assertEqual(imgs.shape[0], 4)
        self.assertEqual(len(names), 4)

    def test_create_folder(self):
        """
        Tests the creation of a folder
        """
        with patch("os.makedirs") as mock_makedirs:
            create_folder("mock_folder", verbose=True)
            mock_makedirs.assert_called_once()

    @unittest.skipUnless(AIND_DATA_SCHEMA_AVAILABLE, "aind_data_schema not installed")
    def test_generate_processing(self):
        """
        Tests that we generate the processing manifest
        """
        generate_processing(
            data_processes=[],
            dest_processing=self.temp_folder,
            pipeline_name="SmartSPIM Pipeline",
            pipeline_version="1.0",
            pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        )

        processing_path = Path(self.temp_folder).joinpath("processing.json")

        self.assertEqual(processing_path.exists(), True)

    @unittest.skipUnless(AIND_DATA_SCHEMA_AVAILABLE, "aind_data_schema not installed")
    def test_resource_monitor(self):
        """
        Tests that the resource monitor collects CPU/RAM samples and
        produces a valid ResourceUsage record.
        """
        monitor = ResourceMonitor(interval_seconds=0.05)
        monitor.start()
        try:
            time.sleep(0.2)
        finally:
            monitor.stop()

        resources = monitor.to_resource_usage(cpu_cores=4)

        self.assertEqual(resources.os, platform.system())
        self.assertEqual(resources.architecture, platform.machine())
        self.assertEqual(resources.cpu_cores, 4)
        self.assertGreater(len(resources.cpu_usage), 0)
        self.assertGreater(len(resources.ram_usage), 0)
        for sample in resources.cpu_usage + resources.ram_usage:
            self.assertGreaterEqual(sample.usage, 0)
            self.assertLessEqual(sample.usage, 100)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)


class TestGetCodeOceanCpuLimitReturnType(unittest.TestCase):
    """get_code_ocean_cpu_limit must always return int, never str."""

    @patch("os.environ.get")
    def test_co_cpus_env_returns_int(self, mock_env_get):
        mock_env_get.side_effect = lambda x: "8" if x == "CO_CPUS" else None
        result = get_code_ocean_cpu_limit()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 8)

    @patch.dict(os.environ, {"AWS_BATCH_JOB_ID": "some-id"}, clear=True)
    def test_aws_batch_returns_int(self):
        result = get_code_ocean_cpu_limit()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)


class TestPickSlicesEagerPath(unittest.TestCase):
    """pick_slices with read_lazy=False must return a numpy array, not None."""

    def test_eager_returns_numpy_array(self):
        image_stack = np.random.rand(20, 16, 16)
        picked, indices = pick_slices(image_stack, percentage=0.3, read_lazy=False)
        self.assertIsNotNone(picked)
        self.assertIsInstance(picked, np.ndarray)
        self.assertEqual(picked.shape[0], len(indices))

    def test_lazy_returns_dask_array(self):
        image_stack = np.random.rand(20, 16, 16)
        picked, indices = pick_slices(image_stack, percentage=0.3, read_lazy=True)
        self.assertIsNotNone(picked)
        self.assertEqual(picked.shape[0], len(indices))

    def test_too_low_percentage_raises(self):
        image_stack = np.random.rand(5, 4, 4)
        with self.assertRaises(ValueError):
            pick_slices(image_stack, percentage=0.001)


class TestCreateFolderErrnoFix(unittest.TestCase):
    """create_folder must not raise AttributeError from os.errno."""

    def test_existing_folder_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            create_folder(tmp)  # already exists — must not crash

    def test_new_folder_is_created(self):
        with tempfile.TemporaryDirectory() as parent:
            new_dir = os.path.join(parent, "new_subfolder")
            create_folder(new_dir)
            self.assertTrue(os.path.isdir(new_dir))


class TestGetColRowsTypoFix(unittest.TestCase):
    """get_col_rows_per_laser must work without crashing on the variable name typo fix."""

    def test_reads_metadata_without_attribute_error(self):
        result = get_col_rows_per_laser(METADATA_FILE_PATH)
        self.assertIn("0", result)
        self.assertIn("1", result)
        for side_tiles in result.values():
            self.assertIsInstance(side_tiles, list)


class TestGetSlicerPerSideSetLookup(unittest.TestCase):
    """get_slicer_per_side must use O(1) set lookups (laser_sets) and not crash."""

    @patch("aind_smartspim_flatfield_estimation.utils.get_brain_slices")
    @patch("aind_smartspim_flatfield_estimation.utils.Path.glob")
    def test_correct_assignment_with_set_lookups(
        self, mock_glob, mock_get_brain_slices
    ):
        tiles_per_laser = {"0": ["A_1", "B_2"], "1": ["C_3"]}

        mock_folders = [
            MagicMock(suffix=".zarr", stem="A_1"),
            MagicMock(suffix=".zarr", stem="C_3"),
        ]
        mock_glob.return_value = mock_folders

        mock_slices = np.random.rand(2, 8, 8)
        mock_names = ["A_1.zarr", "C_3.zarr"]
        mock_get_brain_slices.return_value = (mock_slices, mock_names)

        from aind_smartspim_flatfield_estimation.utils import \
            get_slicer_per_side

        result = get_slicer_per_side(tiles_per_laser, "/mock/path", indices=[0])

        self.assertEqual(result["0"].shape[0], 1)
        self.assertEqual(result["1"].shape[0], 1)

    @patch("aind_smartspim_flatfield_estimation.utils.get_brain_slices")
    @patch("aind_smartspim_flatfield_estimation.utils.Path.glob")
    def test_unknown_tile_raises_value_error(self, mock_glob, mock_get_brain_slices):
        tiles_per_laser = {"0": ["A_1"], "1": ["B_2"]}
        mock_glob.return_value = [MagicMock(suffix=".zarr", stem="X_9")]
        mock_get_brain_slices.return_value = (
            np.random.rand(1, 8, 8),
            ["X_9.zarr"],
        )

        from aind_smartspim_flatfield_estimation.utils import \
            get_slicer_per_side

        with self.assertRaises(ValueError):
            get_slicer_per_side(tiles_per_laser, "/mock/path", indices=[0])
