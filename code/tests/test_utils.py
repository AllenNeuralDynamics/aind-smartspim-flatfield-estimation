"""Test module for utils"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import dask.array as da
import numpy as np

sys.path.append("../")

from aind_smartspim_flatfield_estimation.utils import (
    create_folder, generate_processing, get_brain_slices,
    get_code_ocean_cpu_limit, get_col_rows_per_laser, get_slicer_per_side,
    pick_slices, read_json_as_dict)

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

        self.assertEqual(get_code_ocean_cpu_limit(), "4")

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

    def test_generate_processing(self):
        """
        Tests that we generate the processing manifest
        """
        generate_processing(
            data_processes=[],
            dest_processing=self.temp_folder,
            processor_full_name="Test User",
            pipeline_version="1.0",
        )

        processing_path = Path(self.temp_folder).joinpath("processing.json")

        self.assertEqual(processing_path.exists(), True)

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)
