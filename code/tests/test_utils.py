import os
import sys
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


class TestUtilities(unittest.TestCase):

    @patch("os.environ.get")
    @patch("psutil.cpu_count")
    def test_get_code_ocean_cpu_limit(self, mock_cpu_count, mock_env_get):
        mock_env_get.side_effect = lambda x: "4" if x == "CO_CPUS" else None
        mock_cpu_count.return_value = 8

        self.assertEqual(get_code_ocean_cpu_limit(), "4")

        mock_env_get.side_effect = lambda x: None
        with patch("builtins.open", mock_open(read_data="100000")) as mock_file:
            self.assertEqual(get_code_ocean_cpu_limit(), 1)

        mock_file.side_effect = FileNotFoundError
        self.assertEqual(get_code_ocean_cpu_limit(), 8)

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_read_json_as_dict(self, mock_file, mock_exists):
        result = read_json_as_dict("mock_path.json")
        self.assertEqual(result, {"key": "value"})

    def test_pick_slices(self):
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
        with self.assertRaises(FileNotFoundError):
            get_col_rows_per_laser(Path("mock_metadata.json"))

    # @patch("aind_smartspim_flatfield_estimation.utils.read_json_as_dict", return_value={"tile_config": {"1": {"Side": "0", "X": 1, "Y": 1}}})
    # def test_get_col_rows_per_laser(self, mock_read_json):
    #     Path.exists = MagicMock(return_value=True)
    #     result = get_col_rows_per_laser(Path("mock_metadata.json"))
    #     self.assertIn("0", result)
    #     self.assertEqual(result["0"], ["1_1"])

    # @patch("dask.array.from_zarr", return_value=da.zeros((1, 1, 1, 20, 20)))
    # def test_get_brain_slices(self, mock_from_zarr):
    #     dataset_path = Path("/mock_path")
    #     cols = ["1", "2"]
    #     rows = ["A", "B"]
    #     slide_idx = 0

    #     imgs, names = get_brain_slices(dataset_path, cols, rows, slide_idx)
    #     self.assertEqual(imgs.shape[0], 4)
    #     self.assertEqual(len(names), 4)

    # def test_create_folder(self):
    #     with patch("os.makedirs") as mock_makedirs:
    #         create_folder("mock_folder", verbose=True)
    #         mock_makedirs.assert_called_once()

    # @patch("aind_data_schema.core.processing.PipelineProcess")
    # @patch("aind_data_schema.core.processing.Processing")
    # def test_generate_processing(self, mock_processing, mock_pipeline_process):
    #     mock_pipeline_instance = MagicMock()
    #     mock_pipeline_process.return_value = mock_pipeline_instance
    #     mock_processing_instance = MagicMock()
    #     mock_processing.return_value = mock_processing_instance

    #     generate_processing(
    #         data_processes=[],
    #         dest_processing="mock_path",
    #         processor_full_name="Test User",
    #         pipeline_version="1.0",
    #     )

    #     mock_pipeline_process.assert_called_once()
    #     mock_processing.assert_called_once()
    #     mock_processing_instance.write_standard_file.assert_called_with(output_directory="mock_path")
