"""Test module for utils"""

import os
import platform
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_smartspim_flatfield_estimation.utils import (
    ResourceMonitor,
    create_folder,
    generate_processing,
    get_col_rows_per_laser,
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
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

    @patch("os.path.exists", return_value=False)
    def test_get_col_rows_per_laser_file_not_found(self, mock_exists):
        """
        Tests getting columns and rows when the metadata
        file does not exist.
        """
        with self.assertRaises(FileNotFoundError):
            get_col_rows_per_laser(Path("mock_metadata.json"))

    def test_get_col_rows_per_laser(self):
        """
        Tests we get columns and rows per laser.
        """
        result = get_col_rows_per_laser(METADATA_FILE_PATH)
        expected_result = {"0": ["439030_262420"], "1": []}

        self.assertEqual(result, expected_result)

    def test_get_col_rows_per_laser_sides_are_lists(self):
        """
        Tests that every side has a list of tiles, even an empty one.
        """
        result = get_col_rows_per_laser(METADATA_FILE_PATH)

        self.assertIn("0", result)
        self.assertIn("1", result)
        for side_tiles in result.values():
            self.assertIsInstance(side_tiles, list)

    def test_read_json_as_dict(self):
        """
        Tests successful reading of a dictionary
        """
        expected_result = {"some_key": "some_value"}
        result = read_json_as_dict(JSON_FILE_PATH)
        self.assertEqual(expected_result, result)

    def test_read_json_as_dict_missing_file(self):
        """
        Tests that a missing file gives an empty dictionary
        """
        self.assertEqual(read_json_as_dict(RESOURCES_DIR / "does_not_exist.json"), {})

    def test_create_folder(self):
        """
        Tests the creation of a folder
        """
        with patch("os.makedirs") as mock_makedirs:
            create_folder("mock_folder", verbose=True)
            mock_makedirs.assert_called_once()

    def test_create_folder_existing_does_not_raise(self):
        """
        Tests that an existing folder does not raise an error
        """
        with tempfile.TemporaryDirectory() as tmp:
            create_folder(tmp)

    def test_create_folder_new_folder_is_created(self):
        """
        Tests that a new folder is created on disk
        """
        with tempfile.TemporaryDirectory() as parent:
            new_dir = os.path.join(parent, "new_subfolder")
            create_folder(new_dir)
            self.assertTrue(os.path.isdir(new_dir))

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


if __name__ == "__main__":
    unittest.main()
