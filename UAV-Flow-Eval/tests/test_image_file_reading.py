# SPDX-License-Identifier: Apache-2.0
"""Read the UE file-mode response as an actual image without running UE."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from gym_unrealcv.envs.agent.character import Character_API


class ImageFileReadingTests(unittest.TestCase):
    def test_file_mode_reads_png_in_opencv_channel_order(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "frame.png"
            Image.new("RGB", (8, 6), (17, 35, 67)).save(image)
            api = Character_API.__new__(Character_API)
            api.ip = "127.0.0.1"
            api.docker = False
            api.client = MagicMock()
            api.client.request.return_value = str(image)
            result = api.read_image(5, "lit", "file")
            self.assertEqual(result.shape, (6, 8, 3))
            np.testing.assert_array_equal(result[0, 0], [67, 35, 17])
            api.client.request.assert_called_once_with("vget /camera/5/lit lit127.0.0.1.png")
