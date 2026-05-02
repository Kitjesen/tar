# Copyright (c) 2024-2026 Inovxio
# SPDX-License-Identifier: Apache-2.0

"""Asset paths used by the Thunder training configs in this repository."""

from __future__ import annotations

import os

ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
"""Path to this repository's robot_lab extension root."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to bundled robot asset data such as URDF and meshes."""

__version__ = "0.1.0"
