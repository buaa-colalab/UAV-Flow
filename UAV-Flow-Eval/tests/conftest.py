# SPDX-License-Identifier: Apache-2.0
"""Only mark isolated suites here; GPU simulators are never started by default."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.parent.name == "evaluation" or item.path.name in {
            "test_training_service_regressions.py",
            "test_shared_workflows.py",
        }:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
