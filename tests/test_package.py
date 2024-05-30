from __future__ import annotations

import importlib.metadata

import polygenic_adaptation as m


def test_version():
    assert importlib.metadata.version("polygenic_adaptation") == m.__version__
