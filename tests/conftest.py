"""Pytest configuration and fixtures for interactive_figure tests."""

import matplotlib
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def reset_matplotlib() -> None:
    """Reset matplotlib state between tests."""
    import matplotlib.pyplot as plt

    plt.close("all")
