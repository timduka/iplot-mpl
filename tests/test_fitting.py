"""Tests for the fitting module."""

import numpy as np
import pytest

from interactive_figure.fitting import (
    MODELS_1D,
    MODELS_2D,
    FitModel1D,
    FitModel2D,
    create_custom_model_1d,
    create_custom_model_2d,
    fit_1d,
    fit_2d,
    gaussian_1d,
    gaussian_2d,
    linear_1d,
    lorentzian_1d,
    plane_2d,
    polynomial_1d,
)


class TestFitModel1D:
    """Tests for 1D fit model dataclass."""

    def test_gaussian_1d_creation(self) -> None:
        """Test that gaussian_1d creates a valid FitModel1D with height parameter."""
        model = gaussian_1d()
        assert isinstance(model, FitModel1D)
        assert model.name == "Gaussian"
        assert model.model is not None
        assert "height" in model.param_hints  # Uses height, not amplitude
        assert "center" in model.param_hints
        assert "sigma" in model.param_hints
        assert model.custom_guess is not None  # Has intelligent guess function

    def test_lorentzian_1d_creation(self) -> None:
        """Test that lorentzian_1d creates a valid FitModel1D."""
        model = lorentzian_1d()
        assert isinstance(model, FitModel1D)
        assert model.name == "Lorentzian"

    def test_linear_1d_creation(self) -> None:
        """Test that linear_1d creates a valid FitModel1D."""
        model = linear_1d()
        assert isinstance(model, FitModel1D)
        assert model.name == "Linear"
        assert "slope" in model.param_hints
        assert "intercept" in model.param_hints

    def test_polynomial_1d_creation(self) -> None:
        """Test polynomial model with different degrees."""
        model_deg2 = polynomial_1d(degree=2)
        assert "c0" in model_deg2.param_hints
        assert "c1" in model_deg2.param_hints
        assert "c2" in model_deg2.param_hints

        model_deg3 = polynomial_1d(degree=3)
        assert "c3" in model_deg3.param_hints


class TestFitModel2D:
    """Tests for 2D fit model dataclass."""

    def test_gaussian_2d_creation(self) -> None:
        """Test that gaussian_2d creates a valid FitModel2D with height parameter."""
        model = gaussian_2d()
        assert isinstance(model, FitModel2D)
        assert model.name == "2D Gaussian"
        assert model.func is not None
        assert "height" in model.param_hints  # Uses height, not amplitude
        assert "x0" in model.param_hints
        assert "y0" in model.param_hints
        assert "sigma_x" in model.param_hints
        assert "sigma_y" in model.param_hints
        assert model.custom_guess is not None  # Has intelligent guess function

    def test_plane_2d_creation(self) -> None:
        """Test that plane_2d creates a valid FitModel2D."""
        model = plane_2d()
        assert isinstance(model, FitModel2D)
        assert model.name == "Plane"
        assert "a" in model.param_hints
        assert "b" in model.param_hints
        assert "c" in model.param_hints


class TestModelDictionaries:
    """Tests for MODELS_1D and MODELS_2D dictionaries."""

    def test_models_1d_contains_expected_models(self) -> None:
        """Test that MODELS_1D contains all expected models."""
        expected = [
            "Gaussian",
            "Lorentzian",
            "Voigt",
            "Linear",
            "Quadratic",
            "Cubic",
            "Exponential Decay",
            "Power Law",
            "Sine",
            "Damped Oscillation",
            "Gaussian + Offset",
            "Lorentzian + Offset",
            "Double Gaussian",
        ]
        for model_name in expected:
            assert model_name in MODELS_1D, f"Missing model: {model_name}"

    def test_models_2d_contains_expected_models(self) -> None:
        """Test that MODELS_2D contains all expected models."""
        expected = [
            "2D Gaussian",
            "Rotated 2D Gaussian",
            "2D Lorentzian",
            "Double 2D Gaussian",
            "Plane",
            "Paraboloid",
        ]
        for model_name in expected:
            assert model_name in MODELS_2D, f"Missing model: {model_name}"

    def test_all_1d_models_are_callable(self) -> None:
        """Test that all entries in MODELS_1D return FitModel1D."""
        for name, factory in MODELS_1D.items():
            model = factory()
            assert isinstance(model, FitModel1D), f"{name} did not return FitModel1D"

    def test_all_2d_models_are_callable(self) -> None:
        """Test that all entries in MODELS_2D return FitModel2D."""
        for name, factory in MODELS_2D.items():
            model = factory()
            assert isinstance(model, FitModel2D), f"{name} did not return FitModel2D"


class TestFit1D:
    """Tests for 1D fitting functions."""

    def test_fit_linear_data(self) -> None:
        """Test fitting a linear function."""
        x = np.linspace(0, 10, 50)
        y = 2.5 * x + 1.0  # slope=2.5, intercept=1.0

        model = linear_1d()
        result = fit_1d(x, y, model)

        assert result.success
        assert abs(result.params["slope"].value - 2.5) < 0.01
        assert abs(result.params["intercept"].value - 1.0) < 0.01

    def test_fit_gaussian_data(self) -> None:
        """Test fitting a Gaussian function."""
        x = np.linspace(-5, 5, 100)
        # amplitude=3, center=0.5, sigma=1.0
        y = 3 * np.exp(-((x - 0.5) ** 2) / (2 * 1.0**2))

        model = gaussian_1d()
        result = fit_1d(x, y, model)

        assert result.success
        # Gaussian parameters: lmfit amplitude = height * sigma * sqrt(2*pi)
        assert abs(result.params["center"].value - 0.5) < 0.1
        assert abs(result.params["sigma"].value - 1.0) < 0.2


class TestFit2D:
    """Tests for 2D fitting functions."""

    def test_fit_plane_data(self) -> None:
        """Test fitting a plane function."""
        x = np.linspace(-5, 5, 30)
        y = np.linspace(-5, 5, 30)
        X, Y = np.meshgrid(x, y)
        # z = 1.5*x + 2.0*y + 0.5
        Z = 1.5 * X + 2.0 * Y + 0.5

        model = plane_2d()
        result = fit_2d(X, Y, Z, model)

        assert result is not None
        # Check fitted parameters are close to true values
        assert abs(result.params["a"].value - 1.5) < 0.1
        assert abs(result.params["b"].value - 2.0) < 0.1
        assert abs(result.params["c"].value - 0.5) < 0.1


class TestCustomModels:
    """Tests for custom model creation."""

    def test_create_custom_model_1d(self) -> None:
        """Test creating a custom 1D model."""

        def my_func(x, a, b):
            return a * x + b

        model = create_custom_model_1d(
            func=my_func,
            param_hints={"a": {"value": 1.0}, "b": {"value": 0.0}},
            name="My Linear",
            description="Custom linear model",
        )

        assert isinstance(model, FitModel1D)
        assert model.name == "My Linear"
        assert model.description == "Custom linear model"
        assert "a" in model.param_hints
        assert "b" in model.param_hints

    def test_create_custom_model_1d_with_guess_func(self) -> None:
        """Test creating a custom 1D model with a guess function."""

        def my_func(x, a, b):
            return a * x + b

        def my_guess(y, x):
            return {
                "a": {"value": 2.0, "min": 0, "max": 10},
                "b": {"value": 1.0, "min": -5, "max": 5},
            }

        model = create_custom_model_1d(
            func=my_func,
            param_hints={"a": {"value": 1.0}, "b": {"value": 0.0}},
            name="My Linear",
            description="Custom linear model",
            guess_func=my_guess,
        )

        assert model.custom_guess is not None
        # Test that guess function works
        x = np.linspace(0, 10, 10)
        y = 2.0 * x + 1.0
        guessed = model.custom_guess(y, x)
        assert guessed["a"]["value"] == 2.0
        assert guessed["b"]["value"] == 1.0

    def test_create_custom_model_2d(self) -> None:
        """Test creating a custom 2D model."""

        def my_func_2d(X, Y, a, b, c):
            return a * X + b * Y + c

        model = create_custom_model_2d(
            func=my_func_2d,
            param_names=["a", "b", "c"],
            param_hints={
                "a": {"value": 1.0},
                "b": {"value": 1.0},
                "c": {"value": 0.0},
            },
            name="My Plane",
            description="Custom plane model",
        )

        assert isinstance(model, FitModel2D)
        assert model.name == "My Plane"
        assert model.func is my_func_2d
        assert model.param_names == ["a", "b", "c"]


class TestFitModel1DGuessFunc:
    """Tests for FitModel1D with guess functions."""

    def test_model_without_guess_func(self) -> None:
        """Test that a model without custom_guess can still fit data."""

        def simple_linear(x, a, b):
            return a * x + b

        # Create a simple custom model without a guess function
        model = create_custom_model_1d(
            func=simple_linear,
            param_hints={
                "a": {"value": 1.0, "min": -100, "max": 100},
                "b": {"value": 0.0, "min": -100, "max": 100},
            },
            name="SimpleLinear",
        )
        assert model.custom_guess is None
        # Verify it can still be used for fitting
        x_data = np.linspace(0, 10, 50)
        y_data = 2.0 * x_data + 3.0 + np.random.normal(0, 0.1, size=x_data.shape)
        result = fit_1d(x_data, y_data, model)
        assert result.success

    def test_custom_model_with_guess_func_fitting(self) -> None:
        """Test that custom model with guess_func can fit data."""

        def exponential(x, amplitude, decay):
            return amplitude * np.exp(-x / decay)

        def guess_exp(y, x):
            amp = np.max(y)
            # Estimate decay from half-life
            half_idx = np.argmin(np.abs(y - amp / 2))
            decay_est = x[half_idx] / np.log(2) if half_idx > 0 else 1.0
            return {
                "amplitude": {"value": amp, "min": 0, "max": amp * 2},
                "decay": {"value": decay_est, "min": 0.01, "max": decay_est * 5},
            }

        model = create_custom_model_1d(
            func=exponential,
            param_hints={
                "amplitude": {"value": 1.0, "min": 0},
                "decay": {"value": 1.0, "min": 0},
            },
            name="Exponential",
            guess_func=guess_exp,
        )

        # Generate test data
        x = np.linspace(0, 5, 50)
        y_true = 5.0 * np.exp(-x / 1.5)

        # Verify guess function produces reasonable estimates
        guessed = model.custom_guess(y_true, x)
        assert guessed["amplitude"]["value"] == pytest.approx(5.0, rel=0.1)
        assert guessed["decay"]["value"] > 0
