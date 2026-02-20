"""Fitting Models for Interactive Figure Package.

This module provides 1D and 2D fitting models using lmfit for use with
InteractiveXYPlot and InteractiveHeatmap classes.

Key Design Decisions:
    - Peak functions (Gaussian, Lorentzian, Voigt) use **height** as the peak
      amplitude rather than area under the curve, making interactive fitting
      more intuitive.
    - All models include intelligent guess functions that constrain initial
      parameter bounds to realistic ranges based on the data extent.
    - Center positions are constrained to the data range.
    - Width parameters (sigma, gamma, FWHM) are constrained to at most the
      data extent.

Supported 1D Models:
    - Gaussian, Lorentzian, Voigt profiles (using peak height)
    - Linear, Polynomial (quadratic, cubic)
    - Exponential decay, Power law
    - Sine, Damped oscillation
    - Models with offset (Gaussian + Offset, etc.)
    - Double Gaussian

Supported 2D Models:
    - 2D Gaussian, Rotated 2D Gaussian
    - 2D Lorentzian
    - Double 2D Gaussian
    - Plane, Paraboloid

Example:
    >>> from interactive_figure.fitting import gaussian_1d, fit_1d
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.exp(-x**2) + np.random.normal(0, 0.1, 100)
    >>> model = gaussian_1d()
    >>> result = fit_1d(x, y, model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from lmfit import Model, Parameters
from lmfit.models import (
    ExponentialModel,
    LinearModel,
    PolynomialModel,
    PowerLawModel,
    SineModel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Custom Peak Functions (using height instead of area)
# =============================================================================


def _gaussian_height(x: NDArray, height: float, center: float, sigma: float) -> NDArray:
    """
    Gaussian peak function using peak height.

    Args:
        x: X values
        height: Peak height (value at center)
        center: Peak center position
        sigma: Standard deviation (width parameter)

    Returns:
        y = height * exp(-(x - center)² / (2 * sigma²))
    """
    return height * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _lorentzian_height(x: NDArray, height: float, center: float, sigma: float) -> NDArray:
    """
    Lorentzian peak function using peak height.

    Args:
        x: X values
        height: Peak height (value at center)
        center: Peak center position
        sigma: Half-width at half-maximum (HWHM)

    Returns:
        y = height / (1 + ((x - center) / sigma)²)
    """
    return height / (1 + ((x - center) / sigma) ** 2)


def _voigt_height(x: NDArray, height: float, center: float, sigma: float, gamma: float) -> NDArray:
    """
    Pseudo-Voigt peak function using peak height.

    This is a linear combination of Gaussian and Lorentzian with mixing
    parameter eta = gamma / (sigma + gamma).

    Args:
        x: X values
        height: Peak height (value at center)
        center: Peak center position
        sigma: Gaussian width parameter
        gamma: Lorentzian width parameter (HWHM)

    Returns:
        Pseudo-Voigt profile with specified peak height
    """
    # Mixing parameter
    eta = gamma / (sigma + gamma + 1e-10)
    # Gaussian component (normalized to height 1 at center)
    gauss = np.exp(-((x - center) ** 2) / (2 * sigma**2))
    # Lorentzian component (normalized to height 1 at center)
    lorentz = 1 / (1 + ((x - center) / gamma) ** 2)
    # Mix and scale
    return height * ((1 - eta) * gauss + eta * lorentz)


# =============================================================================
# 1D Fitting Models for InteractiveXYPlot
# =============================================================================


@dataclass
class FitModel1D:
    """
    Container for a 1D fit model with parameter information.

    Attributes:
        name: Display name of the model
        model: lmfit Model object
        param_hints: Dictionary of parameter hints (min, max, value, etc.)
        description: Short description of the model
        custom_guess: Optional custom guess function with signature (y, x) -> dict

    Note:
        Peak models (Gaussian, Lorentzian, Voigt) use 'height' as the peak
        amplitude rather than area under the curve, making interactive fitting
        more intuitive. The guess functions automatically constrain parameters
        to realistic ranges based on the data.
    """

    name: str
    model: Model
    param_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str = ""
    custom_guess: Callable[[NDArray, NDArray], dict[str, dict[str, Any]]] | None = None

    def guess(self, y: NDArray, x: NDArray) -> Parameters:
        """
        Make an initial guess for the model parameters.

        Args:
            y: Y data array
            x: X data array

        Returns:
            lmfit Parameters object with initial guesses
        """
        # If custom guess function is provided, use it
        if self.custom_guess is not None:
            try:
                guess_dict = self.custom_guess(y, x)
                params = self.model.make_params()
                for name, hints in guess_dict.items():
                    if name in params:
                        if isinstance(hints, dict):
                            # hints is a dict with value, min, max, etc.
                            for key, val in hints.items():
                                setattr(params[name], key, val)
                        else:
                            # hints is just the value
                            params[name].value = hints
                return params
            except Exception:
                pass

        try:
            # Use lmfit's built-in guess method if available
            return self.model.guess(y, x=x)
        except (NotImplementedError, AttributeError):
            # Fall back to default parameters
            return self.model.make_params()

    def get_param_bounds(self, y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        """
        Get parameter bounds (min, max, value) for slider initialization.

        Args:
            y: Y data array
            x: X data array

        Returns:
            Dictionary mapping parameter names to dicts with 'value', 'min', 'max'
        """
        bounds = {}

        # Start with param_hints as defaults
        for name in self.model.param_names:
            hints = self.param_hints.get(name, {})
            bounds[name] = {
                "value": hints.get("value", 1.0),
                "min": hints.get("min", -np.inf),
                "max": hints.get("max", np.inf),
            }

        # If custom guess function is provided, use it to update bounds
        if self.custom_guess is not None:
            try:
                guess_dict = self.custom_guess(y, x)
                for name, hints in guess_dict.items():
                    if name in bounds and isinstance(hints, dict):
                        for key in ["value", "min", "max"]:
                            if key in hints:
                                bounds[name][key] = hints[key]
            except Exception:
                pass
        else:
            # Try to use lmfit's guess to get values
            try:
                params = self.model.guess(y, x=x)
                for name, param in params.items():
                    if name in bounds:
                        bounds[name]["value"] = param.value
                        if param.min is not None and param.min > -np.inf:
                            bounds[name]["min"] = param.min
                        if param.max is not None and param.max < np.inf:
                            bounds[name]["max"] = param.max
            except (NotImplementedError, AttributeError):
                pass

        return bounds


def gaussian_1d() -> FitModel1D:
    """
    Create a 1D Gaussian model using peak height.

    Function: height * exp(-(x - center)² / (2 * sigma²))

    Parameters:
        height: Peak height (value at center)
        center: Peak center position
        sigma: Standard deviation (width parameter)

    The guess function constrains:
        - center to data x-range
        - sigma to at most the data x-extent
        - height estimated from data range
    """
    model = Model(_gaussian_height)

    def guess_gaussian(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        # Find peak location
        max_idx = int(np.nanargmax(y))
        center_guess = float(x[max_idx])
        height_guess = y_max - y_min

        # Estimate sigma from FWHM
        half_max = (y_max + y_min) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            fwhm = float(x[above_half[-1]] - x[above_half[0]])
            sigma_guess = fwhm / 2.355  # FWHM = 2.355 * sigma
        else:
            sigma_guess = x_extent / 6

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center": {"value": center_guess, "min": x_min, "max": x_max},
            "sigma": {"value": sigma_guess, "min": x_extent / 100, "max": x_extent / 2},
        }

    return FitModel1D(
        name="Gaussian",
        model=model,
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "center": {"value": 0.0},
            "sigma": {"min": 0, "value": 1.0},
        },
        description="Gaussian peak: height * exp(-(x-c)²/(2σ²))",
        custom_guess=guess_gaussian,
    )


def lorentzian_1d() -> FitModel1D:
    """
    Create a 1D Lorentzian model using peak height.

    Function: height / (1 + ((x - center) / sigma)²)

    Parameters:
        height: Peak height (value at center)
        center: Peak center position
        sigma: Half-width at half-maximum (HWHM)

    The guess function constrains:
        - center to data x-range
        - sigma to at most the data x-extent
        - height estimated from data range
    """
    model = Model(_lorentzian_height)

    def guess_lorentzian(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        # Find peak location
        max_idx = int(np.nanargmax(y))
        center_guess = float(x[max_idx])
        height_guess = y_max - y_min

        # Estimate sigma (HWHM) from data
        half_max = (y_max + y_min) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            fwhm = float(x[above_half[-1]] - x[above_half[0]])
            sigma_guess = fwhm / 2  # HWHM = FWHM / 2
        else:
            sigma_guess = x_extent / 6

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center": {"value": center_guess, "min": x_min, "max": x_max},
            "sigma": {"value": sigma_guess, "min": x_extent / 100, "max": x_extent / 2},
        }

    return FitModel1D(
        name="Lorentzian",
        model=model,
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "center": {"value": 0.0},
            "sigma": {"min": 0, "value": 1.0},
        },
        description="Lorentzian peak: height / (1 + ((x-c)/σ)²)",
        custom_guess=guess_lorentzian,
    )


def voigt_1d() -> FitModel1D:
    """
    Create a 1D Pseudo-Voigt model using peak height.

    The Pseudo-Voigt is a linear combination of Gaussian and Lorentzian
    profiles, controlled by a mixing parameter eta = gamma / (sigma + gamma).

    Function: height * ((1-eta) * Gaussian + eta * Lorentzian)

    Parameters:
        height: Peak height (value at center)
        center: Peak center position
        sigma: Gaussian width parameter
        gamma: Lorentzian width parameter (HWHM)

    The guess function constrains:
        - center to data x-range
        - sigma and gamma to at most the data x-extent
        - height estimated from data range
    """
    model = Model(_voigt_height)

    def guess_voigt(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        # Find peak location
        max_idx = int(np.nanargmax(y))
        center_guess = float(x[max_idx])
        height_guess = y_max - y_min

        # Estimate width from FWHM
        half_max = (y_max + y_min) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            fwhm = float(x[above_half[-1]] - x[above_half[0]])
            sigma_guess = fwhm / 2.355
            gamma_guess = fwhm / 2
        else:
            sigma_guess = x_extent / 6
            gamma_guess = x_extent / 6

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center": {"value": center_guess, "min": x_min, "max": x_max},
            "sigma": {"value": sigma_guess, "min": x_extent / 100, "max": x_extent / 2},
            "gamma": {"value": gamma_guess, "min": x_extent / 100, "max": x_extent / 2},
        }

    return FitModel1D(
        name="Voigt",
        model=model,
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "center": {"value": 0.0},
            "sigma": {"min": 0, "value": 1.0},
            "gamma": {"min": 0, "value": 1.0},
        },
        description="Pseudo-Voigt profile (Gaussian + Lorentzian mix)",
        custom_guess=guess_voigt,
    )


def linear_1d() -> FitModel1D:
    """
    Create a linear model: slope * x + intercept.

    Parameters:
        slope: Slope of the line (dy/dx)
        intercept: Y-intercept

    The guess function estimates:
        - slope from data gradient
        - intercept from mean y - slope * mean x
        - bounds based on data ranges
    """
    model = LinearModel()

    def guess_linear(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Estimate slope using linear regression
        x_mean = float(np.mean(x))
        y_mean = float(np.nanmean(y))
        numerator = float(np.sum((x - x_mean) * (y - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))
        slope_guess = numerator / denominator if denominator != 0 else 0.0

        # Intercept from y = mx + b => b = y_mean - m * x_mean
        intercept_guess = y_mean - slope_guess * x_mean

        # Max reasonable slope is when line spans entire y-range over x-range
        max_slope = y_extent / x_extent if x_extent > 0 else 10.0

        return {
            "slope": {"value": slope_guess, "min": -max_slope * 2, "max": max_slope * 2},
            "intercept": {
                "value": intercept_guess,
                "min": y_min - y_extent,
                "max": y_max + y_extent,
            },
        }

    return FitModel1D(
        name="Linear",
        model=model,
        param_hints={
            "slope": {"value": 1.0},
            "intercept": {"value": 0.0},
        },
        description="Linear: y = m*x + b",
        custom_guess=guess_linear,
    )


def polynomial_1d(degree: int = 2) -> FitModel1D:
    """
    Create a polynomial model of given degree.

    Parameters:
        c0, c1, c2, ...: Polynomial coefficients (c0 + c1*x + c2*x² + ...)

    The guess function estimates coefficients based on data range
    and polynomial degree to provide reasonable starting bounds.
    """
    model = PolynomialModel(degree=degree)
    param_hints = {f"c{i}": {"value": 0.0} for i in range(degree + 1)}

    def guess_polynomial(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_extent = y_max - y_min
        x_extent = max(abs(x_max), abs(x_min), 1.0)  # Avoid division by zero

        guesses = {}
        for i in range(degree + 1):
            # Scale coefficient bounds by x_extent^i to keep contributions reasonable
            scale = y_extent / (x_extent**i) if x_extent**i > 0 else y_extent
            guesses[f"c{i}"] = {"value": 0.0, "min": -scale * 10, "max": scale * 10}

        # Try to get better initial guess using polyfit
        try:
            coeffs = np.polyfit(x, y, degree)
            for i, c in enumerate(reversed(coeffs)):
                guesses[f"c{i}"]["value"] = float(c)
        except Exception:
            pass

        return guesses

    return FitModel1D(
        name=f"Polynomial (deg {degree})",
        model=model,
        param_hints=param_hints,
        description=f"Polynomial of degree {degree}",
        custom_guess=guess_polynomial,
    )


def exponential_1d() -> FitModel1D:
    """
    Create an exponential decay model: amplitude * exp(-x / decay).

    Parameters:
        amplitude: Initial amplitude at x=0
        decay: Time constant (larger = slower decay)

    The guess function estimates:
        - amplitude from y-data at start of x-range
        - decay from the x-extent (characteristic decay time)
        - bounds to ensure physically meaningful values
    """
    model = ExponentialModel()

    def guess_exponential(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Sort by x to find values at endpoints
        sort_idx = np.argsort(x)
        y_sorted = y[sort_idx]

        # Amplitude is typically the value at x_min (start)
        # For decay, we expect y to decrease
        amp_guess = float(y_sorted[0]) if len(y_sorted) > 0 else y_max

        # Estimate decay constant: if y decays to y/e at x = decay
        # We can estimate from ratio of end/start values
        y_start = float(np.mean(y_sorted[: max(1, len(y_sorted) // 10)]))
        y_end = float(np.mean(y_sorted[-max(1, len(y_sorted) // 10) :]))

        if y_start > 0 and y_end > 0 and y_start > y_end:
            # y_end/y_start = exp(-x_extent/decay) => decay = -x_extent/ln(y_end/y_start)
            ratio = y_end / y_start
            decay_guess = -x_extent / np.log(ratio) if 0 < ratio < 1 else x_extent / 3
        else:
            decay_guess = x_extent / 3

        decay_guess = max(x_extent / 100, decay_guess)  # Ensure positive

        return {
            "amplitude": {
                "value": amp_guess,
                "min": -y_extent * 3,
                "max": y_extent * 3,
            },
            "decay": {
                "value": decay_guess,
                "min": x_extent / 100,  # At least 1% of x-range
                "max": x_extent * 10,  # At most 10x the x-range
            },
        }

    return FitModel1D(
        name="Exponential Decay",
        model=model,
        param_hints={
            "amplitude": {"value": 1.0},
            "decay": {"min": 0, "value": 1.0},
        },
        description="Exponential decay: A * exp(-x / τ)",
        custom_guess=guess_exponential,
    )


def power_law_1d() -> FitModel1D:
    """
    Create a power law model: amplitude * x^exponent.

    Parameters:
        amplitude: Scaling factor
        exponent: Power law exponent

    The guess function estimates:
        - amplitude and exponent from log-log linear fit
        - bounds based on typical physical power law ranges
    """
    model = PowerLawModel()

    def guess_power_law(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        y_extent = y_max - y_min

        # Try log-log linear fit to estimate parameters
        # log(y) = log(A) + n*log(x) => linear fit gives slope=n, intercept=log(A)
        try:
            # Filter positive values for log
            mask = (x > 0) & (y > 0)
            if np.sum(mask) > 2:
                log_x = np.log(x[mask])
                log_y = np.log(y[mask])
                # Linear regression
                coeffs = np.polyfit(log_x, log_y, 1)
                exponent_guess = float(coeffs[0])
                amplitude_guess = float(np.exp(coeffs[1]))
            else:
                exponent_guess = 1.0
                amplitude_guess = y_max
        except Exception:
            exponent_guess = 1.0
            amplitude_guess = y_max

        return {
            "amplitude": {
                "value": amplitude_guess,
                "min": 0,
                "max": y_extent * 10,
            },
            "exponent": {
                "value": exponent_guess,
                "min": -10,  # Typical physical range
                "max": 10,
            },
        }

    return FitModel1D(
        name="Power Law",
        model=model,
        param_hints={
            "amplitude": {"value": 1.0},
            "exponent": {"value": 1.0},
        },
        description="Power law: A * x^n",
        custom_guess=guess_power_law,
    )


def sine_1d() -> FitModel1D:
    """
    Create a sine model: amplitude * sin(frequency * x + shift).

    Parameters:
        amplitude: Wave amplitude (peak-to-peak / 2)
        frequency: Angular frequency (rad/unit x)
        shift: Phase shift (radians)

    The guess function estimates:
        - amplitude from y-data range (should not exceed data extent)
        - frequency from zero crossings, bounded by Nyquist limit
        - shift within [-π, π]
    """
    model = SineModel()

    def guess_sine(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Amplitude: half the y-range
        amp_guess = y_extent / 2

        # Calculate Nyquist frequency limit based on sampling
        if len(x) > 1:
            dx = float(np.min(np.diff(np.sort(x))))
            nyquist_freq = np.pi / dx if dx > 0 else 100.0  # ω_max = π/Δx
        else:
            nyquist_freq = 100.0

        # Estimate frequency from zero crossings
        y_centered = y - np.mean(y)
        zero_crossings = np.where(np.diff(np.sign(y_centered)))[0]
        if len(zero_crossings) > 1:
            # Average period from zero crossings (half periods)
            avg_half_period = float(np.mean(np.diff(x[zero_crossings])))
            freq_guess = np.pi / avg_half_period if avg_half_period > 0 else 1.0
        else:
            # Default: one full cycle over the data range
            freq_guess = 2 * np.pi / x_extent if x_extent > 0 else 1.0

        # Minimum frequency: at least 1/4 cycle visible in data
        min_freq = np.pi / (2 * x_extent) if x_extent > 0 else 0.01

        return {
            "amplitude": {
                "value": amp_guess,
                "min": 0,
                "max": y_extent,  # Can't be larger than data range
            },
            "frequency": {
                "value": min(freq_guess, nyquist_freq * 0.8),  # Stay below Nyquist
                "min": min_freq,
                "max": nyquist_freq,  # Nyquist limit
            },
            "shift": {
                "value": 0.0,
                "min": -np.pi,
                "max": np.pi,
            },
        }

    return FitModel1D(
        name="Sine",
        model=model,
        param_hints={
            "amplitude": {"value": 1.0},
            "frequency": {"min": 0, "value": 1.0},
            "shift": {"value": 0.0},
        },
        description="Sine: A * sin(ωx + φ)",
        custom_guess=guess_sine,
    )


def damped_oscillation_1d() -> FitModel1D:
    """
    Create a damped oscillation model: A * exp(-decay * x) * sin(freq * x + phase).

    Parameters:
        amplitude: Initial oscillation amplitude
        decay: Exponential decay rate (1/time constant)
        frequency: Angular frequency (rad/unit x)
        phase: Initial phase (radians)

    The guess function estimates:
        - amplitude from envelope of oscillations
        - decay from envelope decay rate
        - frequency from zero crossings, bounded by Nyquist limit
        - phase within [-π, π]
    """

    def damped_osc(x, amplitude, decay, frequency, phase):
        return amplitude * np.exp(-decay * x) * np.sin(frequency * x + phase)

    model = Model(damped_osc)

    def guess_damped_osc(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min

        # Amplitude: half the y-range
        amp_guess = y_extent / 2

        # Calculate Nyquist frequency limit
        if len(x) > 1:
            dx = float(np.min(np.diff(np.sort(x))))
            nyquist_freq = np.pi / dx if dx > 0 else 100.0
        else:
            nyquist_freq = 100.0

        # Estimate frequency from zero crossings
        y_mean = float(np.nanmean(y))
        y_centered = y - y_mean
        zero_crossings = np.where(np.diff(np.sign(y_centered)))[0]
        if len(zero_crossings) > 1:
            avg_half_period = float(np.mean(np.diff(x[zero_crossings])))
            freq_guess = np.pi / avg_half_period if avg_half_period > 0 else 1.0
        else:
            freq_guess = 2 * np.pi / x_extent if x_extent > 0 else 1.0

        # Estimate decay from peak envelope
        peaks = (
            np.where(
                (np.abs(y_centered[1:-1]) > np.abs(y_centered[:-2]))
                & (np.abs(y_centered[1:-1]) > np.abs(y_centered[2:]))
            )[0]
            + 1
        )
        if len(peaks) > 1:
            peak_values = np.abs(y_centered[peaks])
            peak_x = x[peaks]
            # Fit log of peaks: log(A*exp(-γx)) = log(A) - γx
            try:
                coeffs = np.polyfit(peak_x, np.log(peak_values + 1e-10), 1)
                decay_guess = max(0.01, -float(coeffs[0]))
            except Exception:
                decay_guess = 1.0 / x_extent if x_extent > 0 else 0.1
        else:
            decay_guess = 1.0 / x_extent if x_extent > 0 else 0.1

        # Minimum frequency
        min_freq = np.pi / (2 * x_extent) if x_extent > 0 else 0.01

        return {
            "amplitude": {
                "value": amp_guess,
                "min": 0,
                "max": y_extent,  # Can't exceed data range
            },
            "decay": {
                "value": decay_guess,
                "min": 0,
                "max": 10.0 / x_extent if x_extent > 0 else 10.0,  # Reasonable decay rates
            },
            "frequency": {
                "value": min(freq_guess, nyquist_freq * 0.8),
                "min": min_freq,
                "max": nyquist_freq,  # Nyquist limit
            },
            "phase": {
                "value": 0.0,
                "min": -np.pi,
                "max": np.pi,
            },
        }

    return FitModel1D(
        name="Damped Oscillation",
        model=model,
        param_hints={
            "amplitude": {"value": 1.0},
            "decay": {"min": 0, "value": 0.1},
            "frequency": {"min": 0, "value": 1.0},
            "phase": {"value": 0.0},
        },
        description="Damped oscillation: A * exp(-γx) * sin(ωx + φ)",
        custom_guess=guess_damped_osc,
    )


def gaussian_with_offset_1d() -> FitModel1D:
    """
    Create a Gaussian model with constant offset using peak height.

    Function: height * exp(-(x - center)² / (2 * sigma²)) + offset

    Parameters:
        height: Peak height above offset
        center: Peak center position
        sigma: Standard deviation (width parameter)
        offset: Constant background offset

    The guess function constrains:
        - center to data x-range
        - sigma to at most the data x-extent
        - height estimated from data range
    """

    def gaussian_offset(
        x: NDArray, height: float, center: float, sigma: float, offset: float
    ) -> NDArray:
        return height * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset

    model = Model(gaussian_offset)

    def guess_gaussian_offset(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        # Find peak location
        max_idx = int(np.nanargmax(y))
        center_guess = float(x[max_idx])
        offset_guess = y_min
        height_guess = y_max - y_min

        # Estimate sigma from FWHM
        half_max = (y_max + y_min) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            fwhm = float(x[above_half[-1]] - x[above_half[0]])
            sigma_guess = fwhm / 2.355
        else:
            sigma_guess = x_extent / 6

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center": {"value": center_guess, "min": x_min, "max": x_max},
            "sigma": {"value": sigma_guess, "min": x_extent / 100, "max": x_extent / 2},
            "offset": {"value": offset_guess, "min": y_min - abs(y_max - y_min), "max": y_max},
        }

    return FitModel1D(
        name="Gaussian + Offset",
        model=model,
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "center": {"value": 0.0},
            "sigma": {"min": 0, "value": 1.0},
            "offset": {"value": 0.0},
        },
        description="Gaussian with offset: height * exp(-(x-c)²/(2σ²)) + c",
        custom_guess=guess_gaussian_offset,
    )


def lorentzian_with_offset_1d() -> FitModel1D:
    """
    Create a Lorentzian model with constant offset using peak height.

    Function: height / (1 + ((x - center) / sigma)²) + offset

    Parameters:
        height: Peak height above offset
        center: Peak center position
        sigma: Half-width at half-maximum (HWHM)
        offset: Constant background offset

    The guess function constrains:
        - center to data x-range
        - sigma to at most the data x-extent
        - height estimated from data range
    """

    def lorentzian_offset(
        x: NDArray, height: float, center: float, sigma: float, offset: float
    ) -> NDArray:
        return height / (1 + ((x - center) / sigma) ** 2) + offset

    model = Model(lorentzian_offset)

    def guess_lorentzian_offset(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

        # Find peak location
        max_idx = int(np.nanargmax(y))
        center_guess = float(x[max_idx])
        offset_guess = y_min
        height_guess = y_max - y_min

        # Estimate sigma from FWHM
        half_max = (y_max + y_min) / 2
        above_half = np.where(y > half_max)[0]
        if len(above_half) > 1:
            fwhm = float(x[above_half[-1]] - x[above_half[0]])
            sigma_guess = fwhm / 2
        else:
            sigma_guess = x_extent / 6

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center": {"value": center_guess, "min": x_min, "max": x_max},
            "sigma": {"value": sigma_guess, "min": x_extent / 100, "max": x_extent / 2},
            "offset": {"value": offset_guess, "min": y_min - abs(y_max - y_min), "max": y_max},
        }

    return FitModel1D(
        name="Lorentzian + Offset",
        model=model,
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "center": {"value": 0.0},
            "sigma": {"min": 0, "value": 1.0},
            "offset": {"value": 0.0},
        },
        description="Lorentzian with offset: height / (1 + ((x-c)/σ)²) + c",
        custom_guess=guess_lorentzian_offset,
    )


def double_gaussian_1d() -> FitModel1D:
    """
    Create a double Gaussian model (two peaks) using peak heights.

    Function: h1 * exp(-(x-c1)²/(2σ1²)) + h2 * exp(-(x-c2)²/(2σ2²))

    Parameters:
        height1, height2: Peak heights
        center1, center2: Peak center positions
        sigma1, sigma2: Standard deviations

    The guess function constrains:
        - centers to data x-range
        - sigmas to at most half the data x-extent
        - heights estimated from data range
    """

    def double_gaussian(
        x: NDArray,
        height1: float,
        center1: float,
        sigma1: float,
        height2: float,
        center2: float,
        sigma2: float,
    ) -> NDArray:
        g1 = height1 * np.exp(-((x - center1) ** 2) / (2 * sigma1**2))
        g2 = height2 * np.exp(-((x - center2) ** 2) / (2 * sigma2**2))
        return g1 + g2

    model = Model(double_gaussian)

    def guess_double_gaussian(y: NDArray, x: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        x_extent = x_max - x_min
        y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        x_mid = (x_min + x_max) / 2

        # Find the highest peak
        max_idx = int(np.nanargmax(y))
        center1_guess = float(x[max_idx])
        height_guess = y_max - y_min

        # Place second peak on opposite side of center
        center2_guess = x_min + x_extent / 4 if center1_guess > x_mid else x_max - x_extent / 4

        return {
            "height1": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "center1": {"value": center1_guess, "min": x_min, "max": x_max},
            "sigma1": {"value": x_extent / 8, "min": x_extent / 100, "max": x_extent / 2},
            "height2": {"value": height_guess / 2, "min": 0, "max": height_guess * 3},
            "center2": {"value": center2_guess, "min": x_min, "max": x_max},
            "sigma2": {"value": x_extent / 8, "min": x_extent / 100, "max": x_extent / 2},
        }

    return FitModel1D(
        name="Double Gaussian",
        model=model,
        param_hints={
            "height1": {"min": 0, "value": 1.0},
            "center1": {"value": -1.0},
            "sigma1": {"min": 0, "value": 0.5},
            "height2": {"min": 0, "value": 1.0},
            "center2": {"value": 1.0},
            "sigma2": {"min": 0, "value": 0.5},
        },
        description="Double Gaussian (two peaks with height parameters)",
        custom_guess=guess_double_gaussian,
    )


# Dictionary of all available 1D models
MODELS_1D: dict[str, Callable[[], FitModel1D]] = {
    "Gaussian": gaussian_1d,
    "Lorentzian": lorentzian_1d,
    "Voigt": voigt_1d,
    "Linear": linear_1d,
    "Quadratic": lambda: polynomial_1d(2),
    "Cubic": lambda: polynomial_1d(3),
    "Exponential Decay": exponential_1d,
    "Power Law": power_law_1d,
    "Sine": sine_1d,
    "Damped Oscillation": damped_oscillation_1d,
    "Gaussian + Offset": gaussian_with_offset_1d,
    "Lorentzian + Offset": lorentzian_with_offset_1d,
    "Double Gaussian": double_gaussian_1d,
}


# =============================================================================
# 2D Fitting Models for InteractiveHeatmap
# =============================================================================


@dataclass
class FitModel2D:
    """
    Container for a 2D fit model with parameter information.

    Attributes:
        name: Display name of the model
        func: 2D function (X, Y, **params) -> Z
        param_names: List of parameter names
        param_hints: Dictionary of parameter hints (min, max, value, etc.)
        description: Short description of the model
        custom_guess: Optional custom guess function with signature (data, X, Y) -> dict

    Note:
        Peak models (Gaussian, Lorentzian) use 'height' as the peak amplitude
        rather than area, making interactive fitting more intuitive.
    """

    name: str
    func: Callable[..., NDArray]
    param_names: list[str]
    param_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    description: str = ""
    custom_guess: Callable[[NDArray, NDArray, NDArray], dict[str, dict[str, Any]]] | None = None

    def make_model(self) -> Model:
        """Create an lmfit Model from the function."""
        return Model(self.func, independent_vars=["X", "Y"])

    def guess(self, data: NDArray, X: NDArray, Y: NDArray) -> dict[str, float]:
        """
        Make an initial guess for the model parameters.

        Args:
            data: 2D data array
            X: 2D X coordinate array (from meshgrid)
            Y: 2D Y coordinate array (from meshgrid)

        Returns:
            Dictionary of initial parameter values
        """
        # If custom guess function is provided, use it
        if self.custom_guess is not None:
            try:
                guess_dict = self.custom_guess(data, X, Y)
                guesses = {}
                for name, hints in guess_dict.items():
                    if isinstance(hints, dict):
                        guesses[name] = hints.get("value", 1.0)
                    else:
                        guesses[name] = hints
                return guesses
            except Exception:
                pass

        guesses = {}
        for name, hints in self.param_hints.items():
            guesses[name] = hints.get("value", 1.0)

        # Try to make intelligent guesses based on data
        if "height" in self.param_names:
            guesses["height"] = float(np.nanmax(data) - np.nanmin(data))

        if "offset" in self.param_names:
            guesses["offset"] = float(np.nanmin(data))

        if "x0" in self.param_names:
            # Find location of maximum
            max_idx = np.unravel_index(np.nanargmax(data), data.shape)
            guesses["x0"] = float(X[max_idx])

        if "y0" in self.param_names:
            max_idx = np.unravel_index(np.nanargmax(data), data.shape)
            guesses["y0"] = float(Y[max_idx])

        if "sigma_x" in self.param_names:
            guesses["sigma_x"] = float(np.abs(X.max() - X.min()) / 6)

        if "sigma_y" in self.param_names:
            guesses["sigma_y"] = float(np.abs(Y.max() - Y.min()) / 6)

        if "gamma_x" in self.param_names:
            guesses["gamma_x"] = float(np.abs(X.max() - X.min()) / 6)

        if "gamma_y" in self.param_names:
            guesses["gamma_y"] = float(np.abs(Y.max() - Y.min()) / 6)

        return guesses

    def get_param_bounds(self, data: NDArray, X: NDArray, Y: NDArray) -> dict[str, dict[str, Any]]:
        """
        Get parameter bounds (min, max, value) for slider initialization.

        Args:
            data: 2D data array
            X: 2D X coordinate array (from meshgrid)
            Y: 2D Y coordinate array (from meshgrid)

        Returns:
            Dictionary mapping parameter names to dicts with 'value', 'min', 'max'
        """
        bounds = {}

        # Start with param_hints as defaults
        for name in self.param_names:
            hints = self.param_hints.get(name, {})
            bounds[name] = {
                "value": hints.get("value", 1.0),
                "min": hints.get("min", -np.inf),
                "max": hints.get("max", np.inf),
            }

        # If custom guess function is provided, use it to update bounds
        if self.custom_guess is not None:
            try:
                guess_dict = self.custom_guess(data, X, Y)
                for name, hints in guess_dict.items():
                    if name in bounds and isinstance(hints, dict):
                        for key in ["value", "min", "max"]:
                            if key in hints:
                                bounds[name][key] = hints[key]
            except Exception:
                pass
        else:
            # Use default guess to get values
            guesses = self.guess(data, X, Y)
            for name, value in guesses.items():
                if name in bounds:
                    bounds[name]["value"] = value

        return bounds


def gaussian_2d_func(
    X: NDArray,
    Y: NDArray,
    height: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float = 0.0,
) -> NDArray:
    """
    2D Gaussian function using peak height.

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        sigma_x, sigma_y: Standard deviations in x and y
        offset: Constant background offset

    Returns:
        height * exp(-((X-x0)²/(2σx²) + (Y-y0)²/(2σy²))) + offset
    """
    return (
        height * np.exp(-((X - x0) ** 2 / (2 * sigma_x**2) + (Y - y0) ** 2 / (2 * sigma_y**2)))
        + offset
    )


def gaussian_2d() -> FitModel2D:
    """
    Create a 2D Gaussian model using peak height.

    Parameters:
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        sigma_x, sigma_y: Standard deviations
        offset: Constant background

    The guess function constrains:
        - x0 to data x-range, y0 to data y-range
        - sigma_x/sigma_y to at most half the respective data extent
        - height estimated from data range
    """

    def guess_gaussian_2d(data: NDArray, X: NDArray, Y: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))

        # Find peak location
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        x0_guess = float(X[max_idx])
        y0_guess = float(Y[max_idx])
        height_guess = z_max - z_min
        offset_guess = z_min

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "x0": {"value": x0_guess, "min": x_min, "max": x_max},
            "y0": {"value": y0_guess, "min": y_min, "max": y_max},
            "sigma_x": {"value": x_extent / 6, "min": x_extent / 100, "max": x_extent / 2},
            "sigma_y": {"value": y_extent / 6, "min": y_extent / 100, "max": y_extent / 2},
            "offset": {"value": offset_guess, "min": z_min - height_guess, "max": z_max},
        }

    return FitModel2D(
        name="2D Gaussian",
        func=gaussian_2d_func,
        param_names=["height", "x0", "y0", "sigma_x", "sigma_y", "offset"],
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "x0": {"value": 0.0},
            "y0": {"value": 0.0},
            "sigma_x": {"min": 0, "value": 1.0},
            "sigma_y": {"min": 0, "value": 1.0},
            "offset": {"value": 0.0},
        },
        description="2D Gaussian: height * exp(-((x-x₀)²/2σx² + (y-y₀)²/2σy²)) + c",
        custom_guess=guess_gaussian_2d,
    )


def rotated_gaussian_2d_func(
    X: NDArray,
    Y: NDArray,
    height: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float = 0.0,
) -> NDArray:
    """
    2D Rotated Gaussian function using peak height.

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        sigma_x, sigma_y: Standard deviations along principal axes
        theta: Rotation angle in radians
        offset: Constant background offset
    """
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)
    return (
        height * np.exp(-(a * (X - x0) ** 2 + 2 * b * (X - x0) * (Y - y0) + c * (Y - y0) ** 2))
        + offset
    )


def rotated_gaussian_2d() -> FitModel2D:
    """
    Create a rotated 2D Gaussian model using peak height.

    Parameters:
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        sigma_x, sigma_y: Standard deviations along principal axes
        theta: Rotation angle in radians
        offset: Constant background

    The guess function constrains:
        - x0 to data x-range, y0 to data y-range
        - sigma_x/sigma_y to at most half the respective data extent
        - theta to [-π, π]
    """

    def guess_rotated_gaussian_2d(
        data: NDArray, X: NDArray, Y: NDArray
    ) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))

        # Find peak location
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        x0_guess = float(X[max_idx])
        y0_guess = float(Y[max_idx])
        height_guess = z_max - z_min
        offset_guess = z_min

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "x0": {"value": x0_guess, "min": x_min, "max": x_max},
            "y0": {"value": y0_guess, "min": y_min, "max": y_max},
            "sigma_x": {"value": x_extent / 6, "min": x_extent / 100, "max": x_extent / 2},
            "sigma_y": {"value": y_extent / 6, "min": y_extent / 100, "max": y_extent / 2},
            "theta": {"value": 0.0, "min": -np.pi, "max": np.pi},
            "offset": {"value": offset_guess, "min": z_min - height_guess, "max": z_max},
        }

    return FitModel2D(
        name="Rotated 2D Gaussian",
        func=rotated_gaussian_2d_func,
        param_names=["height", "x0", "y0", "sigma_x", "sigma_y", "theta", "offset"],
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "x0": {"value": 0.0},
            "y0": {"value": 0.0},
            "sigma_x": {"min": 0, "value": 1.0},
            "sigma_y": {"min": 0, "value": 1.0},
            "theta": {"min": -np.pi, "max": np.pi, "value": 0.0},
            "offset": {"value": 0.0},
        },
        description="Rotated 2D Gaussian (height parameter, rotation angle θ)",
        custom_guess=guess_rotated_gaussian_2d,
    )


def lorentzian_2d_func(
    X: NDArray,
    Y: NDArray,
    height: float,
    x0: float,
    y0: float,
    gamma_x: float,
    gamma_y: float,
    offset: float = 0.0,
) -> NDArray:
    """
    2D Lorentzian function using peak height.

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        gamma_x, gamma_y: Half-widths at half-maximum
        offset: Constant background offset

    Returns:
        height / (1 + ((X-x0)/γx)² + ((Y-y0)/γy)²) + offset
    """
    return height / (1 + ((X - x0) / gamma_x) ** 2 + ((Y - y0) / gamma_y) ** 2) + offset


def lorentzian_2d() -> FitModel2D:
    """
    Create a 2D Lorentzian model using peak height.

    Parameters:
        height: Peak height (value at center above offset)
        x0, y0: Peak center position
        gamma_x, gamma_y: Half-widths at half-maximum
        offset: Constant background

    The guess function constrains:
        - x0 to data x-range, y0 to data y-range
        - gamma_x/gamma_y to at most half the respective data extent
    """

    def guess_lorentzian_2d(data: NDArray, X: NDArray, Y: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))

        # Find peak location
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        x0_guess = float(X[max_idx])
        y0_guess = float(Y[max_idx])
        height_guess = z_max - z_min
        offset_guess = z_min

        return {
            "height": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "x0": {"value": x0_guess, "min": x_min, "max": x_max},
            "y0": {"value": y0_guess, "min": y_min, "max": y_max},
            "gamma_x": {"value": x_extent / 6, "min": x_extent / 100, "max": x_extent / 2},
            "gamma_y": {"value": y_extent / 6, "min": y_extent / 100, "max": y_extent / 2},
            "offset": {"value": offset_guess, "min": z_min - height_guess, "max": z_max},
        }

    return FitModel2D(
        name="2D Lorentzian",
        func=lorentzian_2d_func,
        param_names=["height", "x0", "y0", "gamma_x", "gamma_y", "offset"],
        param_hints={
            "height": {"min": 0, "value": 1.0},
            "x0": {"value": 0.0},
            "y0": {"value": 0.0},
            "gamma_x": {"min": 0, "value": 1.0},
            "gamma_y": {"min": 0, "value": 1.0},
            "offset": {"value": 0.0},
        },
        description="2D Lorentzian: height / (1 + ((x-x₀)/γx)² + ((y-y₀)/γy)²) + c",
        custom_guess=guess_lorentzian_2d,
    )


def double_gaussian_2d_func(
    X: NDArray,
    Y: NDArray,
    height1: float,
    x01: float,
    y01: float,
    sigma_x1: float,
    sigma_y1: float,
    height2: float,
    x02: float,
    y02: float,
    sigma_x2: float,
    sigma_y2: float,
    offset: float = 0.0,
) -> NDArray:
    """
    Double 2D Gaussian function using peak heights.

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        height1, height2: Peak heights for each Gaussian
        x01, y01, x02, y02: Peak center positions
        sigma_x1, sigma_y1, sigma_x2, sigma_y2: Standard deviations
        offset: Constant background offset
    """
    g1 = height1 * np.exp(
        -((X - x01) ** 2 / (2 * sigma_x1**2) + (Y - y01) ** 2 / (2 * sigma_y1**2))
    )
    g2 = height2 * np.exp(
        -((X - x02) ** 2 / (2 * sigma_x2**2) + (Y - y02) ** 2 / (2 * sigma_y2**2))
    )
    return g1 + g2 + offset


def double_gaussian_2d() -> FitModel2D:
    """
    Create a double 2D Gaussian model (two peaks) using peak heights.

    Parameters:
        height1, height2: Peak heights
        x01, y01, x02, y02: Peak center positions
        sigma_x1, sigma_y1, sigma_x2, sigma_y2: Standard deviations
        offset: Constant background

    The guess function constrains:
        - centers to data ranges
        - sigmas to at most half the data extent
    """

    def guess_double_gaussian_2d(
        data: NDArray, X: NDArray, Y: NDArray
    ) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        x_extent = x_max - x_min
        y_extent = y_max - y_min
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))

        # Find first peak location
        max_idx = np.unravel_index(np.nanargmax(data), data.shape)
        x01_guess = float(X[max_idx])
        y01_guess = float(Y[max_idx])
        height_guess = z_max - z_min
        offset_guess = z_min

        # Place second peak on opposite side
        x02_guess = x_min + x_extent / 4 if x01_guess > x_mid else x_max - x_extent / 4
        y02_guess = y_min + y_extent / 4 if y01_guess > y_mid else y_max - y_extent / 4

        return {
            "height1": {"value": height_guess, "min": 0, "max": height_guess * 3},
            "x01": {"value": x01_guess, "min": x_min, "max": x_max},
            "y01": {"value": y01_guess, "min": y_min, "max": y_max},
            "sigma_x1": {"value": x_extent / 8, "min": x_extent / 100, "max": x_extent / 2},
            "sigma_y1": {"value": y_extent / 8, "min": y_extent / 100, "max": y_extent / 2},
            "height2": {"value": height_guess / 2, "min": 0, "max": height_guess * 3},
            "x02": {"value": x02_guess, "min": x_min, "max": x_max},
            "y02": {"value": y02_guess, "min": y_min, "max": y_max},
            "sigma_x2": {"value": x_extent / 8, "min": x_extent / 100, "max": x_extent / 2},
            "sigma_y2": {"value": y_extent / 8, "min": y_extent / 100, "max": y_extent / 2},
            "offset": {"value": offset_guess, "min": z_min - height_guess, "max": z_max},
        }

    return FitModel2D(
        name="Double 2D Gaussian",
        func=double_gaussian_2d_func,
        param_names=[
            "height1",
            "x01",
            "y01",
            "sigma_x1",
            "sigma_y1",
            "height2",
            "x02",
            "y02",
            "sigma_x2",
            "sigma_y2",
            "offset",
        ],
        param_hints={
            "height1": {"min": 0, "value": 1.0},
            "x01": {"value": -1.0},
            "y01": {"value": 0.0},
            "sigma_x1": {"min": 0, "value": 0.5},
            "sigma_y1": {"min": 0, "value": 0.5},
            "height2": {"min": 0, "value": 1.0},
            "x02": {"value": 1.0},
            "y02": {"value": 0.0},
            "sigma_x2": {"min": 0, "value": 0.5},
            "sigma_y2": {"min": 0, "value": 0.5},
            "offset": {"value": 0.0},
        },
        description="Double 2D Gaussian (two peaks with height parameters)",
        custom_guess=guess_double_gaussian_2d,
    )


def plane_func(
    X: NDArray,
    Y: NDArray,
    a: float,
    b: float,
    c: float,
) -> NDArray:
    """
    2D plane function: z = a*x + b*y + c

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        a: Slope in x direction
        b: Slope in y direction
        c: Constant offset
    """
    return a * X + b * Y + c


def plane_2d() -> FitModel2D:
    """
    Create a 2D plane model.

    Parameters:
        a: Slope in x direction
        b: Slope in y direction
        c: Constant offset

    The guess function estimates slopes from data gradients.
    """

    def guess_plane_2d(data: NDArray, X: NDArray, Y: NDArray) -> dict[str, dict[str, Any]]:
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))
        z_range = z_max - z_min

        # Estimate slopes from data
        try:
            grad_y, grad_x = np.gradient(data)
            a_guess = float(np.nanmean(grad_x))
            b_guess = float(np.nanmean(grad_y))
        except Exception:
            a_guess = 0.0
            b_guess = 0.0

        c_guess = float(np.nanmean(data))

        return {
            "a": {"value": a_guess, "min": -z_range, "max": z_range},
            "b": {"value": b_guess, "min": -z_range, "max": z_range},
            "c": {"value": c_guess, "min": z_min - z_range, "max": z_max + z_range},
        }

    return FitModel2D(
        name="Plane",
        func=plane_func,
        param_names=["a", "b", "c"],
        param_hints={
            "a": {"value": 0.0},
            "b": {"value": 0.0},
            "c": {"value": 0.0},
        },
        description="Plane: z = ax + by + c",
        custom_guess=guess_plane_2d,
    )


def paraboloid_func(
    X: NDArray,
    Y: NDArray,
    amplitude: float,
    x0: float,
    y0: float,
    a: float,
    b: float,
    offset: float = 0.0,
) -> NDArray:
    """
    2D paraboloid function.

    Args:
        X, Y: 2D coordinate arrays (from meshgrid)
        amplitude: Overall scaling factor
        x0, y0: Center position (vertex)
        a, b: Curvature coefficients for x and y
        offset: Constant offset at center

    Returns:
        amplitude * (a*(X-x0)² + b*(Y-y0)²) + offset
    """
    return amplitude * (a * (X - x0) ** 2 + b * (Y - y0) ** 2) + offset


def paraboloid_2d() -> FitModel2D:
    """
    Create a 2D paraboloid model.

    Parameters:
        amplitude: Overall scaling factor
        x0, y0: Center position (vertex)
        a, b: Curvature coefficients
        offset: Constant offset at center

    The guess function constrains:
        - x0 to data x-range, y0 to data y-range
    """

    def guess_paraboloid_2d(data: NDArray, X: NDArray, Y: NDArray) -> dict[str, dict[str, Any]]:
        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        z_min, z_max = float(np.nanmin(data)), float(np.nanmax(data))

        # Find minimum location (vertex of paraboloid)
        min_idx = np.unravel_index(np.nanargmin(data), data.shape)
        x0_guess = float(X[min_idx])
        y0_guess = float(Y[min_idx])
        offset_guess = z_min

        return {
            "amplitude": {"value": 1.0, "min": -10, "max": 10},
            "x0": {"value": x0_guess, "min": x_min, "max": x_max},
            "y0": {"value": y0_guess, "min": y_min, "max": y_max},
            "a": {"value": 1.0, "min": 0, "max": 10},
            "b": {"value": 1.0, "min": 0, "max": 10},
            "offset": {"value": offset_guess, "min": z_min - abs(z_max - z_min), "max": z_max},
        }

    return FitModel2D(
        name="Paraboloid",
        func=paraboloid_func,
        param_names=["amplitude", "x0", "y0", "a", "b", "offset"],
        param_hints={
            "amplitude": {"value": 1.0},
            "x0": {"value": 0.0},
            "y0": {"value": 0.0},
            "a": {"min": 0, "value": 1.0},
            "b": {"min": 0, "value": 1.0},
            "offset": {"value": 0.0},
        },
        description="Paraboloid: A * (a(x-x₀)² + b(y-y₀)²) + c",
        custom_guess=guess_paraboloid_2d,
    )


# Dictionary of all available 2D models
MODELS_2D: dict[str, Callable[[], FitModel2D]] = {
    "2D Gaussian": gaussian_2d,
    "Rotated 2D Gaussian": rotated_gaussian_2d,
    "2D Lorentzian": lorentzian_2d,
    "Double 2D Gaussian": double_gaussian_2d,
    "Plane": plane_2d,
    "Paraboloid": paraboloid_2d,
}


# =============================================================================
# Fitting Helper Functions
# =============================================================================


def fit_1d(
    x: NDArray,
    y: NDArray,
    model: FitModel1D,
    params: Parameters | None = None,
    weights: NDArray | None = None,
) -> Any:
    """
    Perform 1D curve fitting.

    Args:
        x: X data array
        y: Y data array
        model: FitModel1D object
        params: Optional lmfit Parameters (uses model.guess if None)
        weights: Optional weights for the fit (1/sigma)

    Returns:
        lmfit ModelResult object
    """
    if params is None:
        params = model.guess(y, x)

    # Apply parameter hints
    for name, hints in model.param_hints.items():
        if name in params:
            for key, val in hints.items():
                if key != "value":  # Don't overwrite guessed values
                    setattr(params[name], key, val)

    return model.model.fit(y, params, x=x, weights=weights)


def fit_2d(
    X: NDArray,
    Y: NDArray,
    data: NDArray,
    model: FitModel2D,
    params: dict[str, float] | None = None,
    weights: NDArray | None = None,
) -> Any:
    """
    Perform 2D surface fitting.

    Args:
        X: 2D X coordinate array (from meshgrid)
        Y: 2D Y coordinate array (from meshgrid)
        data: 2D data array
        model: FitModel2D object
        params: Optional dictionary of initial parameter values
        weights: Optional weights for the fit (1/sigma)

    Returns:
        lmfit ModelResult object
    """
    if params is None:
        params = model.guess(data, X, Y)

    lmfit_model = model.make_model()
    lmfit_params = lmfit_model.make_params(**params)

    # Apply parameter hints
    for name, hints in model.param_hints.items():
        if name in lmfit_params:
            for key, val in hints.items():
                setattr(lmfit_params[name], key, val)
            # Set value from params dict
            if name in params:
                lmfit_params[name].value = params[name]

    # Flatten arrays for fitting
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    data_flat = data.ravel()
    weights_flat = weights.ravel() if weights is not None else None

    result = lmfit_model.fit(data_flat, lmfit_params, X=X_flat, Y=Y_flat, weights=weights_flat)

    return result


def create_custom_model_1d(
    func: Callable,
    param_hints: dict[str, dict[str, Any]] | None = None,
    name: str = "Custom",
    description: str = "Custom 1D model",
    guess_func: Callable[[NDArray, NDArray], dict[str, dict[str, Any]]] | None = None,
) -> FitModel1D:
    """
    Create a custom 1D fit model from a user-defined function.

    Args:
        func: Function with signature f(x, param1, param2, ...) -> y
        param_hints: Dictionary of parameter hints (min, max, value for each param)
        name: Display name for the model
        description: Short description
        guess_func: Optional custom guess function with signature (y, x) -> dict.
            The returned dict maps parameter names to either:
            - A single value (float)
            - A dict with keys 'value', 'min', 'max'

    Returns:
        FitModel1D object
    """
    model = Model(func)
    param_names = list(model.param_names)

    if param_hints is None:
        param_hints = {p: {"value": 1.0} for p in param_names}

    return FitModel1D(
        name=name,
        model=model,
        param_hints=param_hints,
        description=description,
        custom_guess=guess_func,
    )


def create_custom_model_2d(
    func: Callable,
    param_names: list[str],
    param_hints: dict[str, dict[str, Any]] | None = None,
    name: str = "Custom",
    description: str = "Custom 2D model",
    guess_func: Callable[[NDArray, NDArray, NDArray], dict[str, dict[str, Any]]] | None = None,
) -> FitModel2D:
    """
    Create a custom 2D fit model from a user-defined function.

    Args:
        func: Function with signature f(X, Y, param1, param2, ...) -> Z
        param_names: List of parameter names (excluding X, Y)
        param_hints: Dictionary of parameter hints (min, max, value for each param)
        name: Display name for the model
        description: Short description
        guess_func: Optional custom guess function with signature (data, X, Y) -> dict.
            The returned dict maps parameter names to either:
            - A single value (float)
            - A dict with keys 'value', 'min', 'max'

    Returns:
        FitModel2D object
    """
    if param_hints is None:
        param_hints = {p: {"value": 1.0} for p in param_names}

    return FitModel2D(
        name=name,
        func=func,
        param_names=param_names,
        param_hints=param_hints,
        description=description,
        custom_guess=guess_func,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # 1D Models
    "FitModel1D",
    "MODELS_1D",
    "gaussian_1d",
    "lorentzian_1d",
    "voigt_1d",
    "linear_1d",
    "polynomial_1d",
    "exponential_1d",
    "power_law_1d",
    "sine_1d",
    "damped_oscillation_1d",
    "gaussian_with_offset_1d",
    "lorentzian_with_offset_1d",
    "double_gaussian_1d",
    # 2D Models
    "FitModel2D",
    "MODELS_2D",
    "gaussian_2d",
    "rotated_gaussian_2d",
    "lorentzian_2d",
    "double_gaussian_2d",
    "plane_2d",
    "paraboloid_2d",
    # Functions
    "fit_1d",
    "fit_2d",
    "create_custom_model_1d",
    "create_custom_model_2d",
]
