# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-20

### Fixed

- Updated README with correct installation instructions

## [0.2.0] - 2026-02-20

### Added

- `InteractiveXYPlot` class with built-in X/Y range sliders, grid/legend toggles, and save functionality
- `InteractiveHeatmap` class with colormap selection, color scale control, and 2D visualization
- Built-in curve fitting support using lmfit:
  - 1D models: Gaussian, Lorentzian, Voigt, Linear, Polynomial, Exponential Decay, Sine
  - 2D models: 2D Gaussian, 2D Lorentzian, Plane
- Intelligent parameter guessing with sensible bounds for all built-in models
- Custom fit model support via `create_custom_model_1d` and `create_custom_model_2d`
- Custom guess functions for automatic parameter estimation
- Peak functions now use intuitive `height` parameter instead of `amplitude`
- Fit appearance customization:
  - Show/hide fit in legend
  - Color selection dropdown
  - Line style selection (solid, dashed, dotted, dash-dot)
  - Programmatic control via `fit_color` and `fit_linestyle` parameters
- Fit contour customization for heatmaps via `fit_contour_color` and `fit_contour_linestyle`
- Model name displayed in fit result output
- Parameter bounds with min/max textboxes supporting infinity
- Fix parameter toggle to lock parameters during fitting
- Reset guesses button to restore automatic parameter estimates

### Changed

- Improved slider range handling with Nyquist frequency limits
- Better automatic bounds based on data extent

## [0.1.0] - 2026-02-19

### Added

- Initial release
- `interactive_plot` function for creating interactive matplotlib figures
- Widget configuration classes:
  - Numeric sliders: `IntSlider`, `FloatSlider`, `IntRangeSlider`, `FloatRangeSlider`, `FloatLogSlider`
  - Text inputs: `IntText`, `FloatText`, `BoundedIntText`, `BoundedFloatText`
  - Boolean: `Checkbox`, `ToggleButton`
  - Selection: `Dropdown`, `RadioButtons`, `Select`, `SelectMultiple`, `ToggleButtons`, `SelectionSlider`
  - String: `Text`, `Textarea`
- Automatic widget inference from Python type annotations
- Support for vertical, horizontal, and grid widget layouts
- Type hints and `py.typed` marker for PEP 561 compliance

[Unreleased]: https://github.com/timduka/iplot-mpl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/timduka/iplot-mpl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/timduka/iplot-mpl/releases/tag/v0.1.0
