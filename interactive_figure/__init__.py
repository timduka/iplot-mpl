"""
Interactive Figure Package for Jupyter Notebooks

This module provides a way to create interactive matplotlib figures with ipywidgets.
Users define a plot function with typed parameters, and the module automatically
creates appropriate widgets based on the parameter annotations.
"""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    MODELS_1D,
    MODELS_2D,
    BoundedFloatText,
    BoundedIntText,
    Checkbox,
    Dropdown,
    # Fitting
    FitModel1D,
    FitModel2D,
    FloatLogSlider,
    FloatRangeSlider,
    FloatSlider,
    FloatText,
    InteractiveHeatmap,
    # Interactive Figure Classes
    InteractiveXYPlot,
    IntRangeSlider,
    # Widget configurations
    IntSlider,
    IntText,
    RadioButtons,
    Select,
    SelectionSlider,
    SelectMultiple,
    Text,
    Textarea,
    ToggleButton,
    ToggleButtons,
    create_custom_model_1d,
    create_custom_model_2d,
    create_widget,
    fit_1d,
    fit_2d,
    # Functions
    interactive_plot,
    interactive_plotting,
)

try:
    __version__ = version("interactive-figure")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback for development

__all__ = [
    # Widget configurations
    'IntSlider',
    'FloatSlider',
    'IntRangeSlider',
    'FloatRangeSlider',
    'FloatLogSlider',
    'IntText',
    'FloatText',
    'BoundedIntText',
    'BoundedFloatText',
    'Checkbox',
    'ToggleButton',
    'Dropdown',
    'RadioButtons',
    'Select',
    'SelectMultiple',
    'ToggleButtons',
    'SelectionSlider',
    'Text',
    'Textarea',
    # Functions
    'interactive_plot',
    'interactive_plotting',
    'create_widget',
    # Interactive Figure Classes
    'InteractiveXYPlot',
    'InteractiveHeatmap',
    # Fitting
    'FitModel1D',
    'FitModel2D',
    'MODELS_1D',
    'MODELS_2D',
    'create_custom_model_1d',
    'create_custom_model_2d',
    'fit_1d',
    'fit_2d',
]
