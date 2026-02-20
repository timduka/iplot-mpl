"""Interactive Figure Package for Jupyter Notebooks.

This module provides a way to create interactive matplotlib figures with ipywidgets.
Users define a plot function with typed parameters, and the module automatically
creates appropriate widgets based on the parameter annotations.

Main Components:
    - Widget configuration classes (IntSlider, FloatSlider, Dropdown, etc.)
    - interactive_plot(): Function-based interactive plotting
    - InteractiveXYPlot: Class for interactive line/scatter plots with fitting
    - InteractiveHeatmap: Class for interactive 2D heatmap visualization with fitting

Example:
    >>> from interactive_figure import InteractiveXYPlot
    >>> import numpy as np
    >>> fig = InteractiveXYPlot(figsize=(10, 6), title="My Plot")
    >>> x = np.linspace(0, 10, 100)
    >>> fig.plot(x, np.sin(x), label="sin(x)")
    >>> fig.show()
"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

import ipywidgets as widgets
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

# Import fitting module
from .fitting import (
    MODELS_1D,
    MODELS_2D,
    FitModel1D,
    FitModel2D,
    create_custom_model_1d,
    create_custom_model_2d,
    fit_1d,
    fit_2d,
)

# =============================================================================
# Widget Configuration Classes
# =============================================================================

# --- Numeric Sliders ---


@dataclass
class IntSlider:
    """Configuration for an integer slider widget."""

    min: int = 0
    max: int = 100
    step: int = 1
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True


@dataclass
class FloatSlider:
    """Configuration for a float slider widget."""

    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True
    readout_format: str = ".2f"


@dataclass
class IntRangeSlider:
    """Configuration for an integer range slider widget."""

    min: int = 0
    max: int = 100
    step: int = 1
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True


@dataclass
class FloatRangeSlider:
    """Configuration for a float range slider widget."""

    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True
    readout_format: str = ".2f"


@dataclass
class FloatLogSlider:
    """Configuration for a logarithmic float slider widget."""

    min: float = -2  # 10^min
    max: float = 2  # 10^max
    step: float = 0.1
    base: float = 10.0
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True
    readout_format: str = ".3e"


# --- Numeric Text Inputs ---


@dataclass
class IntText:
    """Configuration for an integer text input widget."""

    description: str = ""
    continuous_update: bool = False


@dataclass
class FloatText:
    """Configuration for a float text input widget."""

    description: str = ""
    continuous_update: bool = False


@dataclass
class BoundedIntText:
    """Configuration for a bounded integer text input widget."""

    min: int = 0
    max: int = 100
    step: int = 1
    description: str = ""
    continuous_update: bool = False


@dataclass
class BoundedFloatText:
    """Configuration for a bounded float text input widget."""

    min: float = 0.0
    max: float = 1.0
    step: float = 0.01
    description: str = ""
    continuous_update: bool = False


# --- Boolean Widgets ---


@dataclass
class Checkbox:
    """Configuration for a checkbox widget."""

    description: str = ""
    indent: bool = True


@dataclass
class ToggleButton:
    """Configuration for a toggle button widget."""

    description: str = ""
    button_style: str = ""  # 'success', 'info', 'warning', 'danger' or ''
    icon: str = "check"


# --- Selection Widgets ---


@dataclass
class Dropdown:
    """Configuration for a dropdown selection widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""


@dataclass
class RadioButtons:
    """Configuration for radio buttons selection widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""


@dataclass
class Select:
    """Configuration for a select list widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""
    rows: int = 5


@dataclass
class SelectMultiple:
    """Configuration for a multiple selection widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""
    rows: int = 5


@dataclass
class ToggleButtons:
    """Configuration for toggle buttons selection widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""
    button_style: str = ""


@dataclass
class SelectionSlider:
    """Configuration for a selection slider widget."""

    options: list[Any] = field(default_factory=list)
    description: str = ""
    continuous_update: bool = True
    orientation: str = "horizontal"
    readout: bool = True


# --- String Widgets ---


@dataclass
class Text:
    """Configuration for a text input widget."""

    placeholder: str = ""
    description: str = ""
    continuous_update: bool = True


@dataclass
class Textarea:
    """Configuration for a textarea widget."""

    placeholder: str = ""
    description: str = ""
    rows: int = 5
    continuous_update: bool = True


# =============================================================================
# Widget Factory
# =============================================================================


def create_widget(param_name: str, annotation: Any, default_value: Any) -> widgets.Widget:
    """
    Create an ipywidget based on the parameter annotation.

    Args:
        param_name: The name of the parameter (used as description if not specified)
        annotation: The type annotation (Annotated type with widget config)
        default_value: The default value for the parameter

    Returns:
        An ipywidget configured according to the annotation
    """
    # Extract the widget config from Annotated type
    origin = get_origin(annotation)

    if origin is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        widget_config = args[1] if len(args) > 1 else None
    else:
        # No Annotated wrapper, try to infer widget from type
        base_type = annotation
        widget_config = None

    # Get description (use param_name if not specified in config)
    description = param_name
    if widget_config and hasattr(widget_config, "description") and widget_config.description:
        description = widget_config.description

    # Create widget based on config type
    if isinstance(widget_config, IntSlider):
        return widgets.IntSlider(
            value=default_value
            if default_value is not inspect.Parameter.empty
            else widget_config.min,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
        )

    elif isinstance(widget_config, FloatSlider):
        return widgets.FloatSlider(
            value=default_value
            if default_value is not inspect.Parameter.empty
            else widget_config.min,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
            readout_format=widget_config.readout_format,
        )

    elif isinstance(widget_config, IntRangeSlider):
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (widget_config.min, widget_config.max)
        )
        return widgets.IntRangeSlider(
            value=default,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
        )

    elif isinstance(widget_config, FloatRangeSlider):
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (widget_config.min, widget_config.max)
        )
        return widgets.FloatRangeSlider(
            value=default,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
            readout_format=widget_config.readout_format,
        )

    elif isinstance(widget_config, FloatLogSlider):
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else widget_config.base**widget_config.min
        )
        return widgets.FloatLogSlider(
            value=default,
            base=widget_config.base,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
            readout_format=widget_config.readout_format,
        )

    elif isinstance(widget_config, IntText):
        return widgets.IntText(
            value=default_value if default_value is not inspect.Parameter.empty else 0,
            description=description,
            continuous_update=widget_config.continuous_update,
        )

    elif isinstance(widget_config, FloatText):
        return widgets.FloatText(
            value=default_value if default_value is not inspect.Parameter.empty else 0.0,
            description=description,
            continuous_update=widget_config.continuous_update,
        )

    elif isinstance(widget_config, BoundedIntText):
        return widgets.BoundedIntText(
            value=default_value
            if default_value is not inspect.Parameter.empty
            else widget_config.min,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
        )

    elif isinstance(widget_config, BoundedFloatText):
        return widgets.BoundedFloatText(
            value=default_value
            if default_value is not inspect.Parameter.empty
            else widget_config.min,
            min=widget_config.min,
            max=widget_config.max,
            step=widget_config.step,
            description=description,
            continuous_update=widget_config.continuous_update,
        )

    elif isinstance(widget_config, Checkbox):
        return widgets.Checkbox(
            value=default_value if default_value is not inspect.Parameter.empty else False,
            description=description,
            indent=widget_config.indent,
        )

    elif isinstance(widget_config, ToggleButton):
        return widgets.ToggleButton(
            value=default_value if default_value is not inspect.Parameter.empty else False,
            description=description,
            button_style=widget_config.button_style,
            icon=widget_config.icon,
        )

    elif isinstance(widget_config, Dropdown):
        options = widget_config.options
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (options[0] if options else None)
        )
        return widgets.Dropdown(options=options, value=default, description=description)

    elif isinstance(widget_config, RadioButtons):
        options = widget_config.options
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (options[0] if options else None)
        )
        return widgets.RadioButtons(options=options, value=default, description=description)

    elif isinstance(widget_config, Select):
        options = widget_config.options
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (options[0] if options else None)
        )
        return widgets.Select(
            options=options, value=default, description=description, rows=widget_config.rows
        )

    elif isinstance(widget_config, SelectMultiple):
        options = widget_config.options
        default = default_value if default_value is not inspect.Parameter.empty else ()
        return widgets.SelectMultiple(
            options=options, value=default, description=description, rows=widget_config.rows
        )

    elif isinstance(widget_config, ToggleButtons):
        options = widget_config.options
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (options[0] if options else None)
        )
        return widgets.ToggleButtons(
            options=options,
            value=default,
            description=description,
            button_style=widget_config.button_style,
        )

    elif isinstance(widget_config, SelectionSlider):
        options = widget_config.options
        default = (
            default_value
            if default_value is not inspect.Parameter.empty
            else (options[0] if options else None)
        )
        return widgets.SelectionSlider(
            options=options,
            value=default,
            description=description,
            continuous_update=widget_config.continuous_update,
            orientation=widget_config.orientation,
            readout=widget_config.readout,
        )

    elif isinstance(widget_config, Text):
        return widgets.Text(
            value=default_value if default_value is not inspect.Parameter.empty else "",
            placeholder=widget_config.placeholder,
            description=description,
            continuous_update=widget_config.continuous_update,
        )

    elif isinstance(widget_config, Textarea):
        return widgets.Textarea(
            value=default_value if default_value is not inspect.Parameter.empty else "",
            placeholder=widget_config.placeholder,
            description=description,
            rows=widget_config.rows,
            continuous_update=widget_config.continuous_update,
        )

    # Fallback: infer widget from base type
    else:
        return _infer_widget_from_type(param_name, base_type, default_value)


def _infer_widget_from_type(param_name: str, base_type: Any, default_value: Any) -> widgets.Widget:
    """
    Infer the appropriate widget from the base type when no explicit config is provided.
    """
    origin = get_origin(base_type)

    # Handle tuple types (for range sliders)
    if origin is tuple:
        args = get_args(base_type)
        if len(args) == 2 and all(arg in (int, float) for arg in args):
            if args[0] is int:
                return widgets.IntRangeSlider(
                    value=default_value
                    if default_value is not inspect.Parameter.empty
                    else (0, 100),
                    min=0,
                    max=100,
                    description=param_name,
                )
            else:
                return widgets.FloatRangeSlider(
                    value=default_value
                    if default_value is not inspect.Parameter.empty
                    else (0.0, 1.0),
                    min=0.0,
                    max=1.0,
                    description=param_name,
                )

    # Handle basic types
    if base_type is int:
        return widgets.IntSlider(
            value=default_value if default_value is not inspect.Parameter.empty else 0,
            min=0,
            max=100,
            description=param_name,
        )

    elif base_type is float:
        return widgets.FloatSlider(
            value=default_value if default_value is not inspect.Parameter.empty else 0.0,
            min=0.0,
            max=1.0,
            description=param_name,
        )

    elif base_type is bool:
        return widgets.Checkbox(
            value=default_value if default_value is not inspect.Parameter.empty else False,
            description=param_name,
        )

    elif base_type is str:
        return widgets.Text(
            value=default_value if default_value is not inspect.Parameter.empty else "",
            description=param_name,
        )

    # Default: return a text widget
    return widgets.Text(
        value=str(default_value) if default_value is not inspect.Parameter.empty else "",
        description=param_name,
    )


# =============================================================================
# Interactive Plotting Function
# =============================================================================


def interactive_plot(
    plot_func: Callable,
    figsize: tuple[float, float] = (10, 6),
    widget_layout: str = "vertical",
    widget_width: str = "400px",
    widgets_only: bool = False,
    figure_widget: widgets.Widget | None = None,
) -> widgets.Widget:
    """
    Create an interactive plot with widgets based on the plot function's parameters.

    Args:
        plot_func: A function that takes a matplotlib figure as first argument
                   and returns the modified figure. Other parameters should be
                   annotated with widget configurations.
        figsize: The size of the matplotlib figure (width, height) in inches.
        widget_layout: Layout of widgets - "vertical", "horizontal", or "grid"
        widget_width: Width of the widget container
        widgets_only: If True, only display the control widgets without creating
                      a figure. Useful when combining with InteractiveXYPlot or
                      other external figure management. The plot_func still receives
                      a dummy figure argument for compatibility.
        figure_widget: An optional external figure widget (e.g., from InteractiveXYPlot.show())
                       to combine with the parameter controls. When provided, the controls
                       are displayed above the figure widget, and widgets_only is implied.

    Returns:
        An ipywidget containing the interactive plot and controls.

    Example:
        ```python
        def my_plot(
            fig: mpl.figure.Figure,
            amplitude: Annotated[float, FloatSlider(0.1, 10.0)] = 1.0,
            frequency: Annotated[float, FloatSlider(0.1, 5.0)] = 1.0
        ) -> mpl.figure.Figure:
            ax = fig.add_subplot(111)
            x = np.linspace(0, 2*np.pi, 100)
            ax.plot(x, amplitude * np.sin(frequency * x))
            return fig

        interactive_plot(my_plot)

        # Or combine with InteractiveXYPlot:
        xy_plot = InteractiveXYPlot()
        def update_plot(fig, param: Annotated[float, FloatSlider()] = 1.0):
            xy_plot.clear()
            xy_plot.plot(x, param * y)
            return fig
        interactive_plot(update_plot, figure_widget=xy_plot.show())
        ```
    """
    # If figure_widget is provided, imply widgets_only mode
    if figure_widget is not None:
        widgets_only = True
    # Get function signature
    sig = inspect.signature(plot_func)
    params = sig.parameters

    # Get type hints
    try:
        hints = get_type_hints(plot_func, include_extras=True)
    except Exception:
        hints = {}

    # Create widgets for each parameter (skip 'fig' parameter)
    param_widgets = {}
    for param_name, param in params.items():
        # Skip the figure parameter
        if param_name == "fig":
            continue

        # Get annotation
        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            continue

        # Get default value
        default_value = param.default

        # Create widget
        widget = create_widget(param_name, annotation, default_value)
        param_widgets[param_name] = widget

    # Create output widget for the plot (only used if not widgets_only)
    output = widgets.Output()

    # Create figure once and reuse it (only used if not widgets_only)
    fig_container = {"fig": None, "initialized": False}

    # Function to update the plot
    def update_plot(*args, **kwargs):
        # Gather current widget values
        current_values = {name: w.value for name, w in param_widgets.items()}

        if widgets_only:
            # In widgets_only mode, just call the function with a dummy figure
            # The function is expected to handle its own figure management
            try:
                # Create a minimal dummy figure that won't be displayed
                dummy_fig = plt.figure(figsize=(0.1, 0.1))
                plot_func(dummy_fig, **current_values)
                plt.close(dummy_fig)  # Close immediately to avoid display
            except Exception as e:
                with output:
                    clear_output(wait=True)
                    print(f"Error in plot function: {e}")
                    import traceback

                    traceback.print_exc()
        elif not fig_container["initialized"]:
            # First time: create and display figure
            with output:
                clear_output(wait=True)
                fig_container["fig"] = plt.figure(figsize=figsize)
                try:
                    plot_func(fig_container["fig"], **current_values)
                    plt.show()
                    fig_container["initialized"] = True
                except Exception as e:
                    print(f"Error in plot function: {e}")
                    import traceback

                    traceback.print_exc()
        else:
            # Subsequent updates: clear and redraw
            fig = fig_container["fig"]
            fig.clear()
            try:
                plot_func(fig, **current_values)
                fig.canvas.draw_idle()
            except Exception as e:
                with output:
                    print(f"Error in plot function: {e}")
                    import traceback

                    traceback.print_exc()

    # Connect widgets to update function
    for widget in param_widgets.values():
        widget.observe(update_plot, names="value")

    # Create widget layout
    widget_list = list(param_widgets.values())

    if widget_layout == "horizontal":
        controls = widgets.HBox(widget_list)
    elif widget_layout == "grid":
        # Arrange in 2 columns
        rows = []
        for i in range(0, len(widget_list), 2):
            row = widget_list[i : i + 2]
            rows.append(widgets.HBox(row))
        controls = widgets.VBox(rows)
    else:  # vertical (default)
        controls = widgets.VBox(widget_list)

    # Style the controls
    controls.layout.width = widget_width

    # Create the main layout
    if figure_widget is not None:
        # Combine controls with the external figure widget
        main_layout = widgets.VBox([controls, figure_widget])
    elif widgets_only:
        # Only return controls, no output widget
        main_layout = controls
    else:
        main_layout = widgets.VBox([controls, output])

    # Initial plot
    update_plot()

    # Return the widget (don't call display - let Jupyter handle it)
    return main_layout


# Alias for convenience
interactive_plotting = interactive_plot


# =============================================================================
# Interactive Figure Classes
# =============================================================================


class InteractiveXYPlot:
    """
    An interactive XY plot class with built-in controls for axis limits,
    grid, legend, and file saving functionality.

    Features:
        - X-axis range slider below the plot for adjusting x_lim
        - Y-axis range slider on the left side for adjusting y_lim
        - Toggle buttons for grid and legend on top
        - File saving controls: filepath, filename, format selection, and save button

    Example:
        ```python
        import numpy as np
        from interactive_figure import InteractiveXYPlot

        # Create the interactive plot
        fig = InteractiveXYPlot(figsize=(10, 6))

        # Plot some data
        x = np.linspace(0, 10, 100)
        fig.plot(x, np.sin(x), label="sin(x)")
        fig.plot(x, np.cos(x), label="cos(x)")

        # Display the interactive figure
        fig.show()
        ```
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (10, 6),
        x_range: tuple[float, float] = (0.0, 10.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        show_grid: bool = True,
        show_legend: bool = True,
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        save_directory: str = "",
        default_dpi: int = 300,
        custom_fit_model: FitModel1D | None = None,
        fit_color: str = "red",
        fit_linestyle: str = "-",
    ):
        """
        Initialize the InteractiveXYPlot.

        Args:
            figsize: Figure size in inches (width, height)
            x_range: Initial x-axis range (min, max)
            y_range: Initial y-axis range (min, max)
            show_grid: Whether to show grid initially
            show_legend: Whether to show legend initially
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_directory: Default directory for saving figures
            default_dpi: Default DPI for saving figures (must be > 0)
            custom_fit_model: Optional custom FitModel1D for fitting
            fit_color: Default color for fit line (e.g., 'red', 'blue', '#FF0000')
            fit_linestyle: Default line style for fit line ('-', '--', ':', '-.')
        """
        self._figsize = figsize
        self._initial_x_range = x_range
        self._initial_y_range = y_range
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._save_directory = save_directory
        self._default_dpi = max(1, default_dpi)  # Ensure DPI is at least 1

        # Fit appearance defaults
        self._default_fit_color = fit_color
        self._default_fit_linestyle = fit_linestyle

        # Store plot data for redrawing
        self._plot_data: list[dict] = []

        # Track data bounds for auto-ranging
        self._x_min = float("inf")
        self._x_max = float("-inf")
        self._y_min = float("inf")
        self._y_max = float("-inf")

        # Create the figure and axis
        self._fig: matplotlib.figure.Figure | None = None
        self._ax: Any = None
        self._initialized = False

        # Fitting state
        self._custom_fit_model = custom_fit_model
        self._fit_result: Any = None
        self._fit_line: Any = None
        self._fit_params_widgets: dict[str, widgets.Widget] = {}
        self._current_fit_model: FitModel1D | None = None

        # Calculate pixel sizes based on figure size (approx 100 dpi for screen)
        fig_width_px = int(self._figsize[0] * 100)
        fig_height_px = int(self._figsize[1] * 100)

        # Create output widget for the plot - match figure size
        self._output = widgets.Output(
            layout=widgets.Layout(
                width=f"{fig_width_px}px",
                height=f"{fig_height_px}px",
            )
        )

        # Create control widgets
        self._create_widgets(show_grid, show_legend)

        # Build the layout
        self._layout = self._build_layout()

    def _create_widgets(self, show_grid: bool, show_legend: bool) -> None:
        """Create all control widgets."""
        # Calculate pixel sizes based on figure size (approx 100 dpi for screen)
        fig_width_px = int(self._figsize[0] * 100)
        fig_height_px = int(self._figsize[1] * 100)

        # Top controls: Grid, Legend checkboxes
        self._grid_checkbox = widgets.Checkbox(
            value=show_grid,
            description="Grid",
            indent=False,
        )
        self._grid_checkbox.observe(self._on_grid_change, names="value")

        self._legend_checkbox = widgets.Checkbox(
            value=show_legend,
            description="Legend",
            indent=False,
        )
        self._legend_checkbox.observe(self._on_legend_change, names="value")

        # Save controls
        self._filepath_text = widgets.Text(
            value=self._save_directory,
            placeholder="Path (empty = current dir)",
            description="Path:",
            layout=widgets.Layout(width="250px"),
        )

        self._filename_text = widgets.Text(
            value="figure",
            placeholder="Filename",
            description="File:",
            layout=widgets.Layout(width="200px"),
        )

        self._format_dropdown = widgets.Dropdown(
            options=[".png", ".jpg", ".svg", ".pdf", ".eps"],
            value=".png",
            description="Format:",
            layout=widgets.Layout(width="160px"),
        )

        self._dpi_text = widgets.BoundedIntText(
            value=self._default_dpi,
            min=1,
            max=2400,
            step=1,
            description="DPI:",
            layout=widgets.Layout(width="140px"),
        )

        self._save_button = widgets.Button(
            description="Save",
            icon="save",
            button_style="success",
            tooltip="Save figure to file",
        )
        self._save_button.on_click(self._on_save_click)

        self._save_status = widgets.HTML(value="")

        # X-axis range slider (horizontal, below plot) - match figure width
        self._x_slider = widgets.FloatRangeSlider(
            value=self._initial_x_range,
            min=self._initial_x_range[0],
            max=self._initial_x_range[1],
            step=0.01,
            description="X range:",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            layout=widgets.Layout(width=f"{fig_width_px}px"),
        )
        self._x_slider.observe(self._on_xlim_change, names="value")

        # Y-axis range slider (vertical, on the left) - match figure height
        self._y_slider = widgets.FloatRangeSlider(
            value=self._initial_y_range,
            min=self._initial_y_range[0],
            max=self._initial_y_range[1],
            step=0.01,
            description="",
            continuous_update=True,
            orientation="vertical",
            readout=True,
            readout_format=".2f",
            layout=widgets.Layout(height=f"{fig_height_px}px", width="50px"),
        )
        self._y_slider.observe(self._on_ylim_change, names="value")

        # Y-axis label
        self._y_label = widgets.HTML(
            value='<div style="writing-mode: vertical-rl; transform: rotate(180deg); '
            'text-align: center; font-size: 12px; height: 50px;">Y range</div>'
        )

        # Fitting controls
        self._fit_checkbox = widgets.Checkbox(
            value=False,
            description="Enable Fitting",
            indent=False,
        )
        self._fit_checkbox.observe(self._on_fit_toggle, names="value")

        # Model selection dropdown
        model_options = list(MODELS_1D.keys())
        if self._custom_fit_model is not None:
            model_options = ["Custom"] + model_options

        self._fit_model_dropdown = widgets.Dropdown(
            options=model_options,
            value=model_options[0],
            description="Model:",
            layout=widgets.Layout(width="200px"),
            disabled=True,
        )
        self._fit_model_dropdown.observe(self._on_fit_model_change, names="value")

        # Data series selection for fitting
        self._fit_data_dropdown = widgets.Dropdown(
            options=["All data"],
            value="All data",
            description="Fit data:",
            layout=widgets.Layout(width="200px"),
            disabled=True,
        )

        # Fit button
        self._fit_button = widgets.Button(
            description="Fit",
            icon="calculator",
            button_style="primary",
            tooltip="Perform fit with current parameters",
            disabled=True,
        )
        self._fit_button.on_click(self._on_fit_click)

        # Reset initial guesses button
        self._reset_guess_button = widgets.Button(
            description="Reset Guesses",
            icon="refresh",
            button_style="warning",
            tooltip="Reset parameters to automatic guesses",
            disabled=True,
        )
        self._reset_guess_button.on_click(self._on_reset_guesses)

        # Fit appearance controls
        self._fit_show_legend_checkbox = widgets.Checkbox(
            value=True,
            description="Show in legend",
            indent=False,
            disabled=True,
            layout=widgets.Layout(width="130px"),
        )

        # Color options for fit line
        fit_color_options = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "black",
            "gray",
        ]
        # Ensure default color is in options
        if self._default_fit_color not in fit_color_options:
            fit_color_options = [self._default_fit_color] + fit_color_options

        self._fit_color_dropdown = widgets.Dropdown(
            options=fit_color_options,
            value=self._default_fit_color,
            description="Color:",
            layout=widgets.Layout(width="160px"),
            disabled=True,
        )

        # Line style options: (label, value) format for ipywidgets
        linestyle_options = [
            ("Solid", "-"),
            ("Dashed", "--"),
            ("Dotted", ":"),
            ("Dash-dot", "-."),
        ]
        self._fit_linestyle_dropdown = widgets.Dropdown(
            options=linestyle_options,
            value=self._default_fit_linestyle,
            description="Style:",
            layout=widgets.Layout(width="160px"),
            disabled=True,
        )

        # Fit result display
        self._fit_result_html = widgets.HTML(
            value="",
            layout=widgets.Layout(width="100%"),
        )

        # Container for parameter sliders (dynamically populated)
        self._fit_params_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                display="none",
                width="100%",
                padding="5px",
                border="1px solid #ddd",
                margin="5px 0",
            ),
        )

        # Fitting controls container (initially hidden)
        self._fitting_controls_box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self._fit_model_dropdown,
                        self._fit_data_dropdown,
                        self._fit_button,
                        self._reset_guess_button,
                    ],
                    layout=widgets.Layout(margin="5px 0"),
                ),
                widgets.HBox(
                    [
                        self._fit_show_legend_checkbox,
                        self._fit_color_dropdown,
                        self._fit_linestyle_dropdown,
                    ],
                    layout=widgets.Layout(margin="5px 0"),
                ),
                self._fit_params_box,
                self._fit_result_html,
            ],
            layout=widgets.Layout(display="none"),
        )

    def _build_layout(self) -> widgets.Widget:
        """Build the complete widget layout."""
        # First row: display options (checkboxes)
        display_controls = widgets.HBox(
            [
                widgets.Label("Display:"),
                self._grid_checkbox,
                self._legend_checkbox,
                self._fit_checkbox,
            ],
            layout=widgets.Layout(
                justify_content="flex-start",
                align_items="center",
                margin="5px 0",
            ),
        )

        # Second row: save controls
        save_controls = widgets.HBox(
            [
                self._filepath_text,
                self._filename_text,
                self._format_dropdown,
                self._dpi_text,
                self._save_button,
                self._save_status,
            ],
            layout=widgets.Layout(
                justify_content="flex-start",
                align_items="center",
                margin="5px 0",
            ),
        )

        # Bottom controls combined (display options and save)
        bottom_controls = widgets.VBox(
            [display_controls, save_controls],
            layout=widgets.Layout(margin="10px 0 0 0"),
        )

        # Y slider with label (vertically stacked)
        y_slider_box = widgets.VBox(
            [self._y_label, self._y_slider],
            layout=widgets.Layout(
                align_items="center",
                justify_content="center",
                margin="0 10px 0 0",
                width="70px",
            ),
        )

        # Main plot area with Y slider on the left
        plot_row = widgets.HBox(
            [y_slider_box, self._output],
            layout=widgets.Layout(align_items="stretch"),
        )

        # X slider below the plot - with spacer matching y_slider_box width
        x_slider_spacer = widgets.HTML(
            value="",
            layout=widgets.Layout(width="70px", margin="0 10px 0 0"),
        )
        x_slider_box = widgets.HBox(
            [x_slider_spacer, self._x_slider],
            layout=widgets.Layout(align_items="center"),
        )

        # Main layout: figure first, then controls below
        main_layout = widgets.VBox(
            [plot_row, x_slider_box, bottom_controls, self._fitting_controls_box],
        )

        return main_layout

    def _on_fit_toggle(self, change: dict) -> None:
        """Handle fitting checkbox toggle."""
        enabled = change["new"]
        self._fit_model_dropdown.disabled = not enabled
        self._fit_data_dropdown.disabled = not enabled
        self._fit_button.disabled = not enabled
        self._reset_guess_button.disabled = not enabled
        self._fit_show_legend_checkbox.disabled = not enabled
        self._fit_color_dropdown.disabled = not enabled
        self._fit_linestyle_dropdown.disabled = not enabled

        if enabled:
            self._fitting_controls_box.layout.display = "block"
            self._update_fit_data_options()
            self._on_fit_model_change({"new": self._fit_model_dropdown.value})
        else:
            self._fitting_controls_box.layout.display = "none"
            self._fit_params_box.layout.display = "none"
            self._fit_result_html.value = ""
            self._clear_fit_line()

    def _update_fit_data_options(self) -> None:
        """Update the data series dropdown options."""
        options = []
        for i, data in enumerate(self._plot_data):
            label = data.get("label") or f"Series {i + 1}"
            options.append(label)
        if not options:
            options = ["No data"]
        self._fit_data_dropdown.options = options
        self._fit_data_dropdown.value = options[0]

    def _on_fit_model_change(self, change: dict) -> None:
        """Handle fit model selection change."""
        model_name = change["new"]

        # Get the model
        if model_name == "Custom" and self._custom_fit_model is not None:
            self._current_fit_model = self._custom_fit_model
        elif model_name in MODELS_1D:
            self._current_fit_model = MODELS_1D[model_name]()
        else:
            return

        # Create parameter sliders
        self._create_param_sliders()
        self._fit_params_box.layout.display = "block"

        # Auto-guess parameters
        self._auto_guess_parameters()

    def _create_param_sliders(self) -> None:
        """Create sliders with min/max bounds and fix toggle for model parameters."""
        if self._current_fit_model is None:
            return

        self._fit_params_widgets.clear()
        param_rows = []

        # Get parameter names from the model
        param_names = list(self._current_fit_model.model.param_names)

        # Get bounds from model (uses custom guess if available)
        x_data, y_data = self._get_fit_data()
        if x_data is not None and len(x_data) > 0:
            try:
                bounds = self._current_fit_model.get_param_bounds(y_data, x_data)
            except Exception:
                bounds = {}
        else:
            bounds = {}

        for param_name in param_names:
            hints = self._current_fit_model.param_hints.get(param_name, {})
            param_bounds = bounds.get(param_name, {})

            # Get initial values from bounds or hints
            value = param_bounds.get("value", hints.get("value", 1.0))
            bound_min = param_bounds.get("min", hints.get("min", -np.inf))
            bound_max = param_bounds.get("max", hints.get("max", np.inf))

            # Calculate slider range (finite values for slider)
            if np.isinf(bound_min) or bound_min is None:
                slider_min = value - abs(value) * 10 if value != 0 else -100
            else:
                slider_min = bound_min

            if np.isinf(bound_max) or bound_max is None:
                slider_max = value + abs(value) * 10 if value != 0 else 100
            else:
                slider_max = bound_max

            # Ensure value is within slider range
            if value < slider_min:
                slider_min = value - abs(value) * 0.5
            if value > slider_max:
                slider_max = value + abs(value) * 0.5

            # Parameter name label
            name_label = widgets.Label(
                value=param_name + ":",
                layout=widgets.Layout(width="80px"),
            )

            # Min bound text box (supports inf)
            min_text = widgets.FloatText(
                value=bound_min if not np.isinf(bound_min) else float("-inf"),
                description="",
                layout=widgets.Layout(width="80px"),
                step=0.1,
            )
            min_text.param_name = param_name  # Store reference
            min_text.observe(
                lambda change, pn=param_name: self._on_bound_change(change, pn, "min"),
                names="value",
            )

            # Value slider
            slider = widgets.FloatSlider(
                value=value,
                min=slider_min,
                max=slider_max,
                step=(slider_max - slider_min) / 100 if slider_max > slider_min else 0.01,
                description="",
                continuous_update=False,
                readout=True,
                readout_format=".4g",
                layout=widgets.Layout(width="250px"),
            )
            slider.observe(self._on_param_slider_change, names="value")

            # Max bound text box (supports inf)
            max_text = widgets.FloatText(
                value=bound_max if not np.isinf(bound_max) else float("inf"),
                description="",
                layout=widgets.Layout(width="80px"),
                step=0.1,
            )
            max_text.param_name = param_name  # Store reference
            max_text.observe(
                lambda change, pn=param_name: self._on_bound_change(change, pn, "max"),
                names="value",
            )

            # Fix toggle button
            fix_toggle = widgets.ToggleButton(
                value=False,
                description="Fix",
                button_style="",
                tooltip=f"Fix {param_name} to current value during fitting",
                icon="lock",
                layout=widgets.Layout(width="70px"),
            )
            fix_toggle.param_name = param_name  # Store reference
            fix_toggle.observe(
                lambda change, pn=param_name: self._on_fix_toggle(change, pn),
                names="value",
            )

            # Store all widgets for this parameter
            self._fit_params_widgets[param_name] = {
                "slider": slider,
                "min": min_text,
                "max": max_text,
                "fix": fix_toggle,
                "bound_min": bound_min,
                "bound_max": bound_max,
            }

            # Create row for this parameter
            param_row = widgets.HBox(
                [
                    name_label,
                    widgets.Label("min:", layout=widgets.Layout(width="30px")),
                    min_text,
                    slider,
                    widgets.Label("max:", layout=widgets.Layout(width="30px")),
                    max_text,
                    fix_toggle,
                ],
                layout=widgets.Layout(align_items="center", margin="2px 0"),
            )
            param_rows.append(param_row)

        self._fit_params_box.children = param_rows

    def _on_bound_change(self, change: dict, param_name: str, bound_type: str) -> None:
        """Handle min/max bound text change - update slider range."""
        if param_name not in self._fit_params_widgets:
            return

        widgets_dict = self._fit_params_widgets[param_name]
        slider = widgets_dict["slider"]
        new_value = change["new"]

        # Handle inf values
        if np.isinf(new_value):
            # For inf bounds, expand slider range instead
            if bound_type == "min":
                new_value = slider.value - abs(slider.value) * 10 if slider.value != 0 else -100
            else:
                new_value = slider.value + abs(slider.value) * 10 if slider.value != 0 else 100

        # Update stored bounds
        widgets_dict[f"bound_{bound_type}"] = change["new"]

        # Update slider range
        if bound_type == "min":
            if new_value < slider.max:
                slider.min = new_value
                # If current value is below new min, adjust it
                if slider.value < new_value:
                    slider.value = new_value
        else:  # max
            if new_value > slider.min:
                slider.max = new_value
                # If current value is above new max, adjust it
                if slider.value > new_value:
                    slider.value = new_value

        # Update step
        if slider.max > slider.min:
            slider.step = (slider.max - slider.min) / 100

        self._update_fit_preview()

    def _on_fix_toggle(self, change: dict, param_name: str) -> None:
        """Handle fix toggle button - enable/disable slider."""
        if param_name not in self._fit_params_widgets:
            return

        widgets_dict = self._fit_params_widgets[param_name]
        is_fixed = change["new"]

        # Disable slider and bound inputs when fixed
        widgets_dict["slider"].disabled = is_fixed
        widgets_dict["min"].disabled = is_fixed
        widgets_dict["max"].disabled = is_fixed

        # Update button style
        widgets_dict["fix"].button_style = "info" if is_fixed else ""
        widgets_dict["fix"].icon = "lock" if is_fixed else "unlock"

    def _auto_guess_parameters(self) -> None:
        """Automatically guess initial parameters based on data."""
        if self._current_fit_model is None or not self._plot_data:
            return

        # Get data for fitting
        x_data, y_data = self._get_fit_data()
        if x_data is None or len(x_data) == 0:
            return

        try:
            # Get bounds from model (uses custom guess if available)
            bounds = self._current_fit_model.get_param_bounds(y_data, x_data)

            # Update widgets with guessed values and bounds
            for name, param_bounds in bounds.items():
                if name in self._fit_params_widgets:
                    widgets_dict = self._fit_params_widgets[name]
                    slider = widgets_dict["slider"]
                    min_text = widgets_dict["min"]
                    max_text = widgets_dict["max"]

                    value = param_bounds.get("value", slider.value)
                    bound_min = param_bounds.get("min", -np.inf)
                    bound_max = param_bounds.get("max", np.inf)

                    # Update stored bounds
                    widgets_dict["bound_min"] = bound_min
                    widgets_dict["bound_max"] = bound_max

                    # Update bound textboxes
                    min_text.value = bound_min if not np.isinf(bound_min) else float("-inf")
                    max_text.value = bound_max if not np.isinf(bound_max) else float("inf")

                    # Calculate slider range
                    slider_min = (
                        bound_min
                        if not np.isinf(bound_min)
                        else (value - abs(value) * 10 if value != 0 else -100)
                    )
                    slider_max = (
                        bound_max
                        if not np.isinf(bound_max)
                        else (value + abs(value) * 10 if value != 0 else 100)
                    )

                    # Ensure value is within range
                    if value < slider_min:
                        slider_min = value - abs(value) * 0.5
                    if value > slider_max:
                        slider_max = value + abs(value) * 0.5

                    # Update slider
                    slider.min = slider_min
                    slider.max = slider_max
                    if slider_max > slider_min:
                        slider.step = (slider_max - slider_min) / 100
                    else:
                        slider.step = 0.01
                    slider.value = value

        except Exception:
            # If guess fails, try using lmfit's built-in guess
            try:
                params = self._current_fit_model.guess(y_data, x_data)
                for name, param in params.items():
                    if name in self._fit_params_widgets:
                        widgets_dict = self._fit_params_widgets[name]
                        slider = widgets_dict["slider"]
                        value = param.value
                        if value < slider.min:
                            slider.min = value - abs(value) * 0.5
                        if value > slider.max:
                            slider.max = value + abs(value) * 0.5
                        slider.value = value
            except Exception:
                pass

    def _get_fit_data(self) -> tuple[Any, Any]:
        """Get x and y data for fitting based on selection."""
        if not self._plot_data:
            return None, None

        selected = self._fit_data_dropdown.value
        for i, data in enumerate(self._plot_data):
            label = data.get("label") or f"Series {i + 1}"
            if label == selected:
                return data["x"], data["y"]

        # Fall back to first data series
        return self._plot_data[0]["x"], self._plot_data[0]["y"]

    def _on_param_slider_change(self, change: dict) -> None:
        """Handle parameter slider change - update preview line."""
        self._update_fit_preview()

    def _update_fit_preview(self) -> None:
        """Update the fit preview line with current parameter values."""
        if self._current_fit_model is None or self._ax is None:
            return

        x_data, _ = self._get_fit_data()
        if x_data is None:
            return

        # Get current parameter values from sliders
        params = {}
        for name, widgets_dict in self._fit_params_widgets.items():
            params[name] = widgets_dict["slider"].value

        try:
            # Evaluate model
            x_fit = np.linspace(float(x_data.min()), float(x_data.max()), 200)
            y_fit = self._current_fit_model.model.eval(
                self._current_fit_model.model.make_params(**params), x=x_fit
            )

            # Update or create fit line
            self._clear_fit_line()
            (self._fit_line,) = self._ax.plot(
                x_fit, y_fit, "r--", linewidth=2, label="Fit preview", alpha=0.7
            )
            self._fig.canvas.draw_idle()
        except Exception:
            pass

    def _clear_fit_line(self) -> None:
        """Remove the fit line from the plot."""
        if self._fit_line is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._fit_line.remove()
            self._fit_line = None
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    def _on_fit_click(self, button: widgets.Button) -> None:
        """Perform the fit with current parameters."""
        if self._current_fit_model is None:
            return

        x_data, y_data = self._get_fit_data()
        if x_data is None:
            self._fit_result_html.value = (
                '<span style="color: red;">No data available for fitting</span>'
            )
            return

        # Get initial parameters from sliders, apply bounds and fixed status
        init_params = self._current_fit_model.model.make_params()
        for name, widgets_dict in self._fit_params_widgets.items():
            if name in init_params:
                # Set value from slider
                init_params[name].value = widgets_dict["slider"].value

                # Apply bounds (support for inf)
                bound_min = widgets_dict.get("bound_min", -np.inf)
                bound_max = widgets_dict.get("bound_max", np.inf)

                if not np.isinf(bound_min):
                    init_params[name].min = bound_min
                if not np.isinf(bound_max):
                    init_params[name].max = bound_max

                # Apply fixed status
                if widgets_dict["fix"].value:
                    init_params[name].vary = False

        try:
            # Perform fit
            self._fit_result = self._current_fit_model.model.fit(y_data, init_params, x=x_data)

            # Update sliders with fitted values (only non-fixed parameters)
            for name, param in self._fit_result.params.items():
                if name in self._fit_params_widgets:
                    widgets_dict = self._fit_params_widgets[name]
                    slider = widgets_dict["slider"]
                    # Only update if not fixed
                    if not widgets_dict["fix"].value:
                        if param.value < slider.min:
                            slider.min = param.value - abs(param.value) * 0.5
                        if param.value > slider.max:
                            slider.max = param.value + abs(param.value) * 0.5
                        slider.value = param.value

            # Update fit line
            x_fit = np.linspace(float(x_data.min()), float(x_data.max()), 200)
            y_fit = self._fit_result.eval(x=x_fit)
            self._clear_fit_line()

            # Get fit appearance settings
            fit_color = self._fit_color_dropdown.value
            fit_linestyle = self._fit_linestyle_dropdown.value
            show_in_legend = self._fit_show_legend_checkbox.value
            fit_label = "Fit" if show_in_legend else "_nolegend_"

            (self._fit_line,) = self._ax.plot(
                x_fit,
                y_fit,
                color=fit_color,
                linestyle=fit_linestyle,
                linewidth=2,
                label=fit_label,
            )

            # Update legend if showing fit in legend
            if show_in_legend and self._legend_checkbox.value:
                self._ax.legend()

            self._fig.canvas.draw_idle()

            # Display results
            result_text = self._format_fit_result()
            self._fit_result_html.value = result_text

        except Exception as e:
            self._fit_result_html.value = f'<span style="color: red;">Fit failed: {e}</span>'

    def _format_fit_result(self) -> str:
        """Format fit result as HTML."""
        if self._fit_result is None:
            return ""

        # Get model name
        model_name = (
            self._current_fit_model.name if self._current_fit_model else "Unknown"
        )

        lines = ['<div style="font-family: monospace; font-size: 12px;">']
        lines.append(f"<b>Model:</b> {model_name}<br>")
        lines.append(
            f"<b>χ²</b> = {self._fit_result.chisqr:.4g}, "
            f"<b>reduced χ²</b> = {self._fit_result.redchi:.4g}<br>"
        )
        lines.append("<b>Parameters:</b><br>")

        for name, param in self._fit_result.params.items():
            stderr = f"± {param.stderr:.4g}" if param.stderr else "± N/A"
            fixed_str = " (fixed)" if not param.vary else ""
            lines.append(f"  {name} = {param.value:.6g} {stderr}{fixed_str}<br>")

        lines.append("</div>")
        return "".join(lines)

    def _on_reset_guesses(self, button: widgets.Button) -> None:
        """Reset parameter sliders to automatic guesses."""
        self._auto_guess_parameters()
        self._update_fit_preview()

    def _update_slider_ranges(self) -> None:
        """Update slider min/max ranges based on plotted data."""
        if self._x_min < float("inf") and self._x_max > float("-inf"):
            x_padding = (self._x_max - self._x_min) * 0.1 or 0.5
            new_x_min = self._x_min - x_padding
            new_x_max = self._x_max + x_padding
            # Update min/max carefully to avoid value out of range errors
            if new_x_min < self._x_slider.min:
                self._x_slider.min = new_x_min
            if new_x_max > self._x_slider.max:
                self._x_slider.max = new_x_max
            self._x_slider.min = new_x_min
            self._x_slider.max = new_x_max

        if self._y_min < float("inf") and self._y_max > float("-inf"):
            y_padding = (self._y_max - self._y_min) * 0.1 or 0.5
            new_y_min = self._y_min - y_padding
            new_y_max = self._y_max + y_padding
            # Update min/max carefully to avoid value out of range errors
            if new_y_min < self._y_slider.min:
                self._y_slider.min = new_y_min
            if new_y_max > self._y_slider.max:
                self._y_slider.max = new_y_max
            self._y_slider.min = new_y_min
            self._y_slider.max = new_y_max

    def _update_slider_values(self) -> None:
        """Update slider values to fit the current data."""
        if self._plot_data:
            x_padding = (self._x_max - self._x_min) * 0.05 or 0.5
            y_padding = (self._y_max - self._y_min) * 0.05 or 0.5
            self._x_slider.value = (self._x_min - x_padding, self._x_max + x_padding)
            self._y_slider.value = (self._y_min - y_padding, self._y_max + y_padding)

    def _redraw(self) -> None:
        """Redraw the plot with current settings."""
        if self._fig is None:
            return

        self._ax.clear()

        # Redraw all plot data
        for data in self._plot_data:
            plot_type = data.get("type", "plot")
            if plot_type == "plot":
                fmt = data.get("fmt", "")
                kwargs = data.get("kwargs", {}).copy()
                # Build plot arguments - only include fmt if non-empty
                plot_args = [data["x"], data["y"]]
                if fmt:
                    plot_args.append(fmt)
                self._ax.plot(
                    *plot_args,
                    label=data.get("label"),
                    **kwargs,
                )
            elif plot_type == "scatter":
                self._ax.scatter(
                    data["x"],
                    data["y"],
                    label=data.get("label"),
                    **data.get("kwargs", {}),
                )

        # Apply settings
        self._ax.set_xlim(self._x_slider.value)
        self._ax.set_ylim(self._y_slider.value)
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        if self._title:
            self._ax.set_title(self._title)

        if self._grid_checkbox.value:
            self._ax.grid(True, alpha=0.3)

        if self._legend_checkbox.value and any(d.get("label") for d in self._plot_data):
            self._ax.legend()

        self._fig.canvas.draw_idle()

    def _on_xlim_change(self, change: dict) -> None:
        """Handle X-axis range slider change."""
        if self._ax is not None:
            self._ax.set_xlim(change["new"])
            self._fig.canvas.draw_idle()

    def _on_ylim_change(self, change: dict) -> None:
        """Handle Y-axis range slider change."""
        if self._ax is not None:
            self._ax.set_ylim(change["new"])
            self._fig.canvas.draw_idle()

    def _on_grid_change(self, change: dict) -> None:
        """Handle grid checkbox change."""
        self._redraw()

    def _on_legend_change(self, change: dict) -> None:
        """Handle legend checkbox change."""
        self._redraw()

    def _on_save_click(self, button: widgets.Button) -> None:
        """Handle save button click."""
        import os

        filepath = self._filepath_text.value.strip()
        filename = self._filename_text.value.strip() or "figure"
        fmt = self._format_dropdown.value
        dpi = max(1, self._dpi_text.value)  # Ensure DPI is at least 1

        full_path = os.path.join(filepath, filename + fmt) if filepath else filename + fmt

        try:
            self._fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
            self._save_status.value = f'<span style="color: green;">✓ Saved: {full_path}</span>'
        except Exception as e:
            self._save_status.value = f'<span style="color: red;">✗ Error: {e}</span>'

    def plot(
        self,
        x,
        y,
        fmt: str = "",
        label: str | None = None,
        **kwargs,
    ) -> None:
        """
        Add a line plot to the figure.

        Args:
            x: X data (array-like)
            y: Y data (array-like)
            fmt: Format string (e.g., 'b-', 'ro')
            label: Label for the legend
            **kwargs: Additional arguments passed to ax.plot()
        """
        x = np.asarray(x)
        y = np.asarray(y)

        self._plot_data.append(
            {"type": "plot", "x": x, "y": y, "fmt": fmt, "label": label, "kwargs": kwargs}
        )

        # Update bounds
        self._x_min = min(self._x_min, float(np.nanmin(x)))
        self._x_max = max(self._x_max, float(np.nanmax(x)))
        self._y_min = min(self._y_min, float(np.nanmin(y)))
        self._y_max = max(self._y_max, float(np.nanmax(y)))

        # Update fit data dropdown
        self._update_fit_data_options()

        # Redraw if figure is already initialized
        if self._initialized:
            self._update_slider_ranges()
            self._update_slider_values()
            self._redraw()

    def scatter(
        self,
        x,
        y,
        label: str | None = None,
        **kwargs,
    ) -> None:
        """
        Add a scatter plot to the figure.

        Args:
            x: X data (array-like)
            y: Y data (array-like)
            label: Label for the legend
            **kwargs: Additional arguments passed to ax.scatter()
        """
        x = np.asarray(x)
        y = np.asarray(y)

        self._plot_data.append(
            {"type": "scatter", "x": x, "y": y, "label": label, "kwargs": kwargs}
        )

        # Update bounds
        self._x_min = min(self._x_min, float(np.nanmin(x)))
        self._x_max = max(self._x_max, float(np.nanmax(x)))
        self._y_min = min(self._y_min, float(np.nanmin(y)))
        self._y_max = max(self._y_max, float(np.nanmax(y)))

        # Update fit data dropdown
        self._update_fit_data_options()

        # Redraw if figure is already initialized
        if self._initialized:
            self._update_slider_ranges()
            self._update_slider_values()
            self._redraw()

    def set_title(self, title: str) -> None:
        """Set the plot title."""
        self._title = title
        if self._ax is not None:
            self._ax.set_title(title)
            self._fig.canvas.draw_idle()

    def set_xlabel(self, label: str) -> None:
        """Set the X-axis label."""
        self._xlabel = label
        if self._ax is not None:
            self._ax.set_xlabel(label)
            self._fig.canvas.draw_idle()

    def set_ylabel(self, label: str) -> None:
        """Set the Y-axis label."""
        self._ylabel = label
        if self._ax is not None:
            self._ax.set_ylabel(label)
            self._fig.canvas.draw_idle()

    def clear(self) -> None:
        """Clear all plot data."""
        self._plot_data.clear()
        self._x_min = float("inf")
        self._x_max = float("-inf")
        self._y_min = float("inf")
        self._y_max = float("-inf")
        # Note: Don't clear the axis here, let _redraw() handle it
        # This prevents issues when clear() is called before new data is added

    def show(self) -> widgets.Widget:
        """
        Display the interactive plot.

        Returns:
            The main widget layout for display in Jupyter.
        """
        # Update slider ranges based on data
        self._update_slider_ranges()

        # Set slider values to fit data
        if self._plot_data:
            x_padding = (self._x_max - self._x_min) * 0.05 or 0.5
            y_padding = (self._y_max - self._y_min) * 0.05 or 0.5
            self._x_slider.value = (self._x_min - x_padding, self._x_max + x_padding)
            self._y_slider.value = (self._y_min - y_padding, self._y_max + y_padding)

        # Create and display the figure
        with self._output:
            clear_output(wait=True)
            self._fig = plt.figure(figsize=self._figsize)
            self._ax = self._fig.add_subplot(111)
            self._redraw()
            plt.show()
            self._initialized = True

        return self._layout

    @property
    def figure(self) -> matplotlib.figure.Figure | None:
        """Get the underlying matplotlib figure."""
        return self._fig

    @property
    def ax(self) -> Any:
        """Get the underlying matplotlib axes."""
        return self._ax

    @property
    def fit_result(self) -> Any:
        """Get the last fit result (lmfit ModelResult object)."""
        return self._fit_result

    def set_custom_fit_model(self, model: FitModel1D) -> None:
        """
        Set a custom fit model.

        Args:
            model: FitModel1D object for custom fitting
        """
        self._custom_fit_model = model
        # Update dropdown options
        model_options = list(MODELS_1D.keys())
        model_options = ["Custom"] + model_options
        self._fit_model_dropdown.options = model_options


# =============================================================================
# InteractiveHeatmap Class
# =============================================================================


class InteractiveHeatmap:
    """
    An interactive 2D heatmap/image plot class with built-in controls for axis limits,
    color scale, colormap selection, and file saving functionality.

    Features:
        - X-axis range slider below the plot for adjusting x_lim
        - Y-axis range slider on the left side for adjusting y_lim
        - Color scale range slider for adjusting vmin/vmax
        - Dropdown for selecting matplotlib colormaps
        - Grid toggle at the top
        - Colorbar display
        - File saving controls: filepath, filename, format selection, DPI, and save button
        - Support for custom colormaps via the cmap parameter

    Example:
        ```python
        import numpy as np
        from interactive_figure import InteractiveHeatmap

        # Create some 2D data
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        # Create the interactive heatmap
        heatmap = InteractiveHeatmap(
            figsize=(10, 8),
            title="2D Heatmap Example",
            xlabel="X",
            ylabel="Y",
        )

        # Set the data
        heatmap.set_data(Z, extent=[-3, 3, -3, 3])

        # Display the interactive heatmap
        heatmap.show()
        ```
    """

    # Common matplotlib colormaps for the dropdown
    COLORMAPS = [
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "gray",
        "hot",
        "cool",
        "coolwarm",
        "bwr",
        "seismic",
        "RdBu",
        "RdYlBu",
        "RdYlGn",
        "Spectral",
        "jet",
        "turbo",
        "rainbow",
        "terrain",
        "ocean",
    ]

    def __init__(
        self,
        figsize: tuple[float, float] = (10, 8),
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
        color_range: tuple[float, float] = (0.0, 1.0),
        cmap: str | Any = "viridis",
        show_grid: bool = False,
        show_colorbar: bool = True,
        title: str = "",
        xlabel: str = "x",
        ylabel: str = "y",
        colorbar_label: str = "",
        save_directory: str = "",
        default_dpi: int = 300,
        aspect: str = "auto",
        interpolation: str = "nearest",
        custom_fit_model: FitModel2D | None = None,
        fit_contour_color: str = "white",
        fit_contour_linestyle: str = "-",
    ):
        """
        Initialize the InteractiveHeatmap.

        Args:
            figsize: Figure size in inches (width, height)
            x_range: Initial x-axis range (min, max) for extent
            y_range: Initial y-axis range (min, max) for extent
            color_range: Initial color scale range (vmin, vmax)
            cmap: Colormap name (string) or custom matplotlib colormap object
            show_grid: Whether to show grid initially
            show_colorbar: Whether to show colorbar
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            colorbar_label: Label for the colorbar
            save_directory: Default directory for saving figures
            default_dpi: Default DPI for saving figures (must be > 0)
            aspect: Aspect ratio ('auto', 'equal', or float)
            interpolation: Interpolation method ('nearest', 'bilinear', 'bicubic', etc.)
            custom_fit_model: Optional custom FitModel2D for fitting
            fit_contour_color: Default color for fit contour lines
            fit_contour_linestyle: Default line style for fit contour lines
        """
        self._figsize = figsize
        self._initial_x_range = x_range
        self._initial_y_range = y_range
        self._initial_color_range = color_range
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._colorbar_label = colorbar_label
        self._save_directory = save_directory
        self._default_dpi = max(1, default_dpi)
        self._aspect = aspect
        self._interpolation = interpolation
        self._show_colorbar = show_colorbar

        # Fit contour appearance defaults
        self._default_fit_contour_color = fit_contour_color
        self._default_fit_contour_linestyle = fit_contour_linestyle

        # Store the colormap - can be string or custom colormap
        self._custom_cmap = cmap if not isinstance(cmap, str) else None
        self._cmap_name = cmap if isinstance(cmap, str) else "custom"

        # Store image data
        self._data: Any = None
        self._extent: tuple[float, float, float, float] | None = None

        # Track data bounds
        self._x_min = x_range[0]
        self._x_max = x_range[1]
        self._y_min = y_range[0]
        self._y_max = y_range[1]
        self._vmin = color_range[0]
        self._vmax = color_range[1]

        # Create the figure and axis
        self._fig: matplotlib.figure.Figure | None = None
        self._ax: Any = None
        self._im: Any = None
        self._colorbar: Any = None
        self._initialized = False

        # Fitting state
        self._custom_fit_model = custom_fit_model
        self._fit_result: Any = None
        self._fit_contour: Any = None
        self._preview_contour: Any = None
        self._fit_params_widgets: dict[str, widgets.Widget] = {}
        self._current_fit_model: FitModel2D | None = None
        self._X_grid: Any = None
        self._Y_grid: Any = None

        # Calculate pixel sizes based on figure size (approx 100 dpi for screen)
        fig_width_px = int(self._figsize[0] * 100)
        fig_height_px = int(self._figsize[1] * 100)

        # Create output widget for the plot
        self._output = widgets.Output(
            layout=widgets.Layout(
                width=f"{fig_width_px}px",
                height=f"{fig_height_px}px",
            )
        )

        # Create control widgets
        self._create_widgets(show_grid, cmap)

        # Build the layout
        self._layout = self._build_layout()

    def _create_widgets(self, show_grid: bool, cmap: str | Any) -> None:
        """Create all control widgets."""
        fig_width_px = int(self._figsize[0] * 100)
        fig_height_px = int(self._figsize[1] * 100)

        # Grid checkbox
        self._grid_checkbox = widgets.Checkbox(
            value=show_grid,
            description="Grid",
            indent=False,
        )
        self._grid_checkbox.observe(self._on_grid_change, names="value")

        # Colormap dropdown
        cmap_options = self.COLORMAPS.copy()
        if self._custom_cmap is not None:
            cmap_options = ["custom"] + cmap_options
        initial_cmap = self._cmap_name if self._cmap_name in cmap_options else "viridis"

        self._cmap_dropdown = widgets.Dropdown(
            options=cmap_options,
            value=initial_cmap,
            description="Colormap:",
            layout=widgets.Layout(width="180px"),
        )
        self._cmap_dropdown.observe(self._on_cmap_change, names="value")

        # Color scale range slider
        self._color_slider = widgets.FloatRangeSlider(
            value=self._initial_color_range,
            min=self._initial_color_range[0],
            max=self._initial_color_range[1],
            step=0.01,
            description="Color range:",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            layout=widgets.Layout(width="400px"),
        )
        self._color_slider.observe(self._on_color_range_change, names="value")

        # Save controls
        self._filepath_text = widgets.Text(
            value=self._save_directory,
            placeholder="Path (empty = current dir)",
            description="Path:",
            layout=widgets.Layout(width="250px"),
        )

        self._filename_text = widgets.Text(
            value="heatmap",
            placeholder="Filename",
            description="File:",
            layout=widgets.Layout(width="200px"),
        )

        self._format_dropdown = widgets.Dropdown(
            options=[".png", ".jpg", ".svg", ".pdf", ".eps"],
            value=".png",
            description="Format:",
            layout=widgets.Layout(width="160px"),
        )

        self._dpi_text = widgets.BoundedIntText(
            value=self._default_dpi,
            min=1,
            max=2400,
            step=1,
            description="DPI:",
            layout=widgets.Layout(width="140px"),
        )

        self._save_button = widgets.Button(
            description="Save",
            icon="save",
            button_style="success",
            tooltip="Save figure to file",
        )
        self._save_button.on_click(self._on_save_click)

        self._save_status = widgets.HTML(value="")

        # X-axis range slider (horizontal, below plot)
        self._x_slider = widgets.FloatRangeSlider(
            value=self._initial_x_range,
            min=self._initial_x_range[0],
            max=self._initial_x_range[1],
            step=0.01,
            description="X range:",
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            layout=widgets.Layout(width=f"{fig_width_px}px"),
        )
        self._x_slider.observe(self._on_xlim_change, names="value")

        # Y-axis range slider (vertical, on the left)
        self._y_slider = widgets.FloatRangeSlider(
            value=self._initial_y_range,
            min=self._initial_y_range[0],
            max=self._initial_y_range[1],
            step=0.01,
            description="",
            continuous_update=True,
            orientation="vertical",
            readout=True,
            readout_format=".2f",
            layout=widgets.Layout(height=f"{fig_height_px}px", width="50px"),
        )
        self._y_slider.observe(self._on_ylim_change, names="value")

        # Y-axis label
        self._y_label = widgets.HTML(
            value='<div style="writing-mode: vertical-rl; transform: rotate(180deg); '
            'text-align: center; font-size: 12px; height: 50px;">Y range</div>'
        )

        # Fitting controls
        self._fit_checkbox = widgets.Checkbox(
            value=False,
            description="Enable Fitting",
            indent=False,
        )
        self._fit_checkbox.observe(self._on_fit_toggle, names="value")

        # Model selection dropdown
        model_options = list(MODELS_2D.keys())
        if self._custom_fit_model is not None:
            model_options = ["Custom"] + model_options

        self._fit_model_dropdown = widgets.Dropdown(
            options=model_options,
            value=model_options[0],
            description="Model:",
            layout=widgets.Layout(width="220px"),
            disabled=True,
        )
        self._fit_model_dropdown.observe(self._on_fit_model_change, names="value")

        # Fit button
        self._fit_button = widgets.Button(
            description="Fit",
            icon="calculator",
            button_style="primary",
            tooltip="Perform 2D fit with current parameters",
            disabled=True,
        )
        self._fit_button.on_click(self._on_fit_click)

        # Reset initial guesses button
        self._reset_guess_button = widgets.Button(
            description="Reset Guesses",
            icon="refresh",
            button_style="warning",
            tooltip="Reset parameters to automatic guesses",
            disabled=True,
        )
        self._reset_guess_button.on_click(self._on_reset_guesses)

        # Show fit contour checkbox
        self._show_fit_contour = widgets.Checkbox(
            value=True,
            description="Show Fit Contour",
            indent=False,
            disabled=True,
        )
        self._show_fit_contour.observe(self._on_show_fit_contour_change, names="value")

        # Fit contour appearance controls
        fit_contour_color_options = [
            "white",
            "black",
            "red",
            "blue",
            "green",
            "yellow",
            "cyan",
            "magenta",
            "orange",
        ]
        # Ensure default color is in options
        if self._default_fit_contour_color not in fit_contour_color_options:
            fit_contour_color_options = [
                self._default_fit_contour_color
            ] + fit_contour_color_options

        self._fit_contour_color_dropdown = widgets.Dropdown(
            options=fit_contour_color_options,
            value=self._default_fit_contour_color,
            description="Color:",
            layout=widgets.Layout(width="160px"),
            disabled=True,
        )

        # Line style options for contour: (label, value) format for ipywidgets
        contour_linestyle_options = [
            ("Solid", "-"),
            ("Dashed", "--"),
            ("Dotted", ":"),
            ("Dash-dot", "-."),
        ]
        self._fit_contour_linestyle_dropdown = widgets.Dropdown(
            options=contour_linestyle_options,
            value=self._default_fit_contour_linestyle,
            description="Style:",
            layout=widgets.Layout(width="160px"),
            disabled=True,
        )

        # Fit result display
        self._fit_result_html = widgets.HTML(
            value="",
            layout=widgets.Layout(width="100%"),
        )

        # Container for parameter sliders (dynamically populated)
        self._fit_params_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                display="none",
                width="100%",
                padding="5px",
                border="1px solid #ddd",
                margin="5px 0",
            ),
        )

        # Fitting controls container (initially hidden)
        self._fitting_controls_box = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self._fit_model_dropdown,
                        self._fit_button,
                        self._reset_guess_button,
                    ],
                    layout=widgets.Layout(margin="5px 0"),
                ),
                widgets.HBox(
                    [
                        self._show_fit_contour,
                        self._fit_contour_color_dropdown,
                        self._fit_contour_linestyle_dropdown,
                    ],
                    layout=widgets.Layout(margin="5px 0"),
                ),
                self._fit_params_box,
                self._fit_result_html,
            ],
            layout=widgets.Layout(display="none"),
        )

    def _build_layout(self) -> widgets.Widget:
        """Build the complete widget layout."""
        # First row: display options (grid, colormap, color range)
        display_controls = widgets.HBox(
            [
                widgets.Label("Display:"),
                self._grid_checkbox,
                self._cmap_dropdown,
                self._color_slider,
                self._fit_checkbox,
            ],
            layout=widgets.Layout(
                justify_content="flex-start",
                align_items="center",
                margin="5px 0",
            ),
        )

        # Second row: save controls
        save_controls = widgets.HBox(
            [
                self._filepath_text,
                self._filename_text,
                self._format_dropdown,
                self._dpi_text,
                self._save_button,
                self._save_status,
            ],
            layout=widgets.Layout(
                justify_content="flex-start",
                align_items="center",
                margin="5px 0",
            ),
        )

        # Bottom controls combined
        bottom_controls = widgets.VBox(
            [display_controls, save_controls],
            layout=widgets.Layout(margin="10px 0 0 0"),
        )

        # Y slider with label
        y_slider_box = widgets.VBox(
            [self._y_label, self._y_slider],
            layout=widgets.Layout(
                align_items="center",
                justify_content="center",
                margin="0 10px 0 0",
                width="70px",
            ),
        )

        # Main plot area with Y slider on the left
        plot_row = widgets.HBox(
            [y_slider_box, self._output],
            layout=widgets.Layout(align_items="stretch"),
        )

        # X slider below the plot
        x_slider_spacer = widgets.HTML(
            value="",
            layout=widgets.Layout(width="70px", margin="0 10px 0 0"),
        )
        x_slider_box = widgets.HBox(
            [x_slider_spacer, self._x_slider],
            layout=widgets.Layout(align_items="center"),
        )

        # Main layout
        main_layout = widgets.VBox(
            [plot_row, x_slider_box, bottom_controls, self._fitting_controls_box],
        )

        return main_layout

    def _on_fit_toggle(self, change: dict) -> None:
        """Handle fitting checkbox toggle."""
        enabled = change["new"]
        self._fit_model_dropdown.disabled = not enabled
        self._fit_button.disabled = not enabled
        self._reset_guess_button.disabled = not enabled
        self._show_fit_contour.disabled = not enabled
        self._fit_contour_color_dropdown.disabled = not enabled
        self._fit_contour_linestyle_dropdown.disabled = not enabled

        if enabled:
            self._fitting_controls_box.layout.display = "block"
            self._on_fit_model_change({"new": self._fit_model_dropdown.value})
        else:
            self._fitting_controls_box.layout.display = "none"
            self._fit_params_box.layout.display = "none"
            self._fit_result_html.value = ""
            self._clear_fit_contour()
            self._clear_preview_contour()

    def _on_fit_model_change(self, change: dict) -> None:
        """Handle fit model selection change."""
        model_name = change["new"]

        # Clear existing contours when changing models
        self._clear_fit_contour()
        self._clear_preview_contour()
        self._fit_result = None
        self._fit_result_html.value = ""

        # Get the model
        if model_name == "Custom" and self._custom_fit_model is not None:
            self._current_fit_model = self._custom_fit_model
        elif model_name in MODELS_2D:
            self._current_fit_model = MODELS_2D[model_name]()
        else:
            return

        # Create parameter sliders
        self._create_param_sliders()
        self._fit_params_box.layout.display = "block"

        # Auto-guess parameters
        self._auto_guess_parameters()

    def _create_param_sliders(self) -> None:
        """Create sliders with min/max bounds and fix toggle for model parameters."""
        if self._current_fit_model is None:
            return

        self._fit_params_widgets.clear()
        param_rows = []

        # Get bounds from model (uses custom guess if available)
        if self._X_grid is None or self._Y_grid is None:
            self._create_coordinate_grids()

        if self._data is not None and self._X_grid is not None:
            try:
                bounds = self._current_fit_model.get_param_bounds(
                    self._data, self._X_grid, self._Y_grid
                )
            except Exception:
                bounds = {}
        else:
            bounds = {}

        for param_name in self._current_fit_model.param_names:
            hints = self._current_fit_model.param_hints.get(param_name, {})
            param_bounds = bounds.get(param_name, {})

            # Get initial values from bounds or hints
            value = param_bounds.get("value", hints.get("value", 1.0))
            bound_min = param_bounds.get("min", hints.get("min", -np.inf))
            bound_max = param_bounds.get("max", hints.get("max", np.inf))

            # Calculate slider range (finite values for slider)
            if np.isinf(bound_min) or bound_min is None:
                slider_min = value - abs(value) * 10 if value != 0 else -100
            else:
                slider_min = bound_min

            if np.isinf(bound_max) or bound_max is None:
                slider_max = value + abs(value) * 10 if value != 0 else 100
            else:
                slider_max = bound_max

            # Ensure value is within slider range
            if value < slider_min:
                slider_min = value - abs(value) * 0.5
            if value > slider_max:
                slider_max = value + abs(value) * 0.5

            # Parameter name label
            name_label = widgets.Label(
                value=param_name + ":",
                layout=widgets.Layout(width="80px"),
            )

            # Min bound text box (supports inf)
            min_text = widgets.FloatText(
                value=bound_min if not np.isinf(bound_min) else float("-inf"),
                description="",
                layout=widgets.Layout(width="80px"),
                step=0.1,
            )
            min_text.param_name = param_name
            min_text.observe(
                lambda change, pn=param_name: self._on_bound_change(change, pn, "min"),
                names="value",
            )

            # Value slider
            slider = widgets.FloatSlider(
                value=value,
                min=slider_min,
                max=slider_max,
                step=(slider_max - slider_min) / 100 if slider_max > slider_min else 0.01,
                description="",
                continuous_update=False,
                readout=True,
                readout_format=".4g",
                layout=widgets.Layout(width="200px"),
            )
            slider.observe(self._on_param_slider_change, names="value")

            # Max bound text box (supports inf)
            max_text = widgets.FloatText(
                value=bound_max if not np.isinf(bound_max) else float("inf"),
                description="",
                layout=widgets.Layout(width="80px"),
                step=0.1,
            )
            max_text.param_name = param_name
            max_text.observe(
                lambda change, pn=param_name: self._on_bound_change(change, pn, "max"),
                names="value",
            )

            # Fix toggle button
            fix_toggle = widgets.ToggleButton(
                value=False,
                description="Fix",
                button_style="",
                tooltip=f"Fix {param_name} to current value during fitting",
                icon="lock",
                layout=widgets.Layout(width="70px"),
            )
            fix_toggle.param_name = param_name
            fix_toggle.observe(
                lambda change, pn=param_name: self._on_fix_toggle(change, pn),
                names="value",
            )

            # Store all widgets for this parameter
            self._fit_params_widgets[param_name] = {
                "slider": slider,
                "min": min_text,
                "max": max_text,
                "fix": fix_toggle,
                "bound_min": bound_min,
                "bound_max": bound_max,
            }

            # Create row for this parameter
            param_row = widgets.HBox(
                [
                    name_label,
                    widgets.Label("min:", layout=widgets.Layout(width="30px")),
                    min_text,
                    slider,
                    widgets.Label("max:", layout=widgets.Layout(width="30px")),
                    max_text,
                    fix_toggle,
                ],
                layout=widgets.Layout(align_items="center", margin="2px 0"),
            )
            param_rows.append(param_row)

        self._fit_params_box.children = param_rows

    def _on_bound_change(self, change: dict, param_name: str, bound_type: str) -> None:
        """Handle min/max bound text change - update slider range."""
        if param_name not in self._fit_params_widgets:
            return

        widgets_dict = self._fit_params_widgets[param_name]
        slider = widgets_dict["slider"]
        new_value = change["new"]

        # Handle inf values
        if np.isinf(new_value):
            if bound_type == "min":
                new_value = slider.value - abs(slider.value) * 10 if slider.value != 0 else -100
            else:
                new_value = slider.value + abs(slider.value) * 10 if slider.value != 0 else 100

        # Update stored bounds
        widgets_dict[f"bound_{bound_type}"] = change["new"]

        # Update slider range
        if bound_type == "min":
            if new_value < slider.max:
                slider.min = new_value
                if slider.value < new_value:
                    slider.value = new_value
        else:
            if new_value > slider.min:
                slider.max = new_value
                if slider.value > new_value:
                    slider.value = new_value

        # Update step
        if slider.max > slider.min:
            slider.step = (slider.max - slider.min) / 100

    def _on_fix_toggle(self, change: dict, param_name: str) -> None:
        """Handle fix toggle button - enable/disable slider."""
        if param_name not in self._fit_params_widgets:
            return

        widgets_dict = self._fit_params_widgets[param_name]
        is_fixed = change["new"]

        # Disable slider and bound inputs when fixed
        widgets_dict["slider"].disabled = is_fixed
        widgets_dict["min"].disabled = is_fixed
        widgets_dict["max"].disabled = is_fixed

        # Update button style
        widgets_dict["fix"].button_style = "info" if is_fixed else ""
        widgets_dict["fix"].icon = "lock" if is_fixed else "unlock"

    def _auto_guess_parameters(self) -> None:
        """Automatically guess initial parameters based on data."""
        if self._current_fit_model is None or self._data is None:
            return

        if self._X_grid is None or self._Y_grid is None:
            self._create_coordinate_grids()

        try:
            # Get bounds from model (uses custom guess if available)
            bounds = self._current_fit_model.get_param_bounds(
                self._data, self._X_grid, self._Y_grid
            )

            # Update widgets with guessed values and bounds
            for name, param_bounds in bounds.items():
                if name in self._fit_params_widgets:
                    widgets_dict = self._fit_params_widgets[name]
                    slider = widgets_dict["slider"]
                    min_text = widgets_dict["min"]
                    max_text = widgets_dict["max"]

                    value = param_bounds.get("value", slider.value)
                    bound_min = param_bounds.get("min", -np.inf)
                    bound_max = param_bounds.get("max", np.inf)

                    # Update stored bounds
                    widgets_dict["bound_min"] = bound_min
                    widgets_dict["bound_max"] = bound_max

                    # Update bound textboxes
                    min_text.value = bound_min if not np.isinf(bound_min) else float("-inf")
                    max_text.value = bound_max if not np.isinf(bound_max) else float("inf")

                    # Calculate slider range
                    slider_min = (
                        bound_min
                        if not np.isinf(bound_min)
                        else (value - abs(value) * 10 if value != 0 else -100)
                    )
                    slider_max = (
                        bound_max
                        if not np.isinf(bound_max)
                        else (value + abs(value) * 10 if value != 0 else 100)
                    )

                    # Ensure value is within range
                    if value < slider_min:
                        slider_min = value - abs(value) * 0.5
                    if value > slider_max:
                        slider_max = value + abs(value) * 0.5

                    # Update slider
                    slider.min = slider_min
                    slider.max = slider_max
                    if slider_max > slider_min:
                        slider.step = (slider_max - slider_min) / 100
                    else:
                        slider.step = 0.01
                    slider.value = value

        except Exception:
            # If guess fails, try using default guess
            try:
                guesses = self._current_fit_model.guess(self._data, self._X_grid, self._Y_grid)
                for name, value in guesses.items():
                    if name in self._fit_params_widgets:
                        widgets_dict = self._fit_params_widgets[name]
                        slider = widgets_dict["slider"]
                        if value < slider.min:
                            slider.min = value - abs(value) * 0.5
                        if value > slider.max:
                            slider.max = value + abs(value) * 0.5
                        slider.value = value
            except Exception:
                pass

    def _create_coordinate_grids(self) -> None:
        """Create X, Y coordinate grids from extent."""
        if self._data is None or self._extent is None:
            return

        h, w = self._data.shape[:2]
        x = np.linspace(self._extent[0], self._extent[1], w)
        y = np.linspace(self._extent[2], self._extent[3], h)
        self._X_grid, self._Y_grid = np.meshgrid(x, y)

    def _on_param_slider_change(self, change: dict) -> None:
        """Handle parameter slider change - update preview contour."""
        self._update_fit_preview()

    def _update_fit_preview(self) -> None:
        """Update the fit preview contour with current parameter values."""
        if self._current_fit_model is None or self._ax is None or self._data is None:
            return

        if self._X_grid is None or self._Y_grid is None:
            self._create_coordinate_grids()

        # Get current parameter values from sliders
        params = {}
        for name, widgets_dict in self._fit_params_widgets.items():
            params[name] = widgets_dict["slider"].value

        try:
            # Evaluate model with current parameters
            preview_data = self._current_fit_model.func(self._X_grid, self._Y_grid, **params)

            # Clear existing preview
            self._clear_preview_contour()

            # Draw preview contour (dashed lines)
            levels = np.linspace(preview_data.min(), preview_data.max(), 8)
            self._preview_contour = self._ax.contour(
                self._X_grid,
                self._Y_grid,
                preview_data,
                levels=levels,
                colors="red",
                linewidths=1,
                linestyles="dashed",
                alpha=0.6,
            )
            self._fig.canvas.draw_idle()
        except Exception:
            pass

    def _clear_preview_contour(self) -> None:
        """Remove the preview contour from the plot."""
        if self._preview_contour is not None:
            with contextlib.suppress(ValueError, AttributeError):
                # Use the contour set's remove method (works in all matplotlib versions)
                self._preview_contour.remove()
            self._preview_contour = None
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    def _clear_fit_contour(self) -> None:
        """Remove the fit contour from the plot."""
        if self._fit_contour is not None:
            with contextlib.suppress(ValueError, AttributeError):
                # Use the contour set's remove method (works in all matplotlib versions)
                self._fit_contour.remove()
            self._fit_contour = None
            if self._fig is not None:
                self._fig.canvas.draw_idle()

    def _on_fit_click(self, button: widgets.Button) -> None:
        """Perform the 2D fit with current parameters."""
        if self._current_fit_model is None or self._data is None:
            return

        if self._X_grid is None or self._Y_grid is None:
            self._create_coordinate_grids()

        # Get initial parameters from sliders, apply bounds and fixed status
        init_params = {}
        fixed_params = {}
        param_bounds = {}

        for name, widgets_dict in self._fit_params_widgets.items():
            init_params[name] = widgets_dict["slider"].value

            # Store bounds
            param_bounds[name] = {
                "min": widgets_dict.get("bound_min", -np.inf),
                "max": widgets_dict.get("bound_max", np.inf),
            }

            # Store fixed status
            if widgets_dict["fix"].value:
                fixed_params[name] = True

        try:
            # Perform fit with custom parameter handling
            lmfit_model = self._current_fit_model.make_model()
            lmfit_params = lmfit_model.make_params(**init_params)

            # Apply bounds and fixed status
            for name, bounds in param_bounds.items():
                if name in lmfit_params:
                    if not np.isinf(bounds["min"]):
                        lmfit_params[name].min = bounds["min"]
                    if not np.isinf(bounds["max"]):
                        lmfit_params[name].max = bounds["max"]

            for name in fixed_params:
                if name in lmfit_params:
                    lmfit_params[name].vary = False

            # Perform fit
            X_flat = self._X_grid.ravel()
            Y_flat = self._Y_grid.ravel()
            data_flat = self._data.ravel()

            self._fit_result = lmfit_model.fit(data_flat, lmfit_params, X=X_flat, Y=Y_flat)

            # Update sliders with fitted values (only non-fixed parameters)
            for name, param in self._fit_result.params.items():
                if name in self._fit_params_widgets:
                    widgets_dict = self._fit_params_widgets[name]
                    slider = widgets_dict["slider"]
                    # Only update if not fixed
                    if not widgets_dict["fix"].value:
                        if param.value < slider.min:
                            slider.min = param.value - abs(param.value) * 0.5
                        if param.value > slider.max:
                            slider.max = param.value + abs(param.value) * 0.5
                        slider.value = param.value

            # Update fit contour if checkbox is checked
            if self._show_fit_contour.value:
                self._draw_fit_contour()

            # Display results
            result_text = self._format_fit_result()
            self._fit_result_html.value = result_text

        except Exception as e:
            self._fit_result_html.value = f'<span style="color: red;">Fit failed: {e}</span>'

    def _draw_fit_contour(self) -> None:
        """Draw contour lines showing the fit result."""
        if self._fit_result is None or self._ax is None:
            return

        self._clear_fit_contour()
        self._clear_preview_contour()

        # Evaluate fit on grid
        fit_data = self._fit_result.eval(X=self._X_grid.ravel(), Y=self._Y_grid.ravel())
        fit_data = fit_data.reshape(self._data.shape)

        # Get fit contour appearance settings
        contour_color = self._fit_contour_color_dropdown.value
        contour_linestyle = self._fit_contour_linestyle_dropdown.value

        # Draw contour
        levels = np.linspace(fit_data.min(), fit_data.max(), 10)
        self._fit_contour = self._ax.contour(
            self._X_grid,
            self._Y_grid,
            fit_data,
            levels=levels,
            colors=contour_color,
            linestyles=contour_linestyle,
            linewidths=1,
            alpha=0.7,
        )
        self._fig.canvas.draw_idle()

    def _on_show_fit_contour_change(self, change: dict) -> None:
        """Handle show fit contour checkbox change."""
        if change["new"] and self._fit_result is not None:
            self._draw_fit_contour()
        else:
            self._clear_fit_contour()
            self._clear_preview_contour()

    def _format_fit_result(self) -> str:
        """Format fit result as HTML."""
        if self._fit_result is None:
            return ""

        # Get model name
        model_name = (
            self._current_fit_model.name if self._current_fit_model else "Unknown"
        )

        lines = ['<div style="font-family: monospace; font-size: 12px;">']
        lines.append(f"<b>Model:</b> {model_name}<br>")
        lines.append(
            f"<b>χ²</b> = {self._fit_result.chisqr:.4g}, "
            f"<b>reduced χ²</b> = {self._fit_result.redchi:.4g}<br>"
        )
        lines.append("<b>Parameters:</b><br>")

        for name, param in self._fit_result.params.items():
            stderr = f"± {param.stderr:.4g}" if param.stderr else "± N/A"
            fixed_str = " (fixed)" if not param.vary else ""
            lines.append(f"  {name} = {param.value:.6g} {stderr}{fixed_str}<br>")

        lines.append("</div>")
        return "".join(lines)

    def _on_reset_guesses(self, button: widgets.Button) -> None:
        """Reset parameter sliders to automatic guesses."""
        self._auto_guess_parameters()

    def _update_slider_ranges(self) -> None:
        """Update slider min/max ranges based on data."""
        if self._extent is not None:
            x_min, x_max, y_min, y_max = self._extent

            # For heatmaps, limit sliders to exact data extent (no padding)
            # Update X slider
            if x_min < self._x_slider.min:
                self._x_slider.min = x_min
            if x_max > self._x_slider.max:
                self._x_slider.max = x_max
            self._x_slider.min = x_min
            self._x_slider.max = x_max

            # Update Y slider
            if y_min < self._y_slider.min:
                self._y_slider.min = y_min
            if y_max > self._y_slider.max:
                self._y_slider.max = y_max
            self._y_slider.min = y_min
            self._y_slider.max = y_max

        if self._data is not None:
            import numpy as np

            data_min = float(np.nanmin(self._data))
            data_max = float(np.nanmax(self._data))

            # Update color slider - limit to exact data range (no padding)
            if data_min < self._color_slider.min:
                self._color_slider.min = data_min
            if data_max > self._color_slider.max:
                self._color_slider.max = data_max
            self._color_slider.min = data_min
            self._color_slider.max = data_max

    def _update_slider_values(self) -> None:
        """Update slider values to fit the current data."""
        if self._extent is not None:
            x_min, x_max, y_min, y_max = self._extent
            self._x_slider.value = (x_min, x_max)
            self._y_slider.value = (y_min, y_max)

        if self._data is not None:
            import numpy as np

            self._color_slider.value = (
                float(np.nanmin(self._data)),
                float(np.nanmax(self._data)),
            )

    def _get_current_cmap(self) -> Any:
        """Get the current colormap object."""
        if self._cmap_dropdown.value == "custom" and self._custom_cmap is not None:
            return self._custom_cmap
        return self._cmap_dropdown.value

    def _redraw(self) -> None:
        """Redraw the heatmap with current settings."""
        if self._fig is None or self._data is None:
            return

        self._ax.clear()

        # Remove existing colorbar if present
        if self._colorbar is not None:
            with contextlib.suppress(AttributeError, ValueError):
                self._colorbar.remove()
            self._colorbar = None

        # Draw the image
        vmin, vmax = self._color_slider.value
        self._im = self._ax.imshow(
            self._data,
            extent=self._extent,
            origin="lower",
            aspect=self._aspect,
            interpolation=self._interpolation,
            cmap=self._get_current_cmap(),
            vmin=vmin,
            vmax=vmax,
        )

        # Add colorbar
        if self._show_colorbar:
            self._colorbar = self._fig.colorbar(self._im, ax=self._ax)
            if self._colorbar_label:
                self._colorbar.set_label(self._colorbar_label)

        # Apply settings
        self._ax.set_xlim(self._x_slider.value)
        self._ax.set_ylim(self._y_slider.value)
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)
        if self._title:
            self._ax.set_title(self._title)

        if self._grid_checkbox.value:
            self._ax.grid(True, alpha=0.3)

        self._fig.canvas.draw_idle()

    def _on_xlim_change(self, change: dict) -> None:
        """Handle X-axis range slider change."""
        if self._ax is not None:
            self._ax.set_xlim(change["new"])
            self._fig.canvas.draw_idle()

    def _on_ylim_change(self, change: dict) -> None:
        """Handle Y-axis range slider change."""
        if self._ax is not None:
            self._ax.set_ylim(change["new"])
            self._fig.canvas.draw_idle()

    def _on_color_range_change(self, change: dict) -> None:
        """Handle color range slider change."""
        if self._im is not None:
            vmin, vmax = change["new"]
            self._im.set_clim(vmin, vmax)
            self._fig.canvas.draw_idle()

    def _on_cmap_change(self, change: dict) -> None:
        """Handle colormap dropdown change."""
        if self._im is not None:
            self._im.set_cmap(self._get_current_cmap())
            self._fig.canvas.draw_idle()

    def _on_grid_change(self, change: dict) -> None:
        """Handle grid checkbox change."""
        if self._ax is not None:
            if change["new"]:
                self._ax.grid(True, alpha=0.3)
            else:
                self._ax.grid(False)
                # Also hide any existing grid lines
                for line in self._ax.xaxis.get_gridlines():
                    line.set_visible(False)
                for line in self._ax.yaxis.get_gridlines():
                    line.set_visible(False)
            self._fig.canvas.draw_idle()

    def _on_save_click(self, button: widgets.Button) -> None:
        """Handle save button click."""
        import os

        filepath = self._filepath_text.value.strip()
        filename = self._filename_text.value.strip() or "heatmap"
        fmt = self._format_dropdown.value
        dpi = max(1, self._dpi_text.value)

        full_path = os.path.join(filepath, filename + fmt) if filepath else filename + fmt

        try:
            self._fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
            self._save_status.value = f'<span style="color: green;">✓ Saved: {full_path}</span>'
        except Exception as e:
            self._save_status.value = f'<span style="color: red;">✗ Error: {e}</span>'

    def set_data(
        self,
        data,
        extent: tuple[float, float, float, float] | None = None,
    ) -> None:
        """
        Set the 2D data for the heatmap.

        Args:
            data: 2D array-like data to display
            extent: The extent of the image (x_min, x_max, y_min, y_max).
                    If None, uses array indices.
        """
        import numpy as np

        self._data = np.asarray(data)

        if extent is not None:
            self._extent = extent
            self._x_min, self._x_max = extent[0], extent[1]
            self._y_min, self._y_max = extent[2], extent[3]
        else:
            # Use array indices
            h, w = self._data.shape[:2]
            self._extent = (0, w, 0, h)
            self._x_min, self._x_max = 0, w
            self._y_min, self._y_max = 0, h

        self._vmin = float(np.nanmin(self._data))
        self._vmax = float(np.nanmax(self._data))

        # Update sliders if initialized
        if self._initialized:
            self._update_slider_ranges()
            self._update_slider_values()
            self._redraw()

    def set_cmap(self, cmap: str | Any) -> None:
        """
        Set a custom colormap.

        Args:
            cmap: Colormap name (string) or custom matplotlib colormap object
        """
        if isinstance(cmap, str):
            self._custom_cmap = None
            self._cmap_name = cmap
            if cmap in self.COLORMAPS:
                self._cmap_dropdown.value = cmap
        else:
            self._custom_cmap = cmap
            self._cmap_name = "custom"
            # Add "custom" to dropdown if not present
            if "custom" not in self._cmap_dropdown.options:
                self._cmap_dropdown.options = ["custom"] + list(self._cmap_dropdown.options)
            self._cmap_dropdown.value = "custom"

        if self._im is not None:
            self._im.set_cmap(self._get_current_cmap())
            self._fig.canvas.draw_idle()

    def set_title(self, title: str) -> None:
        """Set the plot title."""
        self._title = title
        if self._ax is not None:
            self._ax.set_title(title)
            self._fig.canvas.draw_idle()

    def set_xlabel(self, label: str) -> None:
        """Set the X-axis label."""
        self._xlabel = label
        if self._ax is not None:
            self._ax.set_xlabel(label)
            self._fig.canvas.draw_idle()

    def set_ylabel(self, label: str) -> None:
        """Set the Y-axis label."""
        self._ylabel = label
        if self._ax is not None:
            self._ax.set_ylabel(label)
            self._fig.canvas.draw_idle()

    def set_colorbar_label(self, label: str) -> None:
        """Set the colorbar label."""
        self._colorbar_label = label
        if self._colorbar is not None:
            self._colorbar.set_label(label)
            self._fig.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the heatmap data."""
        self._data = None
        self._extent = None
        self._vmin = 0.0
        self._vmax = 1.0

    def show(self) -> widgets.Widget:
        """
        Display the interactive heatmap.

        Returns:
            The main widget layout for display in Jupyter.
        """
        # Update slider ranges based on data
        self._update_slider_ranges()

        # Set slider values to fit data
        if self._extent is not None:
            x_min, x_max, y_min, y_max = self._extent
            self._x_slider.value = (x_min, x_max)
            self._y_slider.value = (y_min, y_max)

        if self._data is not None:
            import numpy as np

            self._color_slider.value = (
                float(np.nanmin(self._data)),
                float(np.nanmax(self._data)),
            )

        # Create and display the figure
        with self._output:
            clear_output(wait=True)
            self._fig = plt.figure(figsize=self._figsize)
            self._ax = self._fig.add_subplot(111)
            self._redraw()
            plt.show()
            self._initialized = True

        return self._layout

    @property
    def figure(self) -> matplotlib.figure.Figure | None:
        """Get the underlying matplotlib figure."""
        return self._fig

    @property
    def ax(self) -> Any:
        """Get the underlying matplotlib axes."""
        return self._ax

    @property
    def image(self) -> Any:
        """Get the underlying matplotlib AxesImage object."""
        return self._im

    @property
    def fit_result(self) -> Any:
        """Get the last fit result (lmfit ModelResult object)."""
        return self._fit_result

    def set_custom_fit_model(self, model: FitModel2D) -> None:
        """
        Set a custom fit model.

        Args:
            model: FitModel2D object for custom fitting
        """
        self._custom_fit_model = model
        # Update dropdown options
        model_options = list(MODELS_2D.keys())
        model_options = ["Custom"] + model_options
        self._fit_model_dropdown.options = model_options


# =============================================================================
# Export all public classes and functions
# =============================================================================

__all__ = [
    # Widget configurations
    "IntSlider",
    "FloatSlider",
    "IntRangeSlider",
    "FloatRangeSlider",
    "FloatLogSlider",
    "IntText",
    "FloatText",
    "BoundedIntText",
    "BoundedFloatText",
    "Checkbox",
    "ToggleButton",
    "Dropdown",
    "RadioButtons",
    "Select",
    "SelectMultiple",
    "ToggleButtons",
    "SelectionSlider",
    "Text",
    "Textarea",
    # Functions
    "interactive_plot",
    "interactive_plotting",
    "create_widget",
    # Interactive Figure Classes
    "InteractiveXYPlot",
    "InteractiveHeatmap",
    # Fitting
    "FitModel1D",
    "FitModel2D",
    "MODELS_1D",
    "MODELS_2D",
    "create_custom_model_1d",
    "create_custom_model_2d",
    "fit_1d",
    "fit_2d",
]
