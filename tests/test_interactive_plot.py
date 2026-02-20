"""Tests for interactive_plot function."""

from typing import Annotated

import ipywidgets as widgets
import matplotlib.figure

from interactive_figure import (
    Checkbox,
    Dropdown,
    FloatSlider,
    IntSlider,
    interactive_plot,
)


class TestInteractivePlot:
    """Tests for the interactive_plot function."""

    def test_returns_widget(self) -> None:
        """Test that interactive_plot returns a widget container."""

        def simple_plot(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(simple_plot)
        assert isinstance(result, widgets.VBox)

    def test_creates_widgets_from_annotations(self) -> None:
        """Test that widgets are created based on parameter annotations."""

        def plot_with_params(
            fig: matplotlib.figure.Figure,
            amplitude: Annotated[float, FloatSlider(min=0, max=10)] = 1.0,
            count: Annotated[int, IntSlider(min=0, max=100)] = 50,
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_with_params)

        # The first child should be the controls VBox
        controls = result.children[0]
        assert isinstance(controls, widgets.VBox)

        # Should have 2 widgets (amplitude and count)
        assert len(controls.children) == 2

    def test_skips_fig_parameter(self) -> None:
        """Test that the fig parameter is skipped when creating widgets."""

        def plot_func(
            fig: matplotlib.figure.Figure,
            value: Annotated[int, IntSlider(min=0, max=10)] = 5,
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func)
        controls = result.children[0]

        # Should only have 1 widget (value), not fig
        assert len(controls.children) == 1

    def test_horizontal_layout(self) -> None:
        """Test horizontal widget layout."""

        def plot_func(
            fig: matplotlib.figure.Figure,
            a: Annotated[int, IntSlider()] = 1,
            b: Annotated[int, IntSlider()] = 2,
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func, widget_layout="horizontal")
        controls = result.children[0]

        assert isinstance(controls, widgets.HBox)

    def test_grid_layout(self) -> None:
        """Test grid widget layout."""

        def plot_func(
            fig: matplotlib.figure.Figure,
            a: Annotated[int, IntSlider()] = 1,
            b: Annotated[int, IntSlider()] = 2,
            c: Annotated[int, IntSlider()] = 3,
            d: Annotated[int, IntSlider()] = 4,
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func, widget_layout="grid")
        controls = result.children[0]

        # Grid layout creates rows of HBoxes in a VBox
        assert isinstance(controls, widgets.VBox)
        # With 4 widgets and 2 per row, should have 2 rows
        assert len(controls.children) == 2
        assert all(isinstance(row, widgets.HBox) for row in controls.children)

    def test_custom_widget_width(self) -> None:
        """Test custom widget container width."""

        def plot_func(
            fig: matplotlib.figure.Figure,
            value: Annotated[int, IntSlider()] = 5,
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func, widget_width="600px")
        controls = result.children[0]

        assert controls.layout.width == "600px"

    def test_multiple_widget_types(self) -> None:
        """Test that different widget types are created correctly."""

        def plot_func(
            fig: matplotlib.figure.Figure,
            amplitude: Annotated[float, FloatSlider(min=0, max=10)] = 1.0,
            show_grid: Annotated[bool, Checkbox()] = True,
            color: Annotated[str, Dropdown(options=["red", "blue"])] = "red",
        ) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func)
        controls = result.children[0]

        assert len(controls.children) == 3

        # Check widget types
        widget_types = [type(w).__name__ for w in controls.children]
        assert "FloatSlider" in widget_types
        assert "Checkbox" in widget_types
        assert "Dropdown" in widget_types

    def test_output_widget_exists(self) -> None:
        """Test that an Output widget is created for the plot."""

        def plot_func(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
            return fig

        result = interactive_plot(plot_func)

        # Second child should be the output widget
        assert len(result.children) == 2
        assert isinstance(result.children[1], widgets.Output)

    def test_no_parameters_except_fig(self) -> None:
        """Test plot function with only fig parameter."""

        def minimal_plot(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 2, 3])
            return fig

        result = interactive_plot(minimal_plot)
        controls = result.children[0]

        # Should have no widgets
        assert len(controls.children) == 0


class TestInteractivePlotAlias:
    """Test that interactive_plotting is an alias for interactive_plot."""

    def test_alias_exists(self) -> None:
        from interactive_figure import interactive_plot, interactive_plotting

        assert interactive_plotting is interactive_plot
