"""Tests for widget creation from annotations."""

import inspect
from typing import Annotated

import ipywidgets as widgets

from interactive_figure import (
    BoundedFloatText,
    BoundedIntText,
    Checkbox,
    Dropdown,
    FloatLogSlider,
    FloatRangeSlider,
    FloatSlider,
    FloatText,
    IntRangeSlider,
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
    create_widget,
)


class TestCreateWidgetWithAnnotations:
    """Tests for create_widget with Annotated types."""

    def test_int_slider_creation(self) -> None:
        annotation = Annotated[int, IntSlider(min=0, max=50, step=2)]
        widget = create_widget("test_param", annotation, 10)

        assert isinstance(widget, widgets.IntSlider)
        assert widget.value == 10
        assert widget.min == 0
        assert widget.max == 50
        assert widget.step == 2
        assert widget.description == "test_param"

    def test_int_slider_with_custom_description(self) -> None:
        annotation = Annotated[int, IntSlider(min=0, max=100, description="Custom Label")]
        widget = create_widget("test_param", annotation, 25)

        assert widget.description == "Custom Label"

    def test_float_slider_creation(self) -> None:
        annotation = Annotated[float, FloatSlider(min=0.0, max=10.0, step=0.5)]
        widget = create_widget("amplitude", annotation, 5.0)

        assert isinstance(widget, widgets.FloatSlider)
        assert widget.value == 5.0
        assert widget.min == 0.0
        assert widget.max == 10.0
        assert widget.step == 0.5

    def test_float_slider_default_from_config_when_empty(self) -> None:
        annotation = Annotated[float, FloatSlider(min=1.0, max=10.0)]
        widget = create_widget("param", annotation, inspect.Parameter.empty)

        assert widget.value == 1.0  # Should use min as default

    def test_int_range_slider_creation(self) -> None:
        annotation = Annotated[tuple[int, int], IntRangeSlider(min=0, max=100)]
        widget = create_widget("range", annotation, (20, 80))

        assert isinstance(widget, widgets.IntRangeSlider)
        assert widget.value == (20, 80)
        assert widget.min == 0
        assert widget.max == 100

    def test_float_range_slider_creation(self) -> None:
        annotation = Annotated[tuple[float, float], FloatRangeSlider(min=0.0, max=1.0)]
        widget = create_widget("range", annotation, (0.2, 0.8))

        assert isinstance(widget, widgets.FloatRangeSlider)
        assert widget.value == (0.2, 0.8)

    def test_float_log_slider_creation(self) -> None:
        annotation = Annotated[float, FloatLogSlider(min=-2, max=2, base=10.0)]
        widget = create_widget("scale", annotation, 1.0)

        assert isinstance(widget, widgets.FloatLogSlider)
        assert widget.value == 1.0
        assert widget.base == 10.0

    def test_int_text_creation(self) -> None:
        annotation = Annotated[int, IntText()]
        widget = create_widget("count", annotation, 42)

        assert isinstance(widget, widgets.IntText)
        assert widget.value == 42

    def test_float_text_creation(self) -> None:
        annotation = Annotated[float, FloatText()]
        widget = create_widget("value", annotation, 3.14)

        assert isinstance(widget, widgets.FloatText)
        assert widget.value == 3.14

    def test_bounded_int_text_creation(self) -> None:
        annotation = Annotated[int, BoundedIntText(min=0, max=10)]
        widget = create_widget("bounded", annotation, 5)

        assert isinstance(widget, widgets.BoundedIntText)
        assert widget.value == 5
        assert widget.min == 0
        assert widget.max == 10

    def test_bounded_float_text_creation(self) -> None:
        annotation = Annotated[float, BoundedFloatText(min=0.0, max=1.0)]
        widget = create_widget("bounded", annotation, 0.5)

        assert isinstance(widget, widgets.BoundedFloatText)
        assert widget.value == 0.5

    def test_checkbox_creation(self) -> None:
        annotation = Annotated[bool, Checkbox()]
        widget = create_widget("enabled", annotation, True)

        assert isinstance(widget, widgets.Checkbox)
        assert widget.value is True

    def test_toggle_button_creation(self) -> None:
        annotation = Annotated[bool, ToggleButton(button_style="success", icon="check")]
        widget = create_widget("toggle", annotation, False)

        assert isinstance(widget, widgets.ToggleButton)
        assert widget.value is False
        assert widget.button_style == "success"

    def test_dropdown_creation(self) -> None:
        annotation = Annotated[str, Dropdown(options=["red", "green", "blue"])]
        widget = create_widget("color", annotation, "green")

        assert isinstance(widget, widgets.Dropdown)
        assert widget.value == "green"
        assert list(widget.options) == ["red", "green", "blue"]

    def test_dropdown_default_first_option(self) -> None:
        annotation = Annotated[str, Dropdown(options=["a", "b", "c"])]
        widget = create_widget("choice", annotation, inspect.Parameter.empty)

        assert widget.value == "a"

    def test_radio_buttons_creation(self) -> None:
        annotation = Annotated[str, RadioButtons(options=["opt1", "opt2"])]
        widget = create_widget("radio", annotation, "opt2")

        assert isinstance(widget, widgets.RadioButtons)
        assert widget.value == "opt2"

    def test_select_creation(self) -> None:
        annotation = Annotated[str, Select(options=["a", "b", "c"], rows=3)]
        widget = create_widget("select", annotation, "b")

        assert isinstance(widget, widgets.Select)
        assert widget.value == "b"
        assert widget.rows == 3

    def test_select_multiple_creation(self) -> None:
        annotation = Annotated[tuple, SelectMultiple(options=["a", "b", "c"])]
        widget = create_widget("multi", annotation, ("a", "c"))

        assert isinstance(widget, widgets.SelectMultiple)
        assert widget.value == ("a", "c")

    def test_toggle_buttons_creation(self) -> None:
        annotation = Annotated[str, ToggleButtons(options=["left", "center", "right"])]
        widget = create_widget("align", annotation, "center")

        assert isinstance(widget, widgets.ToggleButtons)
        assert widget.value == "center"

    def test_selection_slider_creation(self) -> None:
        annotation = Annotated[str, SelectionSlider(options=["low", "medium", "high"])]
        widget = create_widget("level", annotation, "medium")

        assert isinstance(widget, widgets.SelectionSlider)
        assert widget.value == "medium"

    def test_text_creation(self) -> None:
        annotation = Annotated[str, Text(placeholder="Enter name...")]
        widget = create_widget("name", annotation, "Alice")

        assert isinstance(widget, widgets.Text)
        assert widget.value == "Alice"
        assert widget.placeholder == "Enter name..."

    def test_textarea_creation(self) -> None:
        annotation = Annotated[str, Textarea(rows=10)]
        widget = create_widget("description", annotation, "Hello\nWorld")

        assert isinstance(widget, widgets.Textarea)
        assert widget.value == "Hello\nWorld"
        assert widget.rows == 10


class TestCreateWidgetTypeInference:
    """Tests for automatic widget inference from base types."""

    def test_infer_int_slider(self) -> None:
        widget = create_widget("number", int, 50)

        assert isinstance(widget, widgets.IntSlider)
        assert widget.value == 50

    def test_infer_float_slider(self) -> None:
        widget = create_widget("decimal", float, 0.5)

        assert isinstance(widget, widgets.FloatSlider)
        assert widget.value == 0.5

    def test_infer_bool_checkbox(self) -> None:
        widget = create_widget("flag", bool, True)

        assert isinstance(widget, widgets.Checkbox)
        assert widget.value is True

    def test_infer_str_text(self) -> None:
        widget = create_widget("text", str, "hello")

        assert isinstance(widget, widgets.Text)
        assert widget.value == "hello"

    def test_infer_int_tuple_range_slider(self) -> None:
        widget = create_widget("range", tuple[int, int], (10, 90))

        assert isinstance(widget, widgets.IntRangeSlider)
        assert widget.value == (10, 90)

    def test_infer_float_tuple_range_slider(self) -> None:
        widget = create_widget("range", tuple[float, float], (0.2, 0.8))

        assert isinstance(widget, widgets.FloatRangeSlider)
        assert widget.value == (0.2, 0.8)

    def test_fallback_to_text_for_unknown_type(self) -> None:
        widget = create_widget("unknown", dict, {"key": "value"})

        assert isinstance(widget, widgets.Text)
        assert widget.value == "{'key': 'value'}"
