"""Tests for widget configuration dataclasses."""


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
)


class TestIntSlider:
    """Tests for IntSlider configuration."""

    def test_default_values(self) -> None:
        slider = IntSlider()
        assert slider.min == 0
        assert slider.max == 100
        assert slider.step == 1
        assert slider.description == ""
        assert slider.continuous_update is True
        assert slider.orientation == "horizontal"
        assert slider.readout is True

    def test_custom_values(self) -> None:
        slider = IntSlider(min=-10, max=50, step=5, description="My Slider")
        assert slider.min == -10
        assert slider.max == 50
        assert slider.step == 5
        assert slider.description == "My Slider"


class TestFloatSlider:
    """Tests for FloatSlider configuration."""

    def test_default_values(self) -> None:
        slider = FloatSlider()
        assert slider.min == 0.0
        assert slider.max == 1.0
        assert slider.step == 0.01
        assert slider.readout_format == ".2f"

    def test_custom_values(self) -> None:
        slider = FloatSlider(min=-1.0, max=10.0, step=0.1, readout_format=".3f")
        assert slider.min == -1.0
        assert slider.max == 10.0
        assert slider.step == 0.1
        assert slider.readout_format == ".3f"


class TestIntRangeSlider:
    """Tests for IntRangeSlider configuration."""

    def test_default_values(self) -> None:
        slider = IntRangeSlider()
        assert slider.min == 0
        assert slider.max == 100
        assert slider.step == 1


class TestFloatRangeSlider:
    """Tests for FloatRangeSlider configuration."""

    def test_default_values(self) -> None:
        slider = FloatRangeSlider()
        assert slider.min == 0.0
        assert slider.max == 1.0
        assert slider.step == 0.01
        assert slider.readout_format == ".2f"


class TestFloatLogSlider:
    """Tests for FloatLogSlider configuration."""

    def test_default_values(self) -> None:
        slider = FloatLogSlider()
        assert slider.min == -2
        assert slider.max == 2
        assert slider.base == 10.0
        assert slider.readout_format == ".3e"


class TestTextInputs:
    """Tests for text input configurations."""

    def test_int_text_defaults(self) -> None:
        widget = IntText()
        assert widget.description == ""
        assert widget.continuous_update is False

    def test_float_text_defaults(self) -> None:
        widget = FloatText()
        assert widget.description == ""
        assert widget.continuous_update is False

    def test_bounded_int_text_defaults(self) -> None:
        widget = BoundedIntText()
        assert widget.min == 0
        assert widget.max == 100
        assert widget.step == 1

    def test_bounded_float_text_defaults(self) -> None:
        widget = BoundedFloatText()
        assert widget.min == 0.0
        assert widget.max == 1.0
        assert widget.step == 0.01


class TestBooleanWidgets:
    """Tests for boolean widget configurations."""

    def test_checkbox_defaults(self) -> None:
        widget = Checkbox()
        assert widget.description == ""
        assert widget.indent is True

    def test_toggle_button_defaults(self) -> None:
        widget = ToggleButton()
        assert widget.description == ""
        assert widget.button_style == ""
        assert widget.icon == "check"


class TestSelectionWidgets:
    """Tests for selection widget configurations."""

    def test_dropdown_defaults(self) -> None:
        widget = Dropdown()
        assert widget.options == []
        assert widget.description == ""

    def test_dropdown_with_options(self) -> None:
        widget = Dropdown(options=["a", "b", "c"])
        assert widget.options == ["a", "b", "c"]

    def test_radio_buttons_defaults(self) -> None:
        widget = RadioButtons()
        assert widget.options == []

    def test_select_defaults(self) -> None:
        widget = Select()
        assert widget.options == []
        assert widget.rows == 5

    def test_select_multiple_defaults(self) -> None:
        widget = SelectMultiple()
        assert widget.options == []
        assert widget.rows == 5

    def test_toggle_buttons_defaults(self) -> None:
        widget = ToggleButtons()
        assert widget.options == []
        assert widget.button_style == ""

    def test_selection_slider_defaults(self) -> None:
        widget = SelectionSlider()
        assert widget.options == []
        assert widget.continuous_update is True
        assert widget.orientation == "horizontal"


class TestStringWidgets:
    """Tests for string widget configurations."""

    def test_text_defaults(self) -> None:
        widget = Text()
        assert widget.placeholder == ""
        assert widget.description == ""
        assert widget.continuous_update is True

    def test_text_custom(self) -> None:
        widget = Text(placeholder="Enter text...", description="Name")
        assert widget.placeholder == "Enter text..."
        assert widget.description == "Name"

    def test_textarea_defaults(self) -> None:
        widget = Textarea()
        assert widget.placeholder == ""
        assert widget.rows == 5
        assert widget.continuous_update is True
