# iplot-mpl

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Create interactive matplotlib figures with ipywidgets in Jupyter Notebooks using type annotations.

## Features

- 🎛️ **Declarative widget creation** - Define widgets using Python type annotations
- 🔄 **Automatic updates** - Figures update instantly when widget values change
- 🎨 **Multiple widget types** - Sliders, dropdowns, checkboxes, text inputs, and more
- 📐 **Flexible layouts** - Vertical, horizontal, or grid arrangements
- 🔧 **Fully customizable** - Configure every widget parameter

## Installation

> **Note:** This package is currently in development and not yet available on PyPI.
> For now, install from TestPyPI or directly from source.

### From TestPyPI (testing)

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ iplot-mpl
```

### From Source (recommended for now)

```bash
git clone https://github.com/timduka/iplot-mpl.git
cd iplot-mpl
pip install -e ".[dev]"
```

## Quick Start

```python
%matplotlib widget

import numpy as np
import matplotlib.figure
from typing import Annotated
from interactive_figure import interactive_plot, FloatSlider, FloatRangeSlider, Checkbox

def plot_sine(
    fig: matplotlib.figure.Figure,
    amplitude: Annotated[float, FloatSlider(min=0.1, max=5.0, step=0.1)] = 1.0,
    frequency: Annotated[float, FloatSlider(min=0.1, max=5.0, step=0.1)] = 1.0,
    x_range: Annotated[tuple[float, float], FloatRangeSlider(min=0, max=10)] = (0.0, 6.28),
    show_grid: Annotated[bool, Checkbox()] = True,
) -> matplotlib.figure.Figure:
    ax = fig.add_subplot(111)
    
    x = np.linspace(x_range[0], x_range[1], 200)
    y = amplitude * np.sin(frequency * x)
    
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'y = {amplitude:.1f} × sin({frequency:.1f}x)')
    ax.grid(show_grid, alpha=0.3)
    
    return fig

# Create the interactive plot
interactive_plot(plot_sine, figsize=(10, 6))
```

## Available Widget Types

### Numeric Sliders

| Widget | Description | Key Parameters |
|--------|-------------|----------------|
| `IntSlider` | Integer slider | `min`, `max`, `step` |
| `FloatSlider` | Float slider | `min`, `max`, `step`, `readout_format` |
| `IntRangeSlider` | Integer range (min, max) | `min`, `max`, `step` |
| `FloatRangeSlider` | Float range (min, max) | `min`, `max`, `step` |
| `FloatLogSlider` | Logarithmic slider | `min`, `max`, `base` |

### Numeric Text Inputs

| Widget | Description |
|--------|-------------|
| `IntText` | Integer text input |
| `FloatText` | Float text input |
| `BoundedIntText` | Bounded integer input |
| `BoundedFloatText` | Bounded float input |

### Boolean Widgets

| Widget | Description |
|--------|-------------|
| `Checkbox` | Standard checkbox |
| `ToggleButton` | Toggle button with icon |

### Selection Widgets

| Widget | Description |
|--------|-------------|
| `Dropdown` | Dropdown menu |
| `RadioButtons` | Radio button group |
| `Select` | Scrollable list |
| `SelectMultiple` | Multi-select list |
| `ToggleButtons` | Button group |
| `SelectionSlider` | Slider with discrete options |

### String Widgets

| Widget | Description |
|--------|-------------|
| `Text` | Single-line text input |
| `Textarea` | Multi-line text input |

## Advanced Usage

### Custom Widget Layout

```python
interactive_plot(
    plot_func,
    figsize=(12, 8),
    widget_layout="horizontal",  # "vertical", "horizontal", or "grid"
    widget_width="600px"
)
```

### Selection Widget Example

```python
from interactive_figure import Dropdown, ToggleButtons

def plot_function(
    fig: matplotlib.figure.Figure,
    function: Annotated[str, Dropdown(options=["sin", "cos", "tan"])] = "sin",
    style: Annotated[str, ToggleButtons(options=["solid", "dashed", "dotted"])] = "solid",
) -> matplotlib.figure.Figure:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 2 * np.pi, 100)
    
    func_map = {"sin": np.sin, "cos": np.cos, "tan": np.tan}
    style_map = {"solid": "-", "dashed": "--", "dotted": ":"}
    
    ax.plot(x, func_map[function](x), linestyle=style_map[style])
    return fig
```

## Requirements

- Python ≥ 3.9
- matplotlib ≥ 3.5.0
- ipywidgets ≥ 8.0.0
- IPython ≥ 7.0.0

For interactive matplotlib in Jupyter, you'll also need:
- ipympl (`%matplotlib widget` backend)

## Development

### Prerequisites

- Python 3.9+
- [just](https://github.com/casey/just) command runner (optional but recommended)

```bash
# Windows installation
winget install Casey.Just
```

### Setup

```bash
# Clone and install in development mode
git clone https://github.com/timduka/iplot-mpl.git
cd iplot-mpl

# Using just (recommended)
just install

# Or manually
pip install -e ".[dev]"
```

### Available Commands

This project uses `just` as a command runner. Run `just` to see all available commands:

| Command | Description |
|---------|-------------|
| `just install` | Install package with dev dependencies |
| `just test` | Run tests with coverage |
| `just test-fast` | Run tests without coverage |
| `just lint` | Run linter checks (ruff) |
| `just format` | Format code with ruff |
| `just typecheck` | Run type checker (mypy) |
| `just ci` | Run all CI checks (lint + typecheck + test) |
| `just clean` | Clean build artifacts |
| `just build` | Build package for distribution |

### Running Tests

```bash
# Run all tests with coverage
just test

# Run tests without coverage (faster)
just test-fast

# Run specific test file
.venv/Scripts/python -m pytest tests/test_widget_creation.py -v

# Run tests matching a pattern
.venv/Scripts/python -m pytest -k "test_slider" -v
```

### Code Quality

```bash
# Check code style and lint
just lint

# Auto-format code
just format

# Type checking
just typecheck

# Run all checks (recommended before committing)
just ci
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run `just ci` to ensure all checks pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
