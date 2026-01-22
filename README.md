# Hayai

**Rapid Video Projection Mapping for Everyone**

Hayai is a fast, free, and beginner-friendly projection mapping tool with perspective-correct texture warping. The name "hayai" (速い) means "fast" in Japanese-reflecting both its quick setup time and real-time performance suitable for live shows.

## Features

- **Beginner-Friendly** - Intuitive UI with on-screen hints and tooltips
- **Fast Setup** - Create and map shapes in seconds, not hours
- **Live Performance Ready** - GPU-accelerated OpenGL rendering for smooth real-time playback
- **Perspective Correction** - Advanced inverse bilinear interpolation for accurate texture warping
- **Animation Support** - Load animated GIFs directly onto shapes
- **HSV Color Control** - Adjust hue, saturation, value, and alpha per shape
- **Groups & Hierarchy** - Organize shapes into groups for easier management
- **100% Free** - Open source under Creative Commons BY-SA 4.0

## Installation

### Requirements

- Python 3.10 or higher
- Windows (macOS/Linux support not yet tested)

### Quick Start

1. **Install Python** (if you don't have it):
   - Download Python 3.10+ from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Clone or download** this repository

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

4. **Activate the virtual environment**:
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run Hayai**:
   ```bash
   python hayai.py
   ```

### Dependencies

- pygame
- PyOpenGL + PyOpenGL-accelerate
- Pillow
- NumPy

## Usage

Hayai is designed for video projection mapping where the projector displays onto physical surfaces. The application starts in windowed mode-press **F11** for fullscreen when projecting.

### Quick Start Guide

1. **Create a shape**: Click "Freeform" or "Regular" to start creating shapes
2. **Edit the warp**: Select the shape and click "Edit Warp" mode
3. **Add an image**: Click the "Image" button (below Edit Warp) to load a texture or GIF
4. **Adjust the warp**: Drag corners for simple perspective correction, use perspective sliders for more extreme accuracy
5. **Go live**: Press **SPACE** to hide the UI and show only your mapped content

### Operating Modes

| Mode | Description |
|------|-------------|
| **Freeform** | Click to place vertices, click near start or press ENTER to close |
| **Regular** | Click to place regular polygons (3-120 sides) |
| **Move Shape** | Select, move, rotate, scale, and manage shapes |
| **Edit Shape** | Add, move, or delete individual vertices |
| **Edit Warp** | Adjust the 4-corner perspective warp |

### Keyboard Shortcuts

#### Global
| Key | Action |
|-----|--------|
| `SPACE` | Toggle UI visibility (play mode) |
| `F11` | Toggle fullscreen |
| `ESC` | Exit fullscreen |
| `H` | Toggle hierarchy panel |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+C` | Copy selected |
| `Ctrl+V` | Paste |
| `Ctrl+G` | Group selected shapes |
| `Ctrl+U` / `Ctrl+Shift+G` | Ungroup |
| `F2` | Rename selected item |
| `TAB` / `Shift+TAB` | Navigate UI buttons |
| `Arrow Keys` | Move selection (Shift = 10x speed) |

#### Move Shape Mode
| Input | Action |
|-------|--------|
| Click | Select shape (selects entire group) |
| Shift+Click | Select individual shape within group |
| Ctrl+Click | Add to selection |
| Drag | Move selection |
| Right-drag | Rotate selection |
| Mouse wheel | Scale selection (Ctrl=1%, Shift=10%) |
| `DEL` | Delete selected |
| `fx` / `fy` buttons | Flip horizontal / vertical |

#### Edit Shape Mode
| Input | Action |
|-------|--------|
| Click vertex | Select vertex |
| Drag vertex | Move vertex |
| Click edge | Add new vertex |
| `DEL` | Delete selected vertex |
| Arrow keys | Move vertex precisely |

#### Edit Warp Mode
| Input | Action |
|-------|--------|
| Drag corner | Adjust warp point |
| Right-drag | Rotate warp only |
| Mouse wheel | Adjust perspective amount |
| Middle-click | Switch perspective axis (X/Y) |
| Arrow keys | Move all warp points |
| `Image` button | Load image or GIF for shape |
| `Fit` button | Reset warp corners to shape bounds |

### Mouse Controls Summary

| Button | Action |
|--------|--------|
| Left click | Select / Place / Drag |
| Right drag | Rotate |
| Middle click | Switch perspective axis (Edit Warp mode) |
| Scroll wheel | Scale (Move mode) / Perspective (Warp mode) |

### Properties Panel

The right-side panel adapts to your selection:

**Single Shape:**
- **Name** - Editable shape name
- **Image** - Shows loaded image filename
- **Alpha** - Transparency (0 = invisible, 1 = opaque)
- **Hue** - Rotate colors around the color wheel (0-360°)
- **Saturation** - Color intensity (0 = grayscale, 2 = oversaturated)
- **Brightness** - Brightness level (0 = black, 2 = overbright)
- **Anim Speed** - Animation playback speed for GIFs
- **Perspective X/Y** - Fine-tune perspective distortion

**Group:** Shows name and transform info (position, rotation, scale)

**Multiple Items:** Shows selection count and lists selected item names with type indicators

### Display Options

- **Geometry** - Show/hide shape outlines and warp corners
- **Mask** - Enable/disable shape masking (clipping to contour)
- **Cursor** - Show/hide system cursor
- **Crosshair** - Show/hide cursor crosshair overlay
- **Grid** - Show/hide background grid
- **Controls** - Show/hide the controls help panel

### File Operations

- **Save Scene** - Save project as `.hayai` file (JSON format)
- **Load Scene** - Open a saved project
- **New Scene** - Clear everything and start fresh
- **Drag & Drop** - Drop images directly onto shapes, or drop `.hayai` files to load

### Supported File Formats

| Type | Formats |
|------|---------|
| Images | PNG, JPG, JPEG, GIF (animated), BMP |
| Projects | .hayai, .json |

## Tips for Live Performance

1. **Prepare your scene** ahead of time and save it
2. **Use F11** to go fullscreen on your projector output
3. **Press SPACE** to hide all UI elements during the show
4. **Use groups** to move multiple shapes together
5. **Animated GIFs** loop automatically-great for dynamic content

## Projection Mapping Workflow

1. **Physical Setup**: Position your projector aimed at the target surface(s)
2. **Create Shapes**: Trace the edges of physical surfaces with Freeform shapes, or use Regular shapes for geometric objects
3. **Load Content**: Add images or animations to each shape
4. **Warp Correction**: Use Edit Warp mode to match the perspective of each surface
5. **Fine-Tune**: Adjust HSV and alpha for color matching
6. **Perform**: Enter play mode (SPACE) and run your show

## Credits

- Original Processing version by Adam Croston (2022)
- Python rewrite and enhancements by Adam Croston (2026)

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

You are free to:
- **Share** - copy and redistribute the material
- **Adapt** - remix, transform, and build upon the material

Under the following terms:
- **Attribution** - Give appropriate credit
- **ShareAlike** - Distribute contributions under the same license
