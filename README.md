# Hayai

**Rapid Video Projection Mapping for Everyone**

Hayai is a fast, free, and beginner-friendly projection mapping tool with perspective-correct texture Keystoneing. The name "hayai" (速い) means "fast" in Japanese-reflecting both its quick setup time and real-time performance suitable for live shows.

## Features

- **Beginner-Friendly** - Intuitive UI with on-screen hints and tooltips
- **Fast Setup** - Create and map shapes in seconds, not hours
- **Live Performance Ready** - GPU-accelerated OpenGL rendering for smooth real-time playback
- **Perspective Correction** - Advanced inverse bilinear interpolation for accurate texture Keystoneing
- **Animation Support** - Load animated GIFs and video files directly onto shapes
- **Desktop Capture** - Capture any region of your desktop as a live texture source
- **Video Pipe Input** - Receive live video from other applications via Spout
- **HSV Color Control** - Adjust hue, saturation, value, and alpha per shape
- **Imagery Management** - Centralized panel for managing all texture sources
- **Groups & Hierarchy** - Organize shapes into groups for easier management
- **100% Free** - Open source under Creative Commons BY-SA 4.0

## Installation

### Requirements

- Python 3.10 or higher
- Windows (macOS/Linux support not yet tested)

### Quick Start

1. **Install Python** (if you haven't already):

   1.1. Download Python 3.10+ from [python.org](https://www.python.org/downloads/)

   1.2. During installation, check "Add Python to PATH"

2. **Clone or download** this repository

3. **Navigate into it**- into the top level folder of the local repository.

4. **Create and Activate a virtual environment** (optional, but recommended):

   4.1. Create it:

         ```bash
         python -m venv .venv
         ```

   4.2. Activate it:

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
- opencv-python (for video file support)
- mss + pywin32 (for desktop capture)
- SpoutGL (for video pipe input)

## Usage

Hayai is designed for video projection mapping where the projector displays onto physical surfaces. The application starts in windowed mode-press **F11** for fullscreen when projecting.

### Quick Start Guide

1. **Create a shape**: Click "Freeform" or "Regular" to start creating shapes
2. **Add imagery**: Expand the imagery panel (right edge), click + to add an image, animation, desktop capture, or video pipe
3. **Assign to shape**: Select a shape, then click an imagery item to assign it
4. **Edit the Keystone**: Click "Edit Keystone" mode and drag corners for perspective correction
5. **Go live**: Press **SPACE** to hide the UI and show only your mapped content

### Operating Modes

| Mode | Description |
|------|-------------|
| **Freeform** | Click to place vertices, click near start or press ENTER to close |
| **Regular** | Click to place regular polygons (3-120 sides) |
| **Move Shape** | Select, move, rotate, scale, and manage shapes |
| **Edit Shape** | Add, move, or delete individual vertices |
| **Edit Keystone** | Adjust the 4-corner perspective Keystone |

### Keyboard Shortcuts

#### Global
| Key | Action |
|-----|--------|
| `SPACE` | Toggle UI visibility (play mode) |
| `F11` | Toggle fullscreen |
| `ESC` | Exit fullscreen |
| `H` | Toggle hierarchy panel |
| `I` | Toggle imagery panel |
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

#### Edit Keystone Mode
| Input | Action |
|-------|--------|
| Drag corner | Adjust Keystone point |
| Right-drag | Rotate Keystone only |
| Mouse wheel | Adjust perspective amount |
| Middle-click | Switch perspective axis (X/Y) |
| Arrow keys | Move all Keystone points |
| `Fit` button | Reset Keystone corners to shape bounds |

### Mouse Controls Summary

| Button | Action |
|--------|--------|
| Left click | Select / Place / Drag |
| Right drag | Rotate |
| Middle click | Switch perspective axis (Edit Keystone mode) |
| Scroll wheel | Scale (Move mode) / Perspective (Keystone mode) |

### Imagery Panel

The collapsible panel on the right edge manages all texture sources. Press **I** or click the panel edge to expand/collapse.

**Imagery Types:**
- **Image** - Static images (PNG, JPG, BMP)
- **Animation** - Animated GIFs and video files (MP4, AVI, MOV) with a speed control
- **Desktop Capture** - Live capture of any screen region
- **Video Pipe** - Live input from other applications via Spout

**Desktop Capture:**
1. Click the capture button (+screen icon) in the imagery panel
2. A green overlay window appears on your desktop
3. Move and resize the overlay to select the capture region
4. The captured area updates live on any shapes using this imagery

**Video Pipe (Spout):**
1. Click the pipe button in the imagery panel
2. Enter the Spout sender name to connect to
3. Live video from the sender appears on shapes using this imagery

### Properties Panel

The right-side panel adapts to your selection:

**Single Shape:**
- **Name** - Editable shape name
- **Anim Offset** - Animation playback offfset for GIFs
- **Alpha** - Transparency (0 = invisible, 1 = opaque)
- **Hue** - Rotate colors around the color wheel (0-360°)
- **Saturation** - Color intensity (0 = grayscale, 2 = oversaturated)
- **Brightness** - Brightness level (0 = black, 2 = overbright)
- **Perspective X/Y** - Fine-tune perspective distortion

**Group:** Shows name and transform info (position, rotation, scale)

**Multiple Items:** Shows selection count and lists selected item names with type indicators

### Display Options

- **Geometry** - Show/hide shape outlines and Keystone corners
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
| Video | MP4, AVI, MOV |
| Projects | .hayai, .json |

## Tips for Live Performance

1. **Prepare your scene** ahead of time and save it
2. **Use F11** to go fullscreen on your projector output
3. **Press SPACE** to hide all UI elements during the show
4. **Use groups** to move multiple shapes together
5. **Animated GIFs and videos** loop automatically-great for dynamic content
6. **Desktop capture** can display live content from other applications
7. **Spout input** enables integration with VJ software like Resolume, TouchDesigner, etc.

## Projection Mapping Workflow

1. **Physical Setup**: Position your projector aimed at the target surface(s)
2. **Create Shapes**: Trace the edges of physical surfaces with Freeform shapes, or use Regular shapes for geometric objects
3. **Edit Shapes**: Add, remove, and move verts to refine shapes.
4. **Load Content**: Add images or animations to each shape
5. **Keystone Correction**: Use Edit Perspective Keystone corrections to match the perspective of each surface
6. **Fine-Tune**: Adjust HSV and alpha for color matching
7. **Perform**: Enter play mode (SPACE) and run your show

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
