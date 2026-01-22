import pygame
from pygame.locals import *
import json
import math
import os
import sys
import time
import ctypes
from tkinter import Tk, filedialog, messagebox
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeAlias

try:
    from PIL import Image
except ImportError:
    import tkinter.messagebox as messagebox
    root = Tk()
    root.withdraw()
    messagebox.showerror("Missing Dependency", "PIL (Pillow) is required to run this application.\n\nInstall it with: pip install Pillow")
    root.destroy()
    sys.exit(1)

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    import tkinter.messagebox as messagebox
    root = Tk()
    root.withdraw()
    messagebox.showerror("Missing Dependency", "PyOpenGL is required to run this application.\n\nInstall it with: pip install PyOpenGL PyOpenGL-accelerate")
    root.destroy()
    sys.exit(1)

# Fix Windows DPI scaling issues - must be called before pygame.init()
if sys.platform == 'win32':
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Initialize pygame
pygame.init()

# Window dimensions
BASE_WIDTH, BASE_HEIGHT = 2560, 1440
WIDTH, HEIGHT = 2560, 1440

# Colors
BG_COLOR = (0, 0, 0)
BG_COLOR_UI = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
SHAPE_OUTLINE_COLOR = (100, 200, 255)
SELECTED_COLOR = (255, 200, 100)
POINT_COLOR = (255, 100, 100)
WARP_POINT_COLOR = (100, 255, 100)
UI_BG = (45, 45, 50)
UI_TEXT = (220, 220, 220)
UI_HIGHLIGHT = (80, 140, 200)
UI_BUTTON = (60, 60, 70)
UI_BUTTON_HOVER = (80, 80, 90)

# Selection and grasp handle colors
HANDLE_COLOR = (255, 255, 255)           # White grasp handles
HANDLE_HOVER_COLOR = (255, 220, 100)     # Yellow when hovered
HANDLE_SIZE = 8                          # Radius for corner handles
EDGE_HANDLE_SIZE = 6                     # Radius for edge handles
GROUP_MEMBER_TINT = (180, 160, 100)      # Dimmer outline for shapes in selected group

# Marching ants animation
MARCH_SPEED = 100  # Pixels per second
MARCH_DASH = 6     # Dash length
MARCH_GAP = 4      # Gap length

# Default grid pattern surface
def create_grid_pattern(width, height, grid_size=20):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill((60, 60, 70, 255))
    for x in range(0, width, grid_size):
        pygame.draw.line(surface, (80, 80, 90, 255), (x, 0), (x, height))
    for y in range(0, height, grid_size):
        pygame.draw.line(surface, (80, 80, 90, 255), (0, y), (width, y))
    return surface

DEFAULT_PATTERN = create_grid_pattern(400, 400)

# Vertex shader for perspective warping - passes screen position to fragment shader
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;

out vec2 screenPos;

uniform vec2 screenSize;

void main() {
    // Convert from screen coordinates to NDC (-1 to 1)
    vec2 ndc = (position / screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for OpenGL
    gl_Position = vec4(ndc, 0.0, 1.0);
    screenPos = position;
}
"""

# Fragment shader with per-pixel inverse bilinear interpolation and perspective correction
FRAGMENT_SHADER = """
#version 330 core
in vec2 screenPos;
out vec4 fragColor;

uniform sampler2D textureSampler;
uniform vec2 p0, p1, p2, p3;  // quad corners in screen coords: TL, TR, BR, BL
uniform float perspectiveX;   // perspective warp on vertical axis
uniform float perspectiveY;   // perspective warp on horizontal axis
uniform float shapeAlpha;     // shape opacity (0.0-1.0)
uniform float hueShift;       // hue adjustment (0-360 degrees)
uniform float satMult;        // saturation multiplier (0.0-2.0)
uniform float valMult;        // value/brightness multiplier (0.0-2.0)

//----------------------------------------------------------------------------
// Math

bool isClose(float a, float b, float tolerance)
{
    return (a == b) ||
			(	(a <= (b + tolerance)) &&
				(a >= (b - tolerance)));
}
bool OnRangeInclusive(float val, float range_min, float range_max, float tol)
{
    return ((val+tol) >= range_min) && ((val-tol) <= range_max);
}

// Cross product of 2D vectors (returns scalar z-component)
// Easy cross product (used by inverseBilerp).
float cross2d(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

//----------------------------------------------------------------------------
// Geomertry

// Get the distance to a line from a point.
float DistToLine(in vec2 vA, in vec2 vB, in vec2 pt){
	vec2 pa = pt - vA;
	vec2 ba = vB - vA;
	float h = clamp(dot(pa,ba) / dot(ba,ba), 0.0, 1.0);
	return length(pa - ba*h);
}

// Determine the side of the line of a point.
// Leftward is positive and rightward is negative.
float Side(in vec2 vA, in vec2 vB, in vec2 pt){
    return (pt.x - vB.x) * (vA.y - vB.y) - (vA.x - vB.x) * (pt.y - vB.y);
}

// Determine if a point is inside a triangle.
// The triangle should be defined in anti-clockwise order.
bool PointInTriangle (in vec2 vA, in vec2 vB, in vec2 vC, in vec2 pt){
    bool b1 = (Side(vA, vB, pt) > 0.0);
    bool b2 = (Side(vB, vC, pt) > 0.0);
    bool b3 = (Side(vC, vA, pt) > 0.0);

    return ((b1 == b2) && (b2 == b3));
}

/**
 * Computes the inverse of a bilinear map.
 * @param p        The current fragment/pixel coordinate.
 * @param v0,v1,v2,v3 The four vertices of the quadrilateral in counter-clockwise order.
 * @return vec2    The mapped s,t coordinates in the range [0, 1].
 */
vec2 inversebilinear(vec2 p, vec2 v0, vec2 v1, vec2 v2, vec2 v3) {
    // Number of Newton iterations
    const int numiter = 4;
    
    // Start in the center (s0, s1 in Python, s.x, s.y here)
    vec2 s = vec2(0.5, 0.5);

    for (int i = 0; i < numiter; i++) {
        // Residual: The difference between our current guess and the target point p
        // r = v[0]*(1-s0)*(1-s1) + v[1]*s0*(1-s1) + v[2]*s0*s1 + v[3]*(1-s0)*s1 - p
        vec2 r = v0 * (1.0 - s.x) * (1.0 - s.y) + 
                 v1 * s.x * (1.0 - s.y) + 
                 v2 * s.x * s.y + 
                 v3 * (1.0 - s.x) * s.y - p;

        // Jacobian components (Partial derivatives)
        // J11/J21 = dR/ds0 (Change in R relative to s.x)
        // J12/J22 = dR/ds1 (Change in R relative to s.y)
        float J11 = -v0.x * (1.0 - s.y) + v1.x * (1.0 - s.y) + v2.x * s.y - v3.x * s.y;
        float J21 = -v0.y * (1.0 - s.y) + v1.y * (1.0 - s.y) + v2.y * s.y - v3.y * s.y;
        float J12 = -v0.x * (1.0 - s.x) - v1.x * s.x + v2.x * s.x + v3.x * (1.0 - s.x);
        float J22 = -v0.y * (1.0 - s.x) - v1.y * s.x + v2.y * s.x + v3.y * (1.0 - s.x);

        // Determinant for matrix inversion
        float detJ = J11 * J22 - J12 * J21;
        
        // Prevent division by zero
        if (abs(detJ) < 1e-6) break;
        
        float inv_detJ = 1.0 / detJ;

        // Newton step: s = s - J^-1 * r
        s.x -= inv_detJ * (J22 * r.x - J12 * r.y);
        s.y -= inv_detJ * (-J21 * r.x + J11 * r.y);
    }

    return s;
}


// Apply perspective-correct texture coordinate transformation
vec2 perspectiveCorrectUV(vec2 uv, float px, float py) {
    // Calculate virtual depth (w) at each corner based on perspective values
    // Higher w = farther away, affects texture sampling density
    float w0 = max(0.0001, 1.0 - px - py);  // TL
    float w1 = max(0.0001, 1.0 - px + py);  // TR
    float w2 = max(0.0001, 1.0 + px + py);  // BR
    float w3 = max(0.0001, 1.0 + px - py);  // BL

    float u = uv.x;
    float v = uv.y;

    // Bilinear weights for the four corners
    float weight00 = (1.0 - u) * (1.0 - v);
    float weight10 = u * (1.0 - v);
    float weight11 = u * v;
    float weight01 = (1.0 - u) * v;

    // Interpolate 1/w (inverse depth) using bilinear weights
    float invW = weight00 / w0 + weight10 / w1 + weight11 / w2 + weight01 / w3;

    // Texture coordinates at corners: TL=(0,0), TR=(1,0), BR=(1,1), BL=(0,1)
    // Interpolate texU/w and texV/w
    float texU_over_w = (0.0 * weight00 / w0) + (1.0 * weight10 / w1) +
                        (1.0 * weight11 / w2) + (0.0 * weight01 / w3);
    float texV_over_w = (0.0 * weight00 / w0) + (0.0 * weight10 / w1) +
                        (1.0 * weight11 / w2) + (1.0 * weight01 / w3);

    // Perspective divide to get final texture coordinates
    if (invW > 0.0001) {
        return vec2(texU_over_w / invW, texV_over_w / invW);
    }
    return uv;
}
/*
// Apply perspective-correct texture coordinate transformation
vec2 perspectiveCorrectUVOLD(vec2 uv, float px, float py) {
    // Perspective correct equation.
    // From http://en.wikipedia.org/wiki/Texture_mapping#Perspective_correctness
    // The equation is identical for U and V, so we only show the U one below.
    //ualpha = (((1-alpha)*(u0/z0)) + ((alpha)*(u1/z1))) /;
    //         (((1-alpha)*(1.0/z0)) + ((alpha)*(1.0/z1)));

    // Abuse the equation above to apply as a simple user
    // controllable warping amount.
    // We do this by pretending the z value at u0 is just one and
    // giving the user control of the z value at u1.
    // We use vector ops to do this on U and V at the same timeS.
    vec2 warpPerspXY = vec2(px, py);
    vec2 warpUV = warpPerspXY;//vec2(0.5, 0.5);
    vec2 posUVWarped =
        ((uv)*(1.0/warpUV)) /
        (((vec2(1.0, 1.0)-uv)) + ((uv)*(1.0/warpUV)));
    return posUVWarped;
}
*/

//----------------------------------------------------------------------------
// HSV color conversion functions

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // Try inverse bilinear with standard point order first
    vec2 uv = inversebilinear(screenPos, p0, p1, p2, p3);

    // Clamp UV to valid range
    uv = clamp(uv, 0.0, 1.0);

    // Apply perspective correction to texture coordinates
    vec2 texCoord = perspectiveCorrectUV(uv, perspectiveX, perspectiveY);

    // Clamp UV to valid range
    texCoord = clamp(texCoord, 0.0, 1.0);

    // Sample texture (flip V for OpenGL texture coordinates)
    vec4 texColor = texture(textureSampler, vec2(texCoord.x, 1.0 - texCoord.y));

    // Apply HSV adjustments
    vec3 hsv = rgb2hsv(texColor.rgb);
    hsv.x = fract(hsv.x + hueShift / 360.0);  // Shift hue (normalized 0-1)
    hsv.y = clamp(hsv.y * satMult, 0.0, 1.0); // Adjust saturation
    hsv.z = clamp(hsv.z * valMult, 0.0, 1.0); // Adjust brightness
    vec3 adjustedRGB = hsv2rgb(hsv);

    // Apply shape alpha
    fragColor = vec4(adjustedRGB, texColor.a * shapeAlpha);
}
"""


class GLRenderer:
    """OpenGL renderer for perspective-correct quad warping using shaders"""

    def __init__(self):
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.textures = {}
        self.screen_width = 800
        self.screen_height = 600
        # Uniform locations
        self.u_screen_size = -1
        self.u_p0 = -1
        self.u_p1 = -1
        self.u_p2 = -1
        self.u_p3 = -1
        self.u_texture = -1
        self.u_perspective_x = -1
        self.u_perspective_y = -1
        self.u_shape_alpha = -1
        self.u_hue_shift = -1
        self.u_sat_mult = -1
        self.u_val_mult = -1

    def invalidate(self):
        """Invalidate all OpenGL objects - call when context is destroyed"""
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.textures = {}
        self.u_screen_size = -1
        self.u_p0 = -1
        self.u_p1 = -1
        self.u_p2 = -1
        self.u_p3 = -1
        self.u_texture = -1
        self.u_perspective_x = -1
        self.u_perspective_y = -1
        self.u_shape_alpha = -1
        self.u_hue_shift = -1
        self.u_sat_mult = -1
        self.u_val_mult = -1

    def _compile_shader(self, source, shader_type):
        """Compile a shader and return its ID"""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        # Check for compilation errors
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            shader_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
            print(f"Shader compilation error ({shader_name}): {error}")
            glDeleteShader(shader)
            return None
        return shader

    def _create_shader_program(self):
        """Create and link the shader program"""
        vertex_shader = self._compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = self._compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

        if not vertex_shader or not fragment_shader:
            return False

        self.shader_program = glCreateProgram()
        glAttachShader(self.shader_program, vertex_shader)
        glAttachShader(self.shader_program, fragment_shader)
        glLinkProgram(self.shader_program)

        # Check for linking errors
        if not glGetProgramiv(self.shader_program, GL_LINK_STATUS):
            error = glGetProgramInfoLog(self.shader_program).decode()
            print(f"Shader linking error: {error}")
            return False

        # Clean up shaders (they're linked into the program now)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        # Get uniform locations
        self.u_screen_size = glGetUniformLocation(self.shader_program, "screenSize")
        self.u_p0 = glGetUniformLocation(self.shader_program, "p0")
        self.u_p1 = glGetUniformLocation(self.shader_program, "p1")
        self.u_p2 = glGetUniformLocation(self.shader_program, "p2")
        self.u_p3 = glGetUniformLocation(self.shader_program, "p3")
        self.u_texture = glGetUniformLocation(self.shader_program, "textureSampler")
        self.u_perspective_x = glGetUniformLocation(self.shader_program, "perspectiveX")
        self.u_perspective_y = glGetUniformLocation(self.shader_program, "perspectiveY")
        self.u_shape_alpha = glGetUniformLocation(self.shader_program, "shapeAlpha")
        self.u_hue_shift = glGetUniformLocation(self.shader_program, "hueShift")
        self.u_sat_mult = glGetUniformLocation(self.shader_program, "satMult")
        self.u_val_mult = glGetUniformLocation(self.shader_program, "valMult")

        return True

    def _create_quad_vao(self):
        """Create VAO and VBO for rendering quads"""
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # We'll update the buffer data each frame with the quad corners
        # Reserve space for 6 vertices (2 triangles) * 2 floats each
        glBufferData(GL_ARRAY_BUFFER, 6 * 2 * 4, None, GL_DYNAMIC_DRAW)

        # Position attribute (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, None)

        glBindVertexArray(0)

    def init_gl(self, width, height):
        """Initialize OpenGL context with shaders"""
        self.screen_width = width
        self.screen_height = height

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glViewport(0, 0, width, height)

        # Only create shader program once
        if not self.initialized:
            if not self._create_shader_program():
                raise RuntimeError("Failed to create shader program")

            # Create VAO for shader-based rendering
            self._create_quad_vao()

        self.initialized = True

    def surface_to_texture(self, surface):
        """Convert pygame surface to OpenGL texture"""
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        # Get surface data
        width, height = surface.get_size()
        data = pygame.image.tostring(surface, "RGBA", True)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        return tex_id, width, height

    def draw_textured_quad(self, texture_id, quad_points, _tex_width, _tex_height,
                           perspective_x=0.0, perspective_y=0.0, alpha=1.0,
                           hue_shift=0.0, saturation=1.0, color_value=1.0):
        """Draw a textured quad with perspective warping using shaders"""
        p0, p1, p2, p3 = quad_points  # TL, TR, BR, BL

        glUseProgram(self.shader_program)

        # Set uniforms
        glUniform2f(self.u_screen_size, self.screen_width, self.screen_height)
        glUniform2f(self.u_p0, p0[0], p0[1])
        glUniform2f(self.u_p1, p1[0], p1[1])
        glUniform2f(self.u_p2, p2[0], p2[1])
        glUniform2f(self.u_p3, p3[0], p3[1])
        glUniform1f(self.u_perspective_x, perspective_x)
        glUniform1f(self.u_perspective_y, perspective_y)
        glUniform1f(self.u_shape_alpha, alpha)
        glUniform1f(self.u_hue_shift, hue_shift)
        glUniform1f(self.u_sat_mult, saturation)
        glUniform1f(self.u_val_mult, color_value)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(self.u_texture, 0)

        # Calculate bounding box for the quad to minimize fragment processing
        min_x = min(p0[0], p1[0], p2[0], p3[0])
        max_x = max(p0[0], p1[0], p2[0], p3[0])
        min_y = min(p0[1], p1[1], p2[1], p3[1])
        max_y = max(p0[1], p1[1], p2[1], p3[1])

        # Update VBO with bounding box vertices (2 triangles)
        vertices = [
            min_x, min_y,
            max_x, min_y,
            max_x, max_y,
            min_x, min_y,
            max_x, max_y,
            min_x, max_y,
        ]
        vertices_array = (ctypes.c_float * len(vertices))(*vertices)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(vertices) * 4, vertices_array)

        # Draw
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glUseProgram(0)

    def draw_textured_quad_direct(self, texture_id, quad_points, _tex_width, _tex_height,
                                   perspective_x=0.0, perspective_y=0.0, alpha=1.0,
                                   hue_shift=0.0, saturation=1.0, color_value=1.0):
        """Draw a textured quad directly as two triangles (no bounding box)"""
        p0, p1, p2, p3 = quad_points  # TL, TR, BR, BL

        glUseProgram(self.shader_program)

        # Set uniforms
        glUniform2f(self.u_screen_size, self.screen_width, self.screen_height)
        glUniform2f(self.u_p0, p0[0], p0[1])
        glUniform2f(self.u_p1, p1[0], p1[1])
        glUniform2f(self.u_p2, p2[0], p2[1])
        glUniform2f(self.u_p3, p3[0], p3[1])
        glUniform1f(self.u_perspective_x, perspective_x)
        glUniform1f(self.u_perspective_y, perspective_y)
        glUniform1f(self.u_shape_alpha, alpha)
        glUniform1f(self.u_hue_shift, hue_shift)
        glUniform1f(self.u_sat_mult, saturation)
        glUniform1f(self.u_val_mult, color_value)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(self.u_texture, 0)

        # Draw warp quad directly as 2 triangles: TL-TR-BR and TL-BR-BL
        vertices = [
            p0[0], p0[1],  # TL
            p1[0], p1[1],  # TR
            p2[0], p2[1],  # BR
            p0[0], p0[1],  # TL
            p2[0], p2[1],  # BR
            p3[0], p3[1],  # BL
        ]
        vertices_array = (ctypes.c_float * len(vertices))(*vertices)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, len(vertices) * 4, vertices_array)

        # Draw
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindVertexArray(0)
        glUseProgram(0)

    def draw_textured_triangles(self, texture_id, triangles, warp_points, _tex_width, _tex_height,
                                 perspective_x=0.0, perspective_y=0.0, alpha=1.0,
                                 hue_shift=0.0, saturation=1.0, color_value=1.0):
        """Draw triangulated polygon with texture using warp points for UV calculation"""
        if not triangles:
            return

        p0, p1, p2, p3 = warp_points  # TL, TR, BR, BL

        glUseProgram(self.shader_program)

        # Set uniforms - same warp points for UV calculation
        glUniform2f(self.u_screen_size, self.screen_width, self.screen_height)
        glUniform2f(self.u_p0, p0[0], p0[1])
        glUniform2f(self.u_p1, p1[0], p1[1])
        glUniform2f(self.u_p2, p2[0], p2[1])
        glUniform2f(self.u_p3, p3[0], p3[1])
        glUniform1f(self.u_perspective_x, perspective_x)
        glUniform1f(self.u_perspective_y, perspective_y)
        glUniform1f(self.u_shape_alpha, alpha)
        glUniform1f(self.u_hue_shift, hue_shift)
        glUniform1f(self.u_sat_mult, saturation)
        glUniform1f(self.u_val_mult, color_value)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(self.u_texture, 0)

        # Build vertex array from triangles
        vertices = []
        for tri in triangles:
            for x, y in tri:
                vertices.extend([x, y])

        # Resize VBO if needed (current size is 6 vertices = 12 floats)
        num_vertices = len(triangles) * 3
        vertices_array = (ctypes.c_float * len(vertices))(*vertices)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Reallocate buffer if we need more space
        buffer_size = len(vertices) * 4
        glBufferData(GL_ARRAY_BUFFER, buffer_size, vertices_array, GL_DYNAMIC_DRAW)

        # Draw all triangles
        glDrawArrays(GL_TRIANGLES, 0, num_vertices)

        glBindVertexArray(0)
        glUseProgram(0)

    def delete_texture(self, tex_id):
        """Delete an OpenGL texture"""
        if tex_id:
            glDeleteTextures([tex_id])

    def clear(self, r, g, b):
        """Clear the screen"""
        glClearColor(r/255, g/255, b/255, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)


gl_renderer = GLRenderer()


def triangulate_polygon(vertices):
    """Simple ear clipping triangulation for potentially concave polygons"""
    if len(vertices) < 3:
        return []

    # Determine polygon winding order
    def polygon_area_signed(verts):
        """Calculate signed area - positive for CCW, negative for CW"""
        n = len(verts)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += verts[i][0] * verts[j][1]
            area -= verts[j][0] * verts[i][1]
        return area / 2

    # Work with a copy
    verts = list(vertices)
    triangles = []

    # Check winding order and determine which cross product sign indicates convex
    signed_area = polygon_area_signed(verts)
    ccw = signed_area > 0  # True if counter-clockwise

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def point_in_triangle(pt, v1, v2, v3):
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def is_ear(i, verts):
        n = len(verts)
        if n < 3:
            return False
        prev_i = (i - 1) % n
        next_i = (i + 1) % n

        prev_v = verts[prev_i]
        curr_v = verts[i]
        next_v = verts[next_i]

        # Check if convex (cross product sign depends on winding)
        cross = (curr_v[0] - prev_v[0]) * (next_v[1] - prev_v[1]) - \
                (curr_v[1] - prev_v[1]) * (next_v[0] - prev_v[0])
        # For CCW polygons, convex vertices have positive cross; for CW, negative
        if ccw:
            if cross <= 0:  # Reflex vertex
                return False
        else:
            if cross >= 0:  # Reflex vertex for CW winding
                return False

        # Check no other vertex inside the triangle
        for j in range(n):
            if j in (prev_i, i, next_i):
                continue
            if point_in_triangle(verts[j], prev_v, curr_v, next_v):
                return False
        return True

    indices = list(range(len(verts)))
    while len(indices) > 3:
        ear_found = False
        for i in range(len(indices)):
            if is_ear(i, [verts[j] for j in indices]):
                n = len(indices)
                prev_i = (i - 1) % n
                next_i = (i + 1) % n
                triangles.append((verts[indices[prev_i]], verts[indices[i]], verts[indices[next_i]]))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            # Fallback: just create a fan from center
            break

    if len(indices) == 3:
        triangles.append((verts[indices[0]], verts[indices[1]], verts[indices[2]]))
    elif len(indices) > 3:
        # Fallback for degenerate cases
        center = (sum(v[0] for v in verts) / len(verts),
                  sum(v[1] for v in verts) / len(verts))
        for i in range(len(verts)):
            triangles.append((center, verts[i], verts[(i + 1) % len(verts)]))

    return triangles


@dataclass
class Shape:
    contour: List[Tuple[float, float]] = field(default_factory=list)
    warp_points: List[Tuple[float, float]] = field(default_factory=list)
    image_path: Optional[str] = None
    image_surface: Optional[pygame.Surface] = None
    perspective_x: float = 0.0  # Perspective warp strength on Y axis (vertical keystone)
    perspective_y: float = 0.0  # Perspective warp strength on X axis (horizontal keystone)
    perspective_axis: int = 0  # 0 = Y (vertical), 1 = X (horizontal)
    # Animation support
    animation_frames: List[pygame.Surface] = field(default_factory=list)
    frame_durations: List[float] = field(default_factory=list)
    current_frame: int = 0
    last_frame_time: float = 0
    # OpenGL texture cache
    gl_texture_id: int = 0
    gl_texture_dirty: bool = True
    # Transform properties
    position: Tuple[float, float] = (0.0, 0.0)  # World position offset
    rotation: float = 0.0  # Rotation in degrees
    scale: float = 1.0  # Uniform scale factor
    # Identity
    name: str = ""  # User-editable shape name
    # Appearance
    alpha: float = 1.0  # Opacity (0.0-1.0)
    # Color adjustments (HSV)
    hue_shift: float = 0.0  # Hue rotation in degrees (0-360)
    saturation: float = 1.0  # Saturation multiplier (0.0-2.0, 1.0 = normal)
    color_value: float = 1.0  # Brightness/value multiplier (0.0-2.0, 1.0 = normal)
    # Animation control
    playback_speed: float = 1.0  # Animation speed multiplier (1.0 = normal)
    # Transform pivot (frozen during vertex editing to prevent center drift)
    _frozen_pivot: Optional[Tuple[float, float]] = field(default=None, repr=False)
    # Hierarchy parent reference (for Group transforms)
    _parent: Optional['Group'] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.warp_points and len(self.contour) >= 3:
            self.fit_warp_to_contour()

    def get_current_image(self):
        """Get the current frame for animated images, or the static image"""
        if self.animation_frames and len(self.animation_frames) > 0:
            current_time = time.time()
            # Initialize last_frame_time if it's 0 (not yet set)
            if self.last_frame_time == 0:
                self.last_frame_time = current_time

            # Get frame duration, default to 100ms if not available
            if self.frame_durations and self.current_frame < len(self.frame_durations):
                frame_duration = self.frame_durations[self.current_frame]
            else:
                frame_duration = 0.1

            # Apply playback speed (higher = faster, so divide duration)
            effective_duration = frame_duration / max(0.1, self.playback_speed)

            # Check if it's time to advance to the next frame
            elapsed = current_time - self.last_frame_time
            if elapsed >= effective_duration:
                self.last_frame_time = current_time
                old_frame = self.current_frame
                self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
                if old_frame != self.current_frame:
                    self.gl_texture_dirty = True  # Mark for texture update

            # Ensure current_frame is valid
            self.current_frame = self.current_frame % len(self.animation_frames)
            return self.animation_frames[self.current_frame]
        return self.image_surface

    def get_bounds(self):
        if not self.contour:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.contour]
        ys = [p[1] for p in self.contour]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_center(self):
        """Get center in world coordinates"""
        local_center = self._get_local_center()
        return self.get_world_point(local_center[0], local_center[1])

    def _get_local_center(self):
        """Get center of contour in local coordinates"""
        if not self.contour:
            return (0, 0)
        xs = [p[0] for p in self.contour]
        ys = [p[1] for p in self.contour]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _get_pivot(self):
        """Get transform pivot - returns frozen pivot if set, otherwise computed center"""
        if self._frozen_pivot is not None:
            return self._frozen_pivot
        return self._get_local_center()

    def freeze_pivot(self):
        """Freeze the current center as the transform pivot (call before editing vertices)"""
        self._frozen_pivot = self._get_local_center()

    def unfreeze_pivot(self):
        """Unfreeze the pivot, adjusting position to compensate for center change"""
        if self._frozen_pivot is None:
            return

        old_center = self._frozen_pivot
        new_center = self._get_local_center()

        # If center didn't change, just unfreeze
        if old_center == new_center:
            self._frozen_pivot = None
            return

        # Compute position adjustment: (I - R*S) * (old_center - new_center)
        # This keeps all non-edited vertices at their same world positions
        delta_x = old_center[0] - new_center[0]
        delta_y = old_center[1] - new_center[1]

        # Apply rotation and scale to the delta
        angle = math.radians(self.rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rs_delta_x = self.scale * (delta_x * cos_a - delta_y * sin_a)
        rs_delta_y = self.scale * (delta_x * sin_a + delta_y * cos_a)

        # The adjustment is (I - R*S) * delta = delta - R*S*delta
        adj_x = delta_x - rs_delta_x
        adj_y = delta_y - rs_delta_y

        # Apply adjustment to position
        self.position = (self.position[0] + adj_x, self.position[1] + adj_y)

        # Now unfreeze
        self._frozen_pivot = None

    def get_world_point(self, local_x, local_y):
        """Convert local point to world coordinates using transform (including parent chain)"""
        # Get pivot for rotation/scale (frozen during editing to prevent drift)
        cx, cy = self._get_pivot()

        # Apply scale around local center
        sx = cx + (local_x - cx) * self.scale
        sy = cy + (local_y - cy) * self.scale

        # Apply rotation around local center
        angle = math.radians(self.rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = cx + (sx - cx) * cos_a - (sy - cy) * sin_a
        ry = cy + (sx - cx) * sin_a + (sy - cy) * cos_a

        # Apply position offset
        world_x, world_y = rx + self.position[0], ry + self.position[1]

        # Apply parent transforms up the chain
        if self._parent is not None:
            world_x, world_y = self._parent.transform_point(world_x, world_y)

        return (world_x, world_y)

    def get_world_contour(self):
        """Get contour points in world coordinates"""
        return [self.get_world_point(x, y) for x, y in self.contour]

    def get_world_warp_points(self):
        """Get warp points in world coordinates"""
        return [self.get_world_point(x, y) for x, y in self.warp_points]

    def get_local_point(self, world_x, world_y):
        """Convert world point to local coordinates (inverse of get_world_point)"""
        # First apply parent inverse transforms (from top down)
        if self._parent is not None:
            world_x, world_y = self._parent.inverse_transform_point(world_x, world_y)

        # Inverse of: Scale → Rotate → Translate
        # So we do: Inverse-Translate → Inverse-Rotate → Inverse-Scale

        cx, cy = self._get_pivot()

        # 1. Inverse translate: subtract position offset
        tx = world_x - self.position[0]
        ty = world_y - self.position[1]

        # 2. Inverse rotate around local center
        angle = math.radians(-self.rotation)  # Negative angle for inverse
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = cx + (tx - cx) * cos_a - (ty - cy) * sin_a
        ry = cy + (tx - cx) * sin_a + (ty - cy) * cos_a

        # 3. Inverse scale around local center
        if self.scale != 0:
            lx = cx + (rx - cx) / self.scale
            ly = cy + (ry - cy) / self.scale
        else:
            lx, ly = rx, ry

        return (lx, ly)

    def _get_world_rotation(self):
        """Get accumulated world rotation including parent chain."""
        rotation = self.rotation
        if self._parent is not None:
            # Get parent's world rotation
            _, _, _, _, _ = self._parent.get_world_transform_matrix()
            # Accumulate rotations
            parent = self._parent
            while parent is not None:
                rotation += parent.rotation
                parent = parent._parent
        return rotation

    def _get_world_scale(self):
        """Get accumulated world scale including parent chain."""
        scale = self.scale
        parent = self._parent
        while parent is not None:
            scale *= parent.scale
            parent = parent._parent
        return scale

    def world_delta_to_local(self, dx, dy):
        """Convert a world-space movement delta to local-space delta"""
        # Need to account for rotation and scale of shape AND all parent groups

        # Get accumulated world rotation and scale
        world_rotation = self._get_world_rotation()
        world_scale = self._get_world_scale()

        # Inverse rotate the delta
        angle = math.radians(-world_rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        # Inverse scale the delta
        if world_scale != 0:
            rx /= world_scale
            ry /= world_scale

        return (rx, ry)

    def get_world_bounds(self):
        """Get bounding box in world coordinates"""
        world_contour = self.get_world_contour()
        if not world_contour:
            return (0, 0, 0, 0)
        xs = [p[0] for p in world_contour]
        ys = [p[1] for p in world_contour]
        return (min(xs), min(ys), max(xs), max(ys))

    def fit_warp_to_contour(self):
        # If shape has exactly 4 vertices, use them directly as warp points
        if len(self.contour) == 4:
            # Sort vertices to determine TL, TR, BR, BL order
            # First, find the center
            cx = sum(p[0] for p in self.contour) / 4
            cy = sum(p[1] for p in self.contour) / 4

            # Categorize points by quadrant relative to center
            top_left = None
            top_right = None
            bottom_right = None
            bottom_left = None

            for p in self.contour:
                if p[0] <= cx and p[1] <= cy:
                    top_left = p
                elif p[0] > cx and p[1] <= cy:
                    top_right = p
                elif p[0] > cx and p[1] > cy:
                    bottom_right = p
                else:
                    bottom_left = p

            # If all corners found, use them; otherwise fall back to contour order
            if top_left and top_right and bottom_right and bottom_left:
                self.warp_points = [top_left, top_right, bottom_right, bottom_left]
            else:
                # Use contour order directly (assume user drew in correct order)
                self.warp_points = list(self.contour)
        else:
            # For shapes with more or fewer than 4 verts, use bounding box
            bounds = self.get_bounds()
            self.warp_points = [
                (bounds[0], bounds[1]),  # top-left
                (bounds[2], bounds[1]),  # top-right
                (bounds[2], bounds[3]),  # bottom-right
                (bounds[0], bounds[3]),  # bottom-left
            ]

    def contains_point(self, point):
        """Test if point is inside shape using world coordinates"""
        if len(self.contour) < 3:
            return False
        # Use world coordinates for hit testing
        world_contour = self.get_world_contour()
        x, y = point
        n = len(world_contour)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = world_contour[i]
            xj, yj = world_contour[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def move(self, dx, dy):
        """Move shape by updating position offset"""
        self.position = (self.position[0] + dx, self.position[1] + dy)

    def normalize_warp_points(self):
        """Reorder warp points to ensure TL, TR, BR, BL order based on actual positions"""
        if len(self.warp_points) != 4:
            return

        # Find center of warp quad
        cx = sum(p[0] for p in self.warp_points) / 4
        cy = sum(p[1] for p in self.warp_points) / 4

        # Categorize points by quadrant
        top_left = None
        top_right = None
        bottom_right = None
        bottom_left = None

        for p in self.warp_points:
            if p[0] <= cx and p[1] <= cy:
                top_left = p
            elif p[0] > cx and p[1] <= cy:
                top_right = p
            elif p[0] > cx and p[1] > cy:
                bottom_right = p
            else:
                bottom_left = p

        # Only reorder if all corners found
        if top_left and top_right and bottom_right and bottom_left:
            self.warp_points = [top_left, top_right, bottom_right, bottom_left]

    def rotate(self, clockwise=True, degrees=90, warp_only=False):
        """Rotate shape by updating rotation angle"""
        delta = -degrees if clockwise else degrees
        self.rotation = (self.rotation + delta) % 360

    def set_scale(self, scale):
        """Set uniform scale factor"""
        self.scale = max(0.01, min(10.0, scale))  # Clamp between 0.01 and 10.0

    def flip_x(self, warp_only=False):
        """Flip horizontally by modifying local coordinates"""
        cx, _ = self._get_local_center()
        if not warp_only:
            self.contour = [(2 * cx - x, y) for x, y in self.contour]
        self.warp_points = [(2 * cx - x, y) for x, y in self.warp_points]

    def flip_y(self, warp_only=False):
        """Flip vertically by modifying local coordinates"""
        _, cy = self._get_local_center()
        if not warp_only:
            self.contour = [(x, 2 * cy - y) for x, y in self.contour]
        self.warp_points = [(x, 2 * cy - y) for x, y in self.warp_points]

    def flip_x_global(self, pivot: Tuple[float, float], warp_only=False):
        """Flip horizontally about global X axis through pivot point"""
        def flip_point(px: float, py: float) -> Tuple[float, float]:
            world_x, world_y = self.get_world_point(px, py)
            new_world_x = 2 * pivot[0] - world_x
            return self.get_local_point(new_world_x, world_y)

        if not warp_only:
            self.contour = [flip_point(x, y) for x, y in self.contour]
        self.warp_points = [flip_point(x, y) for x, y in self.warp_points]

    def flip_y_global(self, pivot: Tuple[float, float], warp_only=False):
        """Flip vertically about global Y axis through pivot point"""
        def flip_point(px: float, py: float) -> Tuple[float, float]:
            world_x, world_y = self.get_world_point(px, py)
            new_world_y = 2 * pivot[1] - world_y
            return self.get_local_point(world_x, new_world_y)

        if not warp_only:
            self.contour = [flip_point(x, y) for x, y in self.contour]
        self.warp_points = [flip_point(x, y) for x, y in self.warp_points]

    def scale_geometry(self, scale_x: float, scale_y: float,
                       pivot: Tuple[float, float]) -> None:
        """
        Scale the shape's geometry (contour and warp points) around a pivot point.
        This is a permanent deformation - modifies the actual coordinates.

        Args:
            scale_x: Horizontal scale factor (1.0 = no change)
            scale_y: Vertical scale factor (1.0 = no change)
            pivot: The world-space point that remains fixed during scaling
        """
        def scale_point(px: float, py: float) -> Tuple[float, float]:
            """Scale a world point around the pivot"""
            # Get the world position of this local point
            world_x, world_y = self.get_world_point(px, py)
            # Scale in world space
            new_world_x = pivot[0] + (world_x - pivot[0]) * scale_x
            new_world_y = pivot[1] + (world_y - pivot[1]) * scale_y
            # Convert back to local coordinates
            return self.get_local_point(new_world_x, new_world_y)

        # Scale contour points
        self.contour = [scale_point(x, y) for x, y in self.contour]

        # Scale warp points
        self.warp_points = [scale_point(x, y) for x, y in self.warp_points]

    def load_image(self, path):
        """Load image or animation from path. Returns error message string or None on success."""
        try:
            self.image_path = path
            self.animation_frames = []
            self.frame_durations = []
            self.current_frame = 0
            self.last_frame_time = time.time()
            self.gl_texture_dirty = True  # Mark texture for refresh

            # Check for animated GIF
            if path.lower().endswith('.gif'):
                pil_image = Image.open(path)
                try:
                    # Check if it's animated
                    n_frames = getattr(pil_image, 'n_frames', 1)
                    if n_frames > 1:
                        # Create a canvas to properly composite frames
                        # Some GIFs have frames that only update parts of the image
                        canvas = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))

                        for frame_idx in range(n_frames):
                            pil_image.seek(frame_idx)

                            # Get frame duration (in milliseconds, convert to seconds)
                            duration = pil_image.info.get('duration', 100) / 1000.0
                            if duration <= 0:
                                duration = 0.1  # Default 100ms for invalid durations
                            self.frame_durations.append(duration)

                            # Handle disposal method for proper GIF rendering
                            # Convert current frame to RGBA
                            frame_rgba = pil_image.convert('RGBA')

                            # Composite onto canvas
                            canvas.paste(frame_rgba, (0, 0), frame_rgba)

                            # Convert composited frame to pygame surface
                            frame_copy = canvas.copy()
                            mode = frame_copy.mode
                            size = frame_copy.size
                            data = frame_copy.tobytes()
                            pygame_surface = pygame.image.fromstring(data, size, mode)
                            self.animation_frames.append(pygame_surface)

                            # Handle disposal - some GIFs clear the canvas between frames
                            disposal = pil_image.info.get('disposal', 0)
                            if disposal == 2:  # Restore to background
                                canvas = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))

                        self.image_surface = self.animation_frames[0]
                        return None  # Success
                except EOFError:
                    pass

            # Static image
            self.image_surface = pygame.image.load(path).convert_alpha()
            return None  # Success
        except Exception as e:
            print(f"Failed to load image: {e}")
            self.image_surface = None
            return str(e)

    def to_dict(self):
        return {
            'contour': self.contour,
            'warp_points': self.warp_points,
            'image_path': self.image_path,
            'perspective_x': self.perspective_x,
            'perspective_y': self.perspective_y,
            'perspective_axis': self.perspective_axis,
            # Transform properties
            'position': self.position,
            'rotation': self.rotation,
            'scale': self.scale,
            # Identity
            'name': self.name,
            # Appearance
            'alpha': self.alpha,
            # Color adjustments (HSV)
            'hue_shift': self.hue_shift,
            'saturation': self.saturation,
            'color_value': self.color_value,
            # Animation control
            'playback_speed': self.playback_speed,
        }

    @classmethod
    def from_dict(cls, data):
        shape = cls(
            contour=[tuple(p) for p in data.get('contour', [])],
            warp_points=[tuple(p) for p in data.get('warp_points', [])],
            image_path=data.get('image_path'),
            perspective_x=data.get('perspective_x', 0.0),
            perspective_y=data.get('perspective_y', 0.0),
            perspective_axis=data.get('perspective_axis', 0),
            # Transform properties
            position=tuple(data.get('position', (0.0, 0.0))),
            rotation=data.get('rotation', 0.0),
            scale=data.get('scale', 1.0),
            # Identity
            name=data.get('name', ''),
            # Appearance
            alpha=data.get('alpha', 1.0),
            # Color adjustments (HSV)
            hue_shift=data.get('hue_shift', 0.0),
            saturation=data.get('saturation', 1.0),
            color_value=data.get('color_value', 1.0),
            # Animation control
            playback_speed=data.get('playback_speed', 1.0),
        )
        if shape.image_path and os.path.exists(shape.image_path):
            shape.load_image(shape.image_path)
        return shape


# Type alias for hierarchy items
SceneItem = Union['Group', Shape]


@dataclass
class Group:
    """
    A group container for organizing shapes in a hierarchy.
    Groups can contain shapes and other groups, forming a tree structure.
    Transforms (position, rotation, scale) cascade to all children.
    """
    name: str = ""
    children: List[SceneItem] = field(default_factory=list)
    position: Tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0  # Rotation in degrees
    scale: float = 1.0  # Uniform scale factor
    expanded: bool = field(default=True, repr=False)  # UI state for tree view
    _parent: Optional['Group'] = field(default=None, repr=False)

    def get_transform_matrix(self) -> Tuple[float, float, float, float, float]:
        """
        Get local transform as (cos, sin, scale, tx, ty).
        Used for matrix multiplication in transform cascade.
        """
        angle = math.radians(self.rotation)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return (cos_a, sin_a, self.scale, self.position[0], self.position[1])

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform a point by this group's transform and all parent transforms.
        """
        # Apply this group's transform (scale, rotate, translate)
        angle = math.radians(self.rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Scale
        sx = x * self.scale
        sy = y * self.scale

        # Rotate
        rx = sx * cos_a - sy * sin_a
        ry = sx * sin_a + sy * cos_a

        # Translate
        world_x = rx + self.position[0]
        world_y = ry + self.position[1]

        # Apply parent transforms up the chain
        if self._parent is not None:
            world_x, world_y = self._parent.transform_point(world_x, world_y)

        return (world_x, world_y)

    def inverse_transform_point(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates to local coordinates (inverse of transform_point).
        """
        # First apply parent inverse transforms (from top down)
        if self._parent is not None:
            world_x, world_y = self._parent.inverse_transform_point(world_x, world_y)

        # Inverse translate
        tx = world_x - self.position[0]
        ty = world_y - self.position[1]

        # Inverse rotate
        angle = math.radians(-self.rotation)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a

        # Inverse scale
        if self.scale != 0:
            lx = rx / self.scale
            ly = ry / self.scale
        else:
            lx, ly = rx, ry

        return (lx, ly)

    def get_world_transform_matrix(self) -> Tuple[float, float, float, float, float]:
        """
        Get accumulated world transform matrix by multiplying parent chain.
        Returns (cos, sin, scale, tx, ty).
        """
        if self._parent is None:
            return self.get_transform_matrix()

        # Get parent's world transform
        p_cos, p_sin, p_scale, p_tx, p_ty = self._parent.get_world_transform_matrix()

        # Get our local transform
        l_cos, l_sin, l_scale, l_tx, l_ty = self.get_transform_matrix()

        # Multiply matrices: Parent * Local
        # Combined scale
        w_scale = p_scale * l_scale

        # Combined rotation (cos/sin of sum of angles)
        w_cos = p_cos * l_cos - p_sin * l_sin
        w_sin = p_sin * l_cos + p_cos * l_sin

        # Combined translation: parent_rotation * parent_scale * local_translation + parent_translation
        w_tx = p_scale * (p_cos * l_tx - p_sin * l_ty) + p_tx
        w_ty = p_scale * (p_sin * l_tx + p_cos * l_ty) + p_ty

        return (w_cos, w_sin, w_scale, w_tx, w_ty)

    def add_child(self, item: SceneItem) -> None:
        """Add a child item (Shape or Group) to this group."""
        if item not in self.children:
            # Remove from old parent if any
            if item._parent is not None:
                item._parent.children.remove(item)
            item._parent = self
            self.children.append(item)

    def remove_child(self, item: SceneItem) -> None:
        """Remove a child item from this group."""
        if item in self.children:
            self.children.remove(item)
            item._parent = None

    def get_all_shapes(self) -> List[Shape]:
        """Recursively get all Shape objects in this group and its descendants."""
        shapes = []
        for child in self.children:
            if isinstance(child, Shape):
                shapes.append(child)
            elif isinstance(child, Group):
                shapes.extend(child.get_all_shapes())
        return shapes

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of all children in world coordinates."""
        all_shapes = self.get_all_shapes()
        if not all_shapes:
            return (0, 0, 0, 0)

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for shape in all_shapes:
            bounds = shape.get_world_bounds()
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        return (min_x, min_y, max_x, max_y)

    def get_center(self) -> Tuple[float, float]:
        """Get center of group's bounding box."""
        bounds = self.get_bounds()
        return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    def move(self, dx: float, dy: float) -> None:
        """Move group by updating position offset."""
        self.position = (self.position[0] + dx, self.position[1] + dy)

    def rotate(self, degrees: float, center: Optional[Tuple[float, float]] = None) -> None:
        """
        Rotate the group by the specified degrees.
        If center is provided, also adjusts position to rotate around that point.
        """
        self.rotation = (self.rotation + degrees) % 360

        if center is not None:
            # Rotate position around the specified center point
            px, py = self.position
            cx, cy = center
            angle = math.radians(degrees)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            # Translate to origin, rotate, translate back
            dx = px - cx
            dy = py - cy
            new_x = cx + dx * cos_a - dy * sin_a
            new_y = cy + dx * sin_a + dy * cos_a
            self.position = (new_x, new_y)

    def flip_x(self, center: Tuple[float, float], warp_only: bool = False) -> None:
        """
        Flip the group horizontally around a center point.
        Flips all child shapes around the center point.
        """
        for shape in self.get_all_shapes():
            shape.flip_x_global(center, warp_only=warp_only)

    def flip_y(self, center: Tuple[float, float], warp_only: bool = False) -> None:
        """
        Flip the group vertically around a center point.
        Flips all child shapes around the center point.
        """
        for shape in self.get_all_shapes():
            shape.flip_y_global(center, warp_only=warp_only)

    def to_dict(self) -> dict:
        """Serialize group to dictionary."""
        return {
            'type': 'group',
            'name': self.name,
            'position': self.position,
            'rotation': self.rotation,
            'scale': self.scale,
            'expanded': self.expanded,
            'children': [
                child.to_dict() for child in self.children
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Group':
        """Deserialize group from dictionary."""
        group = cls(
            name=data.get('name', ''),
            position=tuple(data.get('position', (0.0, 0.0))),
            rotation=data.get('rotation', 0.0),
            scale=data.get('scale', 1.0),
            expanded=data.get('expanded', True),
        )
        # Recursively deserialize children
        for child_data in data.get('children', []):
            if child_data.get('type') == 'group':
                child = Group.from_dict(child_data)
            else:
                child = Shape.from_dict(child_data)
            group.add_child(child)
        return group


class Button:
    def __init__(self, x, y, width, height, text, callback=None, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.tooltip = tooltip
        self.hovered = False
        self.active = False
        self.focused = False  # For keyboard navigation
        self.hover_start_time = 0
        self.disabled = False  # Disabled state

    def draw(self, screen, font):
        if self.disabled:
            # Disabled state: dimmed colors
            color = (40, 40, 45)  # Very dark gray
            text_color = (80, 80, 90)  # Dimmed text
            border_color = (60, 60, 65)
            border_width = 1
        else:
            color = UI_HIGHLIGHT if self.active else (UI_BUTTON_HOVER if (self.hovered or self.focused) else UI_BUTTON)
            text_color = UI_TEXT
            # Draw focus indicator (thicker border when focused)
            border_color = (100, 180, 255) if self.focused else UI_TEXT
            border_width = 2 if self.focused else 1
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, border_color, self.rect, border_width, border_radius=4)
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def draw_tooltip(self, screen, font, scale=1.0):
        """Draw tooltip if hovered for sufficient time"""
        if self.tooltip and self.hovered:
            elapsed = time.time() - self.hover_start_time
            if elapsed >= 0.5:  # Show tooltip after 500ms
                # Create tooltip surface
                padding = int(6 * scale)
                margin = int(5 * scale)
                cursor_offset = int(16 * scale)  # Offset from cursor
                text_surf = font.render(self.tooltip, True, (255, 255, 255))
                tooltip_w = text_surf.get_width() + padding * 2
                tooltip_h = text_surf.get_height() + padding * 2

                # Position tooltip below and to the right of cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                tooltip_x = mouse_x + cursor_offset
                tooltip_y = mouse_y + cursor_offset

                # Keep tooltip on screen
                screen_w, screen_h = screen.get_size()
                if tooltip_x + tooltip_w > screen_w - margin:
                    tooltip_x = mouse_x - tooltip_w - margin
                if tooltip_y + tooltip_h > screen_h - margin:
                    tooltip_y = mouse_y - tooltip_h - margin

                # Draw tooltip background
                tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_w, tooltip_h)
                border_radius = int(4 * scale)
                pygame.draw.rect(screen, (50, 50, 55), tooltip_rect, border_radius=border_radius)
                pygame.draw.rect(screen, (100, 100, 110), tooltip_rect, 1, border_radius=border_radius)
                screen.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos)
            if self.hovered and not was_hovered:
                self.hover_start_time = time.time()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.callback and not self.disabled:
                self.callback()
                return True
        return False


class Slider:
    """Interactive slider control for numeric values"""
    def __init__(self, x, y, width, height, min_val, max_val, value, label,
                 on_change=None, format_str="{:.2f}", disabled=False, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.on_change = on_change
        self.format_str = format_str
        self.dragging = False
        self.track_height = 4
        self.handle_radius = 8
        self.disabled = disabled
        self.tooltip = tooltip
        self.hovered = False
        self.hover_start_time = 0

    def draw(self, surface, font):
        # Colors based on disabled state
        if self.disabled:
            label_color = (100, 100, 100)
            value_color = (100, 100, 100)
            track_color = (45, 45, 50)
            filled_color = (60, 60, 70)
            handle_color = (70, 70, 80)
        else:
            label_color = (180, 180, 180)
            value_color = (220, 220, 220)
            track_color = (60, 60, 70)
            filled_color = (80, 150, 220)
            handle_color = (100, 180, 255)

        # Draw label
        label_surf = font.render(self.label, True, label_color)
        surface.blit(label_surf, (self.rect.x, self.rect.y))

        # Draw value on right side
        value_str = self.format_str.format(self.value)
        value_surf = font.render(value_str, True, value_color)
        surface.blit(value_surf, (self.rect.right - value_surf.get_width(), self.rect.y))

        # Draw track (position proportional to height for proper scaling)
        track_y = self.rect.y + int(self.rect.height * 0.57)
        track_rect = pygame.Rect(self.rect.x, track_y, self.rect.width, self.track_height)
        pygame.draw.rect(surface, track_color, track_rect, border_radius=2)

        # Draw filled portion
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        filled_width = int(self.rect.width * ratio)
        filled_rect = pygame.Rect(self.rect.x, track_y, filled_width, self.track_height)
        pygame.draw.rect(surface, filled_color, filled_rect, border_radius=2)

        # Draw handle
        handle_x = self.rect.x + filled_width
        handle_y = track_y + self.track_height // 2
        pygame.draw.circle(surface, handle_color, (handle_x, handle_y), self.handle_radius)

    def handle_event(self, event):
        if self.disabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos)
            if self.hovered and not was_hovered:
                self.hover_start_time = time.time()
            if self.dragging:
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if click is on slider area (proportional to height for proper scaling)
            track_y = self.rect.y + int(self.rect.height * 0.57)
            hit_rect = pygame.Rect(self.rect.x - 5, track_y - 10,
                                   self.rect.width + 10, int(self.rect.height * 0.5))
            if hit_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        return False

    def _update_value(self, mouse_x):
        ratio = (mouse_x - self.rect.x) / self.rect.width
        ratio = max(0.0, min(1.0, ratio))
        new_value = self.min_val + ratio * (self.max_val - self.min_val)
        if new_value != self.value:
            self.value = new_value
            if self.on_change:
                self.on_change(self.value)

    def draw_tooltip(self, screen, font, scale=1.0):
        """Draw tooltip if hovered for sufficient time"""
        if self.tooltip and self.hovered:
            elapsed = time.time() - self.hover_start_time
            if elapsed >= 0.5:  # Show tooltip after 500ms
                # Create tooltip surface
                padding = int(6 * scale)
                margin = int(5 * scale)
                cursor_offset = int(16 * scale)  # Offset from cursor
                text_surf = font.render(self.tooltip, True, (255, 255, 255))
                tooltip_w = text_surf.get_width() + padding * 2
                tooltip_h = text_surf.get_height() + padding * 2

                # Position tooltip below and to the right of cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                tooltip_x = mouse_x + cursor_offset
                tooltip_y = mouse_y + cursor_offset

                # Keep tooltip on screen
                screen_w, screen_h = screen.get_size()
                if tooltip_x + tooltip_w > screen_w - margin:
                    tooltip_x = mouse_x - tooltip_w - margin
                if tooltip_y + tooltip_h > screen_h - margin:
                    tooltip_y = mouse_y - tooltip_h - margin

                # Draw tooltip background
                tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_w, tooltip_h)
                border_radius = int(4 * scale)
                pygame.draw.rect(screen, (50, 50, 55), tooltip_rect, border_radius=border_radius)
                pygame.draw.rect(screen, (100, 100, 110), tooltip_rect, 1, border_radius=border_radius)
                screen.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))


class TextInput:
    """Editable text input field"""
    def __init__(self, x, y, width, height, text="", label="", on_change=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.label = label
        self.on_change = on_change
        self.active = False
        self.cursor_pos = len(text)
        self.cursor_visible = True
        self.cursor_timer = 0

    def draw(self, surface, font, scale=1.0):
        # Draw label above
        if self.label:
            label_offset = int(18 * scale)
            label_surf = font.render(self.label, True, (180, 180, 180))
            surface.blit(label_surf, (self.rect.x, self.rect.y - label_offset))

        # Draw background
        bg_color = (70, 70, 80) if self.active else (50, 50, 60)
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=4)

        # Draw border
        border_color = (100, 180, 255) if self.active else (80, 80, 90)
        pygame.draw.rect(surface, border_color, self.rect, 1, border_radius=4)

        # Draw text (clipped to fit)
        text_surf = font.render(self.text, True, (220, 220, 220))
        text_x = self.rect.x + 5
        text_y = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        # Clip text to fit within rect
        clip_rect = pygame.Rect(text_x, text_y, self.rect.width - 10, text_surf.get_height())
        surface.set_clip(clip_rect)
        surface.blit(text_surf, (text_x, text_y))
        surface.set_clip(None)

        # Draw cursor
        if self.active and self.cursor_visible:
            cursor_x = text_x + font.size(self.text[:self.cursor_pos])[0]
            cursor_x = min(cursor_x, self.rect.right - 5)
            pygame.draw.line(surface, (220, 220, 220),
                           (cursor_x, text_y), (cursor_x, text_y + text_surf.get_height()), 1)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            if self.active:
                self.cursor_pos = len(self.text)
                self.cursor_visible = True
                self.cursor_timer = 0
            return self.active or was_active
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
                    if self.on_change:
                        self.on_change(self.text)
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
                    if self.on_change:
                        self.on_change(self.text)
            elif event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                self.active = False
            elif event.unicode and event.unicode.isprintable():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
                if self.on_change:
                    self.on_change(self.text)
            return True
        return False

    def update(self, dt):
        """Update cursor blink"""
        if self.active:
            self.cursor_timer += dt
            if self.cursor_timer >= 0.5:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0


class NumericEntry:
    """Numeric entry field with min/max validation"""
    def __init__(self, x, y, width, height, min_val, max_val, value, label, on_change=None, tooltip=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.on_change = on_change
        self.tooltip = tooltip
        self.text = str(value)
        self.active = False
        self.cursor_pos = len(self.text)
        self.cursor_visible = True
        self.cursor_timer = 0
        self.hovered = False
        self.hover_start_time = 0
        self.disabled = False

    def draw(self, surface, font):
        # Colors based on disabled state
        if self.disabled:
            label_color = (100, 100, 100)
            text_color = (100, 100, 100)
            bg_color = (40, 40, 45)
            border_color = (60, 60, 65)
        else:
            label_color = (180, 180, 180)
            text_color = (220, 220, 220)
            bg_color = (70, 70, 80) if self.active else (50, 50, 60)
            border_color = (100, 180, 255) if self.active else (80, 80, 90)

        # Draw label on left
        label_surf = font.render(self.label, True, label_color)
        label_y = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (self.rect.x, label_y))

        # Entry field on right side of label
        entry_x = self.rect.x + label_surf.get_width() + 8
        entry_width = self.rect.width - label_surf.get_width() - 8
        entry_rect = pygame.Rect(entry_x, self.rect.y, entry_width, self.rect.height)

        # Draw background
        pygame.draw.rect(surface, bg_color, entry_rect, border_radius=4)

        # Draw border
        pygame.draw.rect(surface, border_color, entry_rect, 1, border_radius=4)

        # Draw text centered
        text_surf = font.render(self.text, True, text_color)
        text_x = entry_rect.x + (entry_rect.width - text_surf.get_width()) // 2
        text_y = entry_rect.y + (entry_rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (text_x, text_y))

        # Draw cursor when active (positioned relative to centered text)
        if self.active and self.cursor_visible:
            cursor_x = text_x + font.size(self.text[:self.cursor_pos])[0]
            pygame.draw.line(surface, (220, 220, 220),
                           (cursor_x, text_y), (cursor_x, text_y + text_surf.get_height()), 1)

        # Store entry_rect for hit testing
        self._entry_rect = entry_rect

    def _validate_and_apply(self):
        """Validate current text and apply the value"""
        try:
            val = int(self.text)
            val = max(self.min_val, min(self.max_val, val))
            self.value = val
            self.text = str(val)
            if self.on_change:
                self.on_change(val)
        except ValueError:
            # Revert to previous value
            self.text = str(self.value)
        self.cursor_pos = len(self.text)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            was_hovered = self.hovered
            self.hovered = self.rect.collidepoint(event.pos)
            if self.hovered and not was_hovered:
                self.hover_start_time = time.time()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.disabled:
                return False
            was_active = self.active
            entry_rect = getattr(self, '_entry_rect', self.rect)
            self.active = entry_rect.collidepoint(event.pos)
            if self.active:
                self.cursor_pos = len(self.text)
                self.cursor_visible = True
                self.cursor_timer = 0
            elif was_active:
                # Lost focus - validate
                self._validate_and_apply()
            return self.active or was_active
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_pos > 0:
                    self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                    self.cursor_pos -= 1
            elif event.key == pygame.K_DELETE:
                if self.cursor_pos < len(self.text):
                    self.text = self.text[:self.cursor_pos] + self.text[self.cursor_pos+1:]
            elif event.key == pygame.K_LEFT:
                self.cursor_pos = max(0, self.cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor_pos = min(len(self.text), self.cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.cursor_pos = 0
            elif event.key == pygame.K_END:
                self.cursor_pos = len(self.text)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                self._validate_and_apply()
                self.active = False
            elif event.unicode and event.unicode.isdigit():
                self.text = self.text[:self.cursor_pos] + event.unicode + self.text[self.cursor_pos:]
                self.cursor_pos += 1
            return True
        return False

    def draw_tooltip(self, screen, font, scale=1.0):
        """Draw tooltip if hovered for sufficient time"""
        if self.tooltip and self.hovered:
            elapsed = time.time() - self.hover_start_time
            if elapsed >= 0.5:  # Show tooltip after 500ms
                # Create tooltip surface
                padding = int(6 * scale)
                margin = int(5 * scale)
                cursor_offset = int(16 * scale)  # Offset from cursor
                text_surf = font.render(self.tooltip, True, (255, 255, 255))
                tooltip_w = text_surf.get_width() + padding * 2
                tooltip_h = text_surf.get_height() + padding * 2

                # Position tooltip below and to the right of cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                tooltip_x = mouse_x + cursor_offset
                tooltip_y = mouse_y + cursor_offset

                # Keep tooltip on screen
                screen_w, screen_h = screen.get_size()
                if tooltip_x + tooltip_w > screen_w - margin:
                    tooltip_x = mouse_x - tooltip_w - margin
                if tooltip_y + tooltip_h > screen_h - margin:
                    tooltip_y = mouse_y - tooltip_h - margin

                # Draw tooltip background
                tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_w, tooltip_h)
                border_radius = int(4 * scale)
                pygame.draw.rect(screen, (50, 50, 55), tooltip_rect, border_radius=border_radius)
                pygame.draw.rect(screen, (100, 100, 110), tooltip_rect, 1, border_radius=border_radius)
                screen.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))

    def update(self, dt):
        """Update cursor blink"""
        if self.active:
            self.cursor_timer += dt
            if self.cursor_timer >= 0.5:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0


class UndoManager:
    """Manages undo/redo state for the application (supports hierarchy)"""
    def __init__(self, max_history=50):
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = max_history

    def _serialize_items(self, items: List[SceneItem]) -> List[dict]:
        """Serialize a list of scene items to dicts."""
        return [item.to_dict() for item in items]

    def _deserialize_items(self, items_data: List[dict]) -> List[SceneItem]:
        """Deserialize a list of dicts to scene items."""
        result = []
        for item_data in items_data:
            if item_data.get('type') == 'group':
                result.append(Group.from_dict(item_data))
            else:
                result.append(Shape.from_dict(item_data))
        return result

    def save_state(self, scene_root: List[SceneItem]):
        """Save current state to undo stack"""
        # Serialize scene_root to dicts (handles both shapes and groups)
        state = self._serialize_items(scene_root)
        self.undo_stack.append(state)
        # Clear redo stack when new action is performed
        self.redo_stack = []
        # Limit history size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

    def undo(self, current_scene_root: List[SceneItem]) -> Optional[List[SceneItem]]:
        """Undo last action, returns new scene_root or None if nothing to undo"""
        if not self.undo_stack:
            return None
        # Save current state to redo stack
        current_state = self._serialize_items(current_scene_root)
        self.redo_stack.append(current_state)
        # Restore previous state
        prev_state = self.undo_stack.pop()
        return self._deserialize_items(prev_state)

    def redo(self, current_scene_root: List[SceneItem]) -> Optional[List[SceneItem]]:
        """Redo last undone action, returns new scene_root or None if nothing to redo"""
        if not self.redo_stack:
            return None
        # Save current state to undo stack
        current_state = self._serialize_items(current_scene_root)
        self.undo_stack.append(current_state)
        # Restore redo state
        next_state = self.redo_stack.pop()
        return self._deserialize_items(next_state)

    def can_undo(self):
        return len(self.undo_stack) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0

    def clear(self):
        """Clear all history"""
        self.undo_stack = []
        self.redo_stack = []


class Toast:
    """Toast notification for displaying temporary messages"""
    def __init__(self, message, duration=3.0, toast_type="info"):
        self.message = message
        self.duration = duration
        self.toast_type = toast_type  # "info", "error", "success"
        self.created_at = time.time()
        self.alpha = 255

    def is_expired(self):
        elapsed = time.time() - self.created_at
        return elapsed > self.duration

    def get_alpha(self):
        """Get current alpha for fade out effect"""
        elapsed = time.time() - self.created_at
        fade_start = self.duration - 0.5  # Start fading 0.5s before expiry
        if elapsed > fade_start:
            fade_progress = (elapsed - fade_start) / 0.5
            return int(255 * (1 - fade_progress))
        return 255

    def draw(self, screen, font, y_offset=0, scale=1.0):
        """Draw toast notification"""
        alpha = self.get_alpha()
        if alpha <= 0:
            return 0

        # Colors based on type
        if self.toast_type == "error":
            bg_color = (180, 50, 50)
            border_color = (220, 80, 80)
        elif self.toast_type == "success":
            bg_color = (50, 150, 50)
            border_color = (80, 200, 80)
        else:  # info
            bg_color = (60, 60, 70)
            border_color = (100, 100, 120)

        padding = int(12 * scale)
        text_surf = font.render(self.message, True, (255, 255, 255))
        toast_w = text_surf.get_width() + padding * 2
        toast_h = text_surf.get_height() + padding * 2

        screen_w, screen_h = screen.get_size()
        toast_x = (screen_w - toast_w) // 2
        toast_y = screen_h - int(80 * scale) - y_offset

        # Create surface with alpha
        border_radius = int(6 * scale)
        toast_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
        pygame.draw.rect(toast_surf, (*bg_color, alpha), (0, 0, toast_w, toast_h), border_radius=border_radius)
        pygame.draw.rect(toast_surf, (*border_color, alpha), (0, 0, toast_w, toast_h), 2, border_radius=border_radius)

        # Render text with alpha
        text_surf.set_alpha(alpha)
        toast_surf.blit(text_surf, (padding, padding))

        screen.blit(toast_surf, (toast_x, toast_y))
        return toast_h + int(10 * scale)  # Return height for stacking


class ContextMenu:
    """Popup context menu for hierarchy panel"""
    def __init__(self):
        self.visible = False
        self.x = 0
        self.y = 0
        self.items: List[Tuple[str, callable]] = []
        self.width = 150
        self.item_height = 28
        self.hovered_index = -1

    def show(self, x: int, y: int, items: List[Tuple[str, callable]]):
        """Show menu at position with given items [(label, callback), ...]"""
        self.visible = True
        self.x = x
        self.y = y
        self.items = items
        self.hovered_index = -1

    def hide(self):
        """Hide the context menu"""
        self.visible = False
        self.items = []

    def get_rect(self) -> pygame.Rect:
        """Get the menu's bounding rectangle"""
        height = len(self.items) * self.item_height + 8
        return pygame.Rect(self.x, self.y, self.width, height)

    def handle_event(self, event) -> bool:
        """Handle event, returns True if event was consumed"""
        if not self.visible:
            return False

        if event.type == pygame.MOUSEMOTION:
            rect = self.get_rect()
            if rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.y - 4
                self.hovered_index = rel_y // self.item_height
                if self.hovered_index >= len(self.items):
                    self.hovered_index = -1
            else:
                self.hovered_index = -1
            return True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            rect = self.get_rect()
            if rect.collidepoint(event.pos):
                if event.button == 1 and 0 <= self.hovered_index < len(self.items):
                    # Execute callback
                    _, callback = self.items[self.hovered_index]
                    self.hide()
                    if callback:
                        callback()
                    return True
            # Click outside - close menu
            self.hide()
            return True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide()
                return True

        return False

    def draw(self, surface, font):
        """Draw the context menu"""
        if not self.visible:
            return

        rect = self.get_rect()

        # Background
        pygame.draw.rect(surface, (50, 50, 55), rect, border_radius=4)
        pygame.draw.rect(surface, (80, 80, 90), rect, 1, border_radius=4)

        # Items
        y = self.y + 4
        for i, (label, _) in enumerate(self.items):
            item_rect = pygame.Rect(self.x + 4, y, self.width - 8, self.item_height)

            # Highlight hovered item
            if i == self.hovered_index:
                pygame.draw.rect(surface, (70, 100, 150), item_rect, border_radius=3)

            # Text
            text_surf = font.render(label, True, (220, 220, 220))
            surface.blit(text_surf, (self.x + 12, y + (self.item_height - text_surf.get_height()) // 2))

            y += self.item_height


class HierarchyPanel:
    """
    Collapsible sidebar panel showing scene hierarchy tree.
    Supports expand/collapse groups, selection, drag-drop reordering.
    """
    # Colors
    BG_COLOR = (35, 35, 40)
    HEADER_COLOR = (45, 45, 50)
    ROW_COLOR = (40, 40, 45)
    ROW_HOVER_COLOR = (55, 55, 65)
    ROW_SELECTED_COLOR = (60, 90, 140)
    TEXT_COLOR = (220, 220, 220)
    TEXT_DIM_COLOR = (140, 140, 140)
    GROUP_ICON_COLOR = (255, 200, 100)
    SHAPE_ICON_COLOR = (100, 200, 255)
    RESIZE_HANDLE_COLOR = (80, 80, 90)
    DROP_LINE_COLOR = (100, 200, 255)
    DROP_INTO_COLOR = (80, 150, 220, 100)

    MIN_WIDTH = 150
    MAX_WIDTH = 400
    HEADER_HEIGHT = 70  # Includes search bar
    ROW_HEIGHT = 26
    INDENT_SIZE = 18
    ICON_SIZE = 14
    RESIZE_HANDLE_WIDTH = 6

    # Collapsed state constants
    COLLAPSED_WIDTH = 24        # Width of collapsed tab
    HANDLE_WIDTH = 24           # Width of the handle/tab
    CHEVRON_SIZE = 12           # Size of expansion chevron

    def __init__(self, app):
        self.app = app
        self.width = 250
        self.scroll_offset = 0
        self.max_scroll = 0
        self.hovered_row = None
        self.dragging_resize = False
        self.search_text = ""
        self.search_active = False  # Is search input focused
        self.search_cursor = 0

        # Collapsed state
        self.collapsed = True       # Start collapsed
        self.hover_handle = False   # Is mouse over the handle

        # Animation state
        self.animation_progress = 0.0  # 0=collapsed, 1=expanded
        self.animating = False
        self.animation_start = 0.0
        self.animation_duration = 0.15  # 150ms

        # Drag-drop state
        self.drag_start_pos = None
        self.drag_start_item = None  # The actual item clicked for drag
        self.dragging_item = None
        self.drag_threshold = 5
        self.drop_target = None  # (item, position: "before"|"into"|"after")

        # Visible rows cache (recalculated each frame)
        self._visible_rows: List[Tuple[SceneItem, int]] = []  # (item, indent_level)

        # Inline rename state
        self.rename_item: Optional[SceneItem] = None  # Item being renamed
        self.rename_text: str = ""  # Current text in rename field
        self.rename_cursor: int = 0  # Cursor position in rename field

        # Double-click detection for rename
        self.last_click_item: Optional[SceneItem] = None
        self.last_click_time: float = 0.0
        self.double_click_threshold: float = 0.4  # 400ms

        # Chevron tooltip state
        self.chevron_hovered = False
        self.chevron_hover_start = 0.0

        # Search field tooltip state
        self.search_hovered = False
        self.search_hover_start = 0.0

    def update_animation(self):
        """Update panel animation state"""
        if not self.animating:
            return

        import time
        elapsed = time.time() - self.animation_start
        t = min(1.0, elapsed / self.animation_duration)

        # Ease-out: 1 - (1 - t)^2
        eased = 1 - (1 - t) ** 2

        if self.collapsed:
            # Collapsing: 1.0 -> 0.0
            self.animation_progress = 1.0 - eased
        else:
            # Expanding: 0.0 -> 1.0
            self.animation_progress = eased

        if t >= 1.0:
            self.animating = False
            self.animation_progress = 0.0 if self.collapsed else 1.0

    def get_rect(self) -> pygame.Rect:
        """Get panel rectangle based on animation progress"""
        scale = self.app.ui_scale
        handle_w = int(self.HANDLE_WIDTH * scale)
        expanded_w = int(self.width * scale)

        # Interpolate width based on animation progress
        current_w = handle_w + int((expanded_w - handle_w) * self.animation_progress)
        return pygame.Rect(0, 0, current_w, HEIGHT)

    def _get_handle_rect(self) -> pygame.Rect:
        """Get the clickable handle rectangle at right edge of panel"""
        scale = self.app.ui_scale
        handle_w = int(self.HANDLE_WIDTH * scale)
        panel_rect = self.get_rect()
        return pygame.Rect(panel_rect.width - handle_w, 0, handle_w, HEIGHT)

    def get_content_rect(self) -> pygame.Rect:
        """Get content area rectangle (excluding header)"""
        header_h = int(self.HEADER_HEIGHT * self.app.ui_scale)
        return pygame.Rect(0, header_h, int(self.width * self.app.ui_scale), HEIGHT - header_h)

    def get_resize_rect(self) -> pygame.Rect:
        """Get resize handle rectangle"""
        w = int(self.RESIZE_HANDLE_WIDTH * self.app.ui_scale)
        return pygame.Rect(int(self.width * self.app.ui_scale) - w, 0, w, HEIGHT)

    def _build_visible_rows(self):
        """Build list of visible rows based on expanded state and search filter"""
        self._visible_rows = []
        search = self.search_text.lower().strip()

        def should_show(item: SceneItem) -> bool:
            """Check if item or any descendant matches search"""
            if not search:
                return True
            if search in item.name.lower():
                return True
            if isinstance(item, Group):
                return any(should_show(child) for child in item.children)
            return False

        def add_items(items: List[SceneItem], indent: int):
            for item in items:
                if should_show(item):
                    self._visible_rows.append((item, indent))
                    if isinstance(item, Group) and item.expanded:
                        add_items(item.children, indent + 1)

        add_items(self.app.scene_root, 0)

    def _get_row_at(self, y: int) -> Optional[Tuple[SceneItem, int, pygame.Rect]]:
        """Get row at y position, returns (item, index, rect) or None"""
        content_rect = self.get_content_rect()
        if y < content_rect.top:
            return None

        row_h = int(self.ROW_HEIGHT * self.app.ui_scale)
        rel_y = y - content_rect.top + self.scroll_offset
        row_index = rel_y // row_h

        if 0 <= row_index < len(self._visible_rows):
            item, indent = self._visible_rows[row_index]
            row_y = content_rect.top + row_index * row_h - self.scroll_offset
            row_rect = pygame.Rect(0, row_y, content_rect.width, row_h)
            return (item, row_index, row_rect)

        return None

    def _get_drop_position(self, y: int, row_rect: pygame.Rect) -> str:
        """Determine drop position within a row: 'before', 'into', or 'after'"""
        rel_y = y - row_rect.top
        third = row_rect.height / 3

        if rel_y < third:
            return "before"
        elif rel_y > third * 2:
            return "after"
        else:
            return "into"

    def _is_descendant(self, potential_descendant: SceneItem, ancestor: SceneItem) -> bool:
        """Check if potential_descendant is a descendant of ancestor."""
        if not isinstance(ancestor, Group):
            return False
        current = potential_descendant._parent
        while current is not None:
            if current is ancestor:
                return True
            current = current._parent
        return False

    def _get_search_rect(self) -> pygame.Rect:
        """Get search input rectangle (accounting for handle on right edge)"""
        scale = self.app.ui_scale
        margin = int(8 * scale)
        title_height = int(28 * scale)
        input_height = int(26 * scale)
        # Content width is panel width minus handle
        content_w = int(self.width * scale) - int(self.HANDLE_WIDTH * scale)
        return pygame.Rect(margin, title_height, content_w - margin * 2, input_height)

    def handle_event(self, event) -> bool:
        """Handle event, returns True if event was consumed"""
        import time
        handle_rect = self._get_handle_rect()
        panel_rect = self.get_rect()

        # Handle click on handle (toggle) - works in both collapsed and expanded states
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Block clicks during animation to prevent state desync
            # Return True if click is in panel area to consume it and prevent click-through
            if self.animating:
                return panel_rect.collidepoint(event.pos)

            # Check header chevron click (when expanded)
            if not self.collapsed and self.animation_progress > 0.95:
                scale = self.app.ui_scale
                handle_w = int(self.HANDLE_WIDTH * scale)
                chevron_size = int(self.CHEVRON_SIZE * scale)
                chevron_x = handle_w // 2
                chevron_y = int(20 * scale)
                # Create clickable area around chevron
                chevron_rect = pygame.Rect(
                    chevron_x - chevron_size,
                    chevron_y - chevron_size,
                    chevron_size * 2,
                    chevron_size * 2
                )
                if chevron_rect.collidepoint(event.pos):
                    self.collapsed = True
                    self.animating = True
                    self.animation_start = time.time()
                    return True

            if handle_rect.collidepoint(event.pos):
                self.collapsed = not self.collapsed
                self.animating = True
                self.animation_start = time.time()
                return True

        # Handle hover state for handle and chevron tooltip
        if event.type == pygame.MOUSEMOTION:
            was_hovering = self.hover_handle
            self.hover_handle = handle_rect.collidepoint(event.pos)
            if self.hover_handle:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            elif was_hovering:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

            # Track hover for tooltip - handle area OR header chevron (when expanded)
            was_chevron_hovered = self.chevron_hovered
            hover_target = handle_rect.collidepoint(event.pos)
            # Also check header chevron when expanded
            if not self.collapsed and self.animation_progress > 0.95:
                scale = self.app.ui_scale
                handle_w = int(self.HANDLE_WIDTH * scale)
                chevron_size = int(self.CHEVRON_SIZE * scale)
                chevron_x = handle_w // 2
                chevron_y = int(20 * scale)
                header_chevron_rect = pygame.Rect(
                    chevron_x - chevron_size,
                    chevron_y - chevron_size,
                    chevron_size * 2,
                    chevron_size * 2
                )
                hover_target = hover_target or header_chevron_rect.collidepoint(event.pos)
            self.chevron_hovered = hover_target
            if self.chevron_hovered and not was_chevron_hovered:
                self.chevron_hover_start = time.time()

            # Track search field hover for tooltip
            if not self.collapsed and self.animation_progress > 0.95:
                search_rect = self._get_search_rect()
                was_search_hovered = self.search_hovered
                self.search_hovered = search_rect.collidepoint(event.pos)
                if self.search_hovered and not was_search_hovered:
                    self.search_hover_start = time.time()
            else:
                self.search_hovered = False

        # If fully collapsed, only handle within handle area
        if self.collapsed and self.animation_progress < 0.05:
            return handle_rect.collidepoint(getattr(event, 'pos', (-1, -1)))

        # Expanded state handling
        resize_rect = self.get_resize_rect()
        search_rect = self._get_search_rect()

        # Handle inline rename keyboard events (takes priority)
        if self.rename_item is not None and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._finish_rename(apply=False)
                return True
            elif event.key == pygame.K_RETURN:
                self._finish_rename(apply=True)
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self.rename_cursor > 0:
                    self.rename_text = self.rename_text[:self.rename_cursor-1] + self.rename_text[self.rename_cursor:]
                    self.rename_cursor -= 1
                return True
            elif event.key == pygame.K_DELETE:
                if self.rename_cursor < len(self.rename_text):
                    self.rename_text = self.rename_text[:self.rename_cursor] + self.rename_text[self.rename_cursor+1:]
                return True
            elif event.key == pygame.K_LEFT:
                self.rename_cursor = max(0, self.rename_cursor - 1)
                return True
            elif event.key == pygame.K_RIGHT:
                self.rename_cursor = min(len(self.rename_text), self.rename_cursor + 1)
                return True
            elif event.key == pygame.K_HOME:
                self.rename_cursor = 0
                return True
            elif event.key == pygame.K_END:
                self.rename_cursor = len(self.rename_text)
                return True
            elif event.unicode and event.unicode.isprintable():
                self.rename_text = self.rename_text[:self.rename_cursor] + event.unicode + self.rename_text[self.rename_cursor:]
                self.rename_cursor += 1
                return True
            return True  # Consume all key events while renaming

        # Handle click outside rename field to finish rename
        if self.rename_item is not None and event.type == pygame.MOUSEBUTTONDOWN:
            # Check if click is on the rename row
            row_data = self._get_row_at(event.pos[1])
            if row_data is None or row_data[0] != self.rename_item:
                self._finish_rename(apply=True)
            # Continue processing the click

        # Handle search input keyboard events
        if self.search_active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.search_active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self.search_cursor > 0:
                    self.search_text = self.search_text[:self.search_cursor-1] + self.search_text[self.search_cursor:]
                    self.search_cursor -= 1
                return True
            elif event.key == pygame.K_DELETE:
                if self.search_cursor < len(self.search_text):
                    self.search_text = self.search_text[:self.search_cursor] + self.search_text[self.search_cursor+1:]
                return True
            elif event.key == pygame.K_LEFT:
                self.search_cursor = max(0, self.search_cursor - 1)
                return True
            elif event.key == pygame.K_RIGHT:
                self.search_cursor = min(len(self.search_text), self.search_cursor + 1)
                return True
            elif event.key == pygame.K_HOME:
                self.search_cursor = 0
                return True
            elif event.key == pygame.K_END:
                self.search_cursor = len(self.search_text)
                return True
            elif event.unicode and event.unicode.isprintable():
                self.search_text = self.search_text[:self.search_cursor] + event.unicode + self.search_text[self.search_cursor:]
                self.search_cursor += 1
                return True

        # Handle resize dragging
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if clicking search input
            if search_rect.collidepoint(event.pos):
                self.search_active = True
                return True

            if resize_rect.collidepoint(event.pos):
                self.dragging_resize = True
                return True
            elif panel_rect.collidepoint(event.pos):
                self.search_active = False  # Unfocus search on other clicks

                # Check for row click
                row_data = self._get_row_at(event.pos[1])
                if row_data:
                    item, idx, row_rect = row_data
                    mods = pygame.key.get_mods()

                    # Check for expand/collapse arrow click
                    if isinstance(item, Group):
                        indent = self._visible_rows[idx][1]
                        arrow_x = int((10 + indent * self.INDENT_SIZE) * self.app.ui_scale)
                        if event.pos[0] < arrow_x + int(16 * self.app.ui_scale):
                            item.expanded = not item.expanded
                            return True

                    # Check for double-click to start rename
                    current_time = time.time()
                    if (self.last_click_item == item and
                        current_time - self.last_click_time < self.double_click_threshold and
                        item in self.app.selected_items):
                        # Double-click on already selected item - start rename
                        self._start_rename()
                        self.last_click_item = None
                        self.last_click_time = 0
                        return True

                    # Track for potential double-click
                    self.last_click_item = item
                    self.last_click_time = current_time

                    # Start potential drag - store the actual clicked item
                    self.drag_start_pos = event.pos
                    self.drag_start_item = item

                    # Selection handling - always select the exact clicked item in hierarchy
                    # Ctrl: toggle individual selection, Shift: range selection
                    # Always use force_individual=True so hierarchy selects exact item clicked
                    if mods & pygame.KMOD_CTRL:
                        self.app.select_item(item, add_to_selection=True, force_individual=True)
                    elif mods & pygame.KMOD_SHIFT:
                        self.app.select_item(item, range_select=True, force_individual=True)
                    else:
                        self.app.select_item(item, force_individual=True)

                    return True
                return True  # Consume click in panel area

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging_resize:
                self.dragging_resize = False
                return True

            if self.dragging_item and self.drop_target:
                # Execute drop
                target_item, position = self.drop_target
                self._execute_drop(self.dragging_item, target_item, position)

            self.dragging_item = None
            self.drag_start_pos = None
            self.drag_start_item = None
            self.drop_target = None

        elif event.type == pygame.MOUSEMOTION:
            # Update cursor for resize handle
            if resize_rect.collidepoint(event.pos) or self.dragging_resize:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
            elif panel_rect.collidepoint(event.pos):
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

            if self.dragging_resize:
                new_width = event.pos[0] / self.app.ui_scale
                self.width = max(self.MIN_WIDTH, min(self.MAX_WIDTH, new_width))
                return True

            # Check for drag start
            if self.drag_start_pos and not self.dragging_item:
                dist = math.dist(self.drag_start_pos, event.pos)
                if dist > self.drag_threshold and self.drag_start_item:
                    self.dragging_item = self.drag_start_item

            # Update drop target during drag
            if self.dragging_item:
                row_data = self._get_row_at(event.pos[1])
                if row_data:
                    target_item, idx, row_rect = row_data
                    # Don't drop onto self or descendants
                    if target_item != self.dragging_item and not self._is_descendant(target_item, self.dragging_item):
                        position = self._get_drop_position(event.pos[1], row_rect)
                        # Can only drop "into" a Group
                        if position == "into" and not isinstance(target_item, Group):
                            position = "after"

                        self.drop_target = (target_item, position)
                    else:
                        self.drop_target = None
                else:
                    # Mouse is below all rows - allow drop at end of root level
                    if self.app.scene_root:
                        last_root_item = self.app.scene_root[-1]
                        if last_root_item != self.dragging_item and not self._is_descendant(last_root_item, self.dragging_item):
                            self.drop_target = (last_root_item, "after_root")
                        else:
                            self.drop_target = None
                    else:
                        self.drop_target = None
                return True

            # Update hovered row
            if panel_rect.collidepoint(event.pos):
                row_data = self._get_row_at(event.pos[1])
                self.hovered_row = row_data[0] if row_data else None
            else:
                self.hovered_row = None

        elif event.type == pygame.MOUSEWHEEL:
            if panel_rect.collidepoint(pygame.mouse.get_pos()):
                row_h = int(self.ROW_HEIGHT * self.app.ui_scale)
                self.scroll_offset = max(0, min(self.max_scroll,
                    self.scroll_offset - event.y * row_h * 3))
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # Right click - context menu
            if panel_rect.collidepoint(event.pos):
                row_data = self._get_row_at(event.pos[1])
                if row_data:
                    item = row_data[0]
                    if item not in self.app.selected_items:
                        self.app.select_item(item, force_individual=True)
                    self._show_context_menu(event.pos)
                return True

        return False

    def _execute_drop(self, item: SceneItem, target: SceneItem, position: str):
        """Execute a drag-drop operation - WYSIWYG behavior.

        The drop position matches exactly what the visual indicator shows:
        - "before": Line above target → insert before target at same level
        - "after": Line below target → insert after target at same level
        - "into": Row highlight on Group → insert into Group at end
        - "after_root": Below all items → insert at end of root level
        """
        self.app.save_undo_state()

        if position == "into" and isinstance(target, Group):
            # Drop INTO group (appends at end)
            self.app.reparent_item(item, target, preserve_world_pos=True)

        elif position == "before":
            # Drop BEFORE target (at same level as target)
            parent = target._parent
            container = parent.children if parent else self.app.scene_root
            idx = container.index(target)
            # Adjust if moving within same container
            if item._parent == parent and item in container:
                if container.index(item) < idx:
                    idx -= 1
            self.app.reparent_item(item, parent, index=idx, preserve_world_pos=True)

        elif position == "after":
            # Drop AFTER target (at same level as target)
            parent = target._parent
            container = parent.children if parent else self.app.scene_root
            idx = container.index(target) + 1
            # Adjust if moving within same container
            if item._parent == parent and item in container:
                if container.index(item) < idx:
                    idx -= 1
            self.app.reparent_item(item, parent, index=idx, preserve_world_pos=True)

        elif position == "after_root":
            # Drop at end of root level
            self.app.reparent_item(item, None, preserve_world_pos=True)

    def _show_context_menu(self, pos: Tuple[int, int]):
        """Show context menu at position"""
        items = []

        # Delete option
        items.append(("Delete", lambda: self.app.delete_selected_items()))

        # Rename option
        if len(self.app.selected_items) == 1:
            items.append(("Rename", lambda: self._start_rename()))

        # Group option
        if len(self.app.selected_items) >= 1:
            items.append(("Group", lambda: self.app.group_selected_items()))

        # Ungroup option
        if any(isinstance(item, Group) for item in self.app.selected_items):
            items.append(("Ungroup", lambda: self.app.ungroup_selected()))

        self.app.context_menu.show(pos[0], pos[1], items)

    def _start_rename(self):
        """Start renaming the selected item"""
        if self.app.selected_items:
            item = self.app.selected_items[0]
            self.rename_item = item
            self.rename_text = item.name
            self.rename_cursor = len(self.rename_text)

    def _finish_rename(self, apply: bool = True):
        """Finish renaming - apply changes if apply is True"""
        if self.rename_item and apply and self.rename_text.strip():
            self.rename_item.name = self.rename_text.strip()
            # Update property panel if this item is selected
            if self.app.name_input and self.rename_item in self.app.selected_items:
                self.app.name_input.text = self.rename_item.name
                self.app.name_input.cursor_pos = len(self.app.name_input.text)
        self.rename_item = None
        self.rename_text = ""
        self.rename_cursor = 0

    def _draw_layers_icon(self, surface, cx: int, cy: int, scale: float):
        """Draw a layers icon (3 stacked horizontal lines)"""
        line_width = int(10 * scale)
        line_height = int(2 * scale)
        gap = int(4 * scale)
        color = (150, 150, 160)

        for i in range(3):
            y = cy - gap + (i * gap)
            rect = pygame.Rect(cx - line_width // 2, y, line_width, line_height)
            pygame.draw.rect(surface, color, rect)

    def _draw_handle(self, surface, panel_rect: pygame.Rect):
        """Draw the handle on the right edge of the panel"""
        scale = self.app.ui_scale
        handle_w = int(self.HANDLE_WIDTH * scale)

        # Position handle at right edge of panel
        handle_x = panel_rect.width - handle_w

        # Background
        bg_color = (50, 50, 55) if self.hover_handle else (40, 40, 45)
        handle_rect = pygame.Rect(handle_x, 0, handle_w, HEIGHT)
        pygame.draw.rect(surface, bg_color, handle_rect)

        # Left border (separator from content)
        pygame.draw.line(surface, (60, 60, 70),
                         (handle_x, 0), (handle_x, HEIGHT), 1)

        # Chevron at top - direction based on collapsed state
        # Fixed position at collapsed handle location (right edge of screen)
        chevron_size = int(self.CHEVRON_SIZE * scale)
        cx = handle_w // 2  # Fixed x position regardless of panel state
        cy = int(20 * scale)  # Top position with margin

        if self.collapsed:
            # Right chevron (>) - expand
            points = [
                (cx - chevron_size // 3, cy - chevron_size // 2),
                (cx + chevron_size // 3, cy),
                (cx - chevron_size // 3, cy + chevron_size // 2)
            ]
        else:
            # Left chevron (<) - collapse
            points = [
                (cx + chevron_size // 3, cy - chevron_size // 2),
                (cx - chevron_size // 3, cy),
                (cx + chevron_size // 3, cy + chevron_size // 2)
            ]
        pygame.draw.lines(surface, (180, 180, 180), False, points, 2)

        # Draw vertical "Hierarchy" text when collapsed
        if self.animation_progress < 0.05:
            text_surf = self.app.font_small.render("Hierarchy", True, (180, 180, 180))
            rotated = pygame.transform.rotate(text_surf, -90)  # 90 degrees clockwise
            text_x = cx - rotated.get_width() // 2
            text_y = cy + chevron_size + int(10 * scale)
            surface.blit(rotated, (text_x, text_y))

    def draw(self, surface):
        """Draw the hierarchy panel"""
        # Update animation state
        self.update_animation()

        scale = self.app.ui_scale
        panel_rect = self.get_rect()
        handle_w = int(self.HANDLE_WIDTH * scale)

        # Always draw the handle on the right edge
        self._draw_handle(surface, panel_rect)

        # Only draw panel content if there's room (animation progress > threshold)
        if self.animation_progress < 0.05:
            return

        self._build_visible_rows()

        content_rect = self.get_content_rect()
        header_h = int(self.HEADER_HEIGHT * scale)
        row_h = int(self.ROW_HEIGHT * scale)

        # Content width (excluding handle)
        content_w = panel_rect.width - handle_w

        # Background (excluding handle area)
        bg_rect = pygame.Rect(0, 0, content_w, HEIGHT)
        pygame.draw.rect(surface, self.BG_COLOR, bg_rect)

        # Header (excluding handle area)
        header_rect = pygame.Rect(0, 0, content_w, header_h)
        pygame.draw.rect(surface, self.HEADER_COLOR, header_rect)

        # Header chevron and title - position matches grab bar chevron for visual continuity
        chevron_size = int(self.CHEVRON_SIZE * scale)
        chevron_x = handle_w // 2
        chevron_y = int(20 * scale)

        # Draw left chevron (<) - collapse indicator
        points = [
            (chevron_x + chevron_size // 3, chevron_y - chevron_size // 2),
            (chevron_x - chevron_size // 3, chevron_y),
            (chevron_x + chevron_size // 3, chevron_y + chevron_size // 2)
        ]
        pygame.draw.lines(surface, self.TEXT_COLOR, False, points, 2)

        # Title shifted right to accommodate chevron
        title_surf = self.app.font.render("Hierarchy", True, self.TEXT_COLOR)
        surface.blit(title_surf, (chevron_x + chevron_size, int(4 * scale)))

        # Search input
        search_rect = self._get_search_rect()
        search_bg = (55, 55, 60) if self.search_active else (45, 45, 50)
        pygame.draw.rect(surface, search_bg, search_rect, border_radius=3)
        pygame.draw.rect(surface, (70, 70, 80) if self.search_active else (60, 60, 65), search_rect, 1, border_radius=3)

        # Search text or placeholder
        text_x = search_rect.x + int(6 * scale)
        text_y = search_rect.y + (search_rect.height - self.app.font_small.get_height()) // 2
        if self.search_text:
            text_surf = self.app.font_small.render(self.search_text, True, self.TEXT_COLOR)
            surface.blit(text_surf, (text_x, text_y))
            # Cursor
            if self.search_active:
                cursor_text = self.search_text[:self.search_cursor]
                cursor_x = text_x + self.app.font_small.size(cursor_text)[0]
                pygame.draw.line(surface, (200, 200, 200), (cursor_x, text_y + 2),
                               (cursor_x, text_y + self.app.font_small.get_height() - 2), 1)
        else:
            placeholder_surf = self.app.font_small.render("Search...", True, self.TEXT_DIM_COLOR)
            surface.blit(placeholder_surf, (text_x, text_y))

        # Calculate max scroll
        total_height = len(self._visible_rows) * row_h
        self.max_scroll = max(0, total_height - content_rect.height)
        self.scroll_offset = min(self.scroll_offset, self.max_scroll)

        # Draw rows (with clipping to content area, excluding handle)
        content_clip = pygame.Rect(0, content_rect.top, content_w, content_rect.height)
        clip_rect = surface.get_clip()
        surface.set_clip(content_clip)

        y = content_rect.top - self.scroll_offset
        for item, indent in self._visible_rows:
            if y + row_h > content_rect.top and y < content_rect.bottom:
                self._draw_row(surface, item, indent, y, row_h, content_w)
            y += row_h

        # Draw drop indicator
        if self.drop_target:
            target_item, position = self.drop_target
            if position == "after_root":
                # Draw line at the bottom of visible rows (root level drop)
                line_y = content_rect.top + len(self._visible_rows) * row_h - self.scroll_offset
                pygame.draw.line(surface, self.DROP_LINE_COLOR,
                    (int(10 * scale), line_y), (content_w - int(10 * scale), line_y), 2)
                pygame.draw.circle(surface, self.DROP_LINE_COLOR, (int(10 * scale), line_y), 4)
            else:
                for i, (item, indent) in enumerate(self._visible_rows):
                    if item == target_item:
                        row_y = content_rect.top + i * row_h - self.scroll_offset
                        if position == "into":
                            # Highlight entire row
                            drop_rect = pygame.Rect(0, row_y, content_w, row_h)
                            drop_surf = pygame.Surface((drop_rect.width, drop_rect.height), pygame.SRCALPHA)
                            drop_surf.fill(self.DROP_INTO_COLOR)
                            surface.blit(drop_surf, drop_rect.topleft)
                        elif position == "before":
                            # Line above, indented to match target level
                            line_x = int((8 + indent * self.INDENT_SIZE) * scale)
                            pygame.draw.line(surface, self.DROP_LINE_COLOR,
                                (line_x, row_y), (content_w - int(10 * scale), row_y), 2)
                            pygame.draw.circle(surface, self.DROP_LINE_COLOR, (line_x, row_y), 4)
                        else:  # after
                            # Line below, indented to match target level
                            line_x = int((8 + indent * self.INDENT_SIZE) * scale)
                            line_y = row_y + row_h
                            pygame.draw.line(surface, self.DROP_LINE_COLOR,
                                (line_x, line_y), (content_w - int(10 * scale), line_y), 2)
                            pygame.draw.circle(surface, self.DROP_LINE_COLOR, (line_x, line_y), 4)
                        break

        surface.set_clip(clip_rect)

    def _draw_row(self, surface, item: SceneItem, indent: int, y: int, row_h: int, width: int):
        """Draw a single row"""
        scale = self.app.ui_scale
        is_selected = item in self.app.selected_items
        is_hovered = item == self.hovered_row

        # Background
        if is_selected:
            bg_color = self.ROW_SELECTED_COLOR
        elif is_hovered:
            bg_color = self.ROW_HOVER_COLOR
        else:
            bg_color = self.ROW_COLOR

        row_rect = pygame.Rect(0, y, width, row_h)
        pygame.draw.rect(surface, bg_color, row_rect)

        x = int((8 + indent * self.INDENT_SIZE) * scale)

        # Expand/collapse arrow for groups
        if isinstance(item, Group):
            arrow_size = int(8 * scale)
            arrow_y = y + (row_h - arrow_size) // 2
            if item.expanded:
                # Down arrow
                points = [(x, arrow_y), (x + arrow_size, arrow_y),
                         (x + arrow_size // 2, arrow_y + arrow_size)]
            else:
                # Right arrow
                points = [(x, arrow_y), (x + arrow_size, arrow_y + arrow_size // 2),
                         (x, arrow_y + arrow_size)]
            pygame.draw.polygon(surface, self.TEXT_DIM_COLOR, points)
            x += int(14 * scale)

        # Icon
        icon_size = int(self.ICON_SIZE * scale)
        icon_y = y + (row_h - icon_size) // 2
        if isinstance(item, Group):
            # Folder icon (simple rectangle)
            pygame.draw.rect(surface, self.GROUP_ICON_COLOR,
                           (x, icon_y, icon_size, icon_size), border_radius=2)
            pygame.draw.rect(surface, (180, 140, 60),
                           (x, icon_y, icon_size // 2, int(4 * scale)), border_radius=1)
        else:
            # Shape icon (diamond)
            cx, cy = x + icon_size // 2, icon_y + icon_size // 2
            half = icon_size // 2
            points = [(cx, icon_y), (x + icon_size, cy), (cx, icon_y + icon_size), (x, cy)]
            pygame.draw.polygon(surface, self.SHAPE_ICON_COLOR, points)

        x += icon_size + int(6 * scale)

        # Name (or rename input if this item is being renamed)
        max_name_width = width - x - int(10 * scale)
        text_y = y + (row_h - self.app.font_small.get_height()) // 2

        if item == self.rename_item:
            # Draw inline rename input
            input_rect = pygame.Rect(x - 2, y + 2, max_name_width + 4, row_h - 4)
            pygame.draw.rect(surface, (60, 60, 70), input_rect, border_radius=3)
            pygame.draw.rect(surface, (100, 180, 255), input_rect, 1, border_radius=3)

            # Draw rename text
            rename_surf = self.app.font_small.render(self.rename_text, True, (240, 240, 240))
            surface.blit(rename_surf, (x, text_y))

            # Draw cursor
            cursor_x = x + self.app.font_small.size(self.rename_text[:self.rename_cursor])[0]
            pygame.draw.line(surface, (240, 240, 240),
                           (cursor_x, text_y + 2),
                           (cursor_x, text_y + self.app.font_small.get_height() - 2), 1)
        else:
            # Draw normal name
            name = item.name or "(unnamed)"
            text_color = self.TEXT_COLOR if is_selected else self.TEXT_COLOR
            name_surf = self.app.font_small.render(name, True, text_color)

            # Truncate name if too long
            if name_surf.get_width() > max_name_width:
                # Truncate with ellipsis
                while name_surf.get_width() > max_name_width and len(name) > 1:
                    name = name[:-1]
                    name_surf = self.app.font_small.render(name + "...", True, text_color)

            surface.blit(name_surf, (x, text_y))

    def draw_tooltip(self, surface, font, scale=1.0):
        """Draw tooltip for chevron if hovered for sufficient time"""
        if self.chevron_hovered:
            elapsed = time.time() - self.chevron_hover_start
            if elapsed >= 0.5:  # Show tooltip after 500ms
                # Determine tooltip text based on collapsed state
                tooltip_text = "Expand hierarchy" if self.collapsed else "Collapse hierarchy"

                # Create tooltip surface
                padding = int(6 * scale)
                margin = int(5 * scale)
                cursor_offset = int(16 * scale)  # Offset from cursor
                text_surf = font.render(tooltip_text, True, (255, 255, 255))
                tooltip_w = text_surf.get_width() + padding * 2
                tooltip_h = text_surf.get_height() + padding * 2

                # Position tooltip below and to the right of cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                tooltip_x = mouse_x + cursor_offset
                tooltip_y = mouse_y + cursor_offset

                # Keep tooltip on screen
                screen_w, screen_h = surface.get_size()
                if tooltip_x + tooltip_w > screen_w - margin:
                    tooltip_x = mouse_x - tooltip_w - margin
                if tooltip_y + tooltip_h > screen_h - margin:
                    tooltip_y = mouse_y - tooltip_h - margin

                # Draw tooltip background
                tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_w, tooltip_h)
                border_radius = int(4 * scale)
                pygame.draw.rect(surface, (50, 50, 55), tooltip_rect, border_radius=border_radius)
                pygame.draw.rect(surface, (100, 100, 110), tooltip_rect, 1, border_radius=border_radius)
                surface.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))

        # Draw search field tooltip
        if self.search_hovered and not self.collapsed:
            elapsed = time.time() - self.search_hover_start
            if elapsed >= 0.5:  # Show tooltip after 500ms
                tooltip_text = "Search shapes and groups by name"

                # Create tooltip surface
                padding = int(6 * scale)
                margin = int(5 * scale)
                cursor_offset = int(16 * scale)
                text_surf = font.render(tooltip_text, True, (255, 255, 255))
                tooltip_w = text_surf.get_width() + padding * 2
                tooltip_h = text_surf.get_height() + padding * 2

                # Position tooltip below and to the right of cursor
                mouse_x, mouse_y = pygame.mouse.get_pos()
                tooltip_x = mouse_x + cursor_offset
                tooltip_y = mouse_y + cursor_offset

                # Keep tooltip on screen
                screen_w, screen_h = surface.get_size()
                if tooltip_x + tooltip_w > screen_w - margin:
                    tooltip_x = mouse_x - tooltip_w - margin
                if tooltip_y + tooltip_h > screen_h - margin:
                    tooltip_y = mouse_y - tooltip_h - margin

                # Draw tooltip background
                tooltip_rect = pygame.Rect(tooltip_x, tooltip_y, tooltip_w, tooltip_h)
                border_radius = int(4 * scale)
                pygame.draw.rect(surface, (50, 50, 55), tooltip_rect, border_radius=border_radius)
                pygame.draw.rect(surface, (100, 100, 110), tooltip_rect, 1, border_radius=border_radius)
                surface.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))


class Hayai:
    # Modes
    MODE_MAKE_SHAPE = 0
    MODE_MOVE_SHAPE = 1
    MODE_EDIT_SHAPE = 2
    MODE_EDIT_WARP = 3

    def __init__(self):
        global WIDTH, HEIGHT
        self.fullscreen = False
        self.windowed_size = (BASE_WIDTH, BASE_HEIGHT)

        # Capture native display resolution BEFORE first set_mode() call
        desktop_sizes = pygame.display.get_desktop_sizes()
        self.native_resolution = desktop_sizes[0] if desktop_sizes else (1920, 1080)

        # OpenGL mode - use OPENGL | DOUBLEBUF flags
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
        self.screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT),
                                               pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        # Initialize OpenGL renderer
        gl_renderer.init_gl(BASE_WIDTH, BASE_HEIGHT)

        pygame.display.set_caption("Hayai : Rapid Projection Mapping")

        # Internal resolution
        WIDTH, HEIGHT = BASE_WIDTH, BASE_HEIGHT

        # Create a pygame surface for UI rendering (blitted on top of OpenGL)
        self.ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        self.clock = pygame.time.Clock()
        self.running = True
        self.play_mode = False
        self.show_mask = True
        self.show_geom = True
        self.show_crosshair = True  # Crosshair toggle (on by default)
        self.show_cursor = True  # System cursor visibility
        self.show_grid = True  # Background grid (on by default)
        self.show_controls = True  # Controls panel visibility (on by default)
        self.mouse_pos = (0, 0)  # Track mouse position for shape preview

        # Shape placement mode
        self.pending_shape_type = None  # Shape type name: "regular" or None for freeform
        self.placing_shape = False  # True when waiting for click to place shape
        self.regular_polygon_sides = 4  # Default number of sides (3-120)
        self.on_corner = False  # If True, polygon sits on vertex; if False, sits on edge

        # Hierarchical scene structure (replaces flat shapes list)
        self.scene_root: List[SceneItem] = []  # Top-level items (groups and shapes)
        self.selected_items: List[SceneItem] = []  # Multi-selection support
        self.shapes: List[Shape] = []  # Flat list for backward compat (auto-generated)
        self.selected_shape: Optional[Shape] = None  # Primary selection (first selected shape)
        self.current_contour: List[Tuple[float, float]] = []

        # Hierarchy panel state (panel itself manages collapsed/expanded state)
        self.hierarchy_panel_width = 250  # Current panel width
        self.hierarchy_scroll_offset = 0  # Scroll position
        self.hierarchy_search_text = ""  # Search filter
        self.hierarchy_drag_item: Optional[SceneItem] = None  # Item being dragged
        self.hierarchy_drop_target: Optional[Tuple[SceneItem, str]] = None  # (target, "into"/"before"/"after")
        self.group_counter = 0  # For generating unique group names

        self.mode = self.MODE_MOVE_SHAPE
        self.dragging = False
        self.drag_start = None
        self.dragging_warp_point = None
        self.dragging_vertex = None  # Index of vertex being dragged in edit shape mode
        self.drag_start_world = None  # World position at drag start (for delta-based dragging)
        self.drag_start_local = None  # Local position at drag start (for delta-based dragging)
        self.hover_vertex = None  # Index of vertex being hovered
        self.selected_vertex = None  # Index of persistently selected vertex
        self.hover_edge = None  # Tuple (index, point) of edge being hovered

        # Marching ants animation state
        self.march_offset = 0.0  # Current animation offset
        self.last_march_time = time.time()  # Time of last march update

        # Grasp handle scaling state
        self.active_handle: Optional[str] = None  # Handle ID ("tl", "tr", "br", "bl", "t", "r", "b", "l")
        self.handle_drag_start: Optional[Tuple[float, float]] = None  # Mouse position at drag start
        self.handle_bounds_start: Optional[Tuple[float, float, float, float]] = None  # Selection bounds at drag start
        self.handle_fixed_point: Optional[Tuple[float, float]] = None  # Point that stays fixed during scaling
        self.hover_handle: Optional[str] = None  # Handle currently being hovered

        # Right-click rotation state
        self.rotating = False  # True when right-click rotating
        self.rotate_start_angle = 0.0  # Initial angle from center to mouse
        self.rotate_start_center: Optional[Tuple[float, float]] = None  # Center of rotation
        self.rotation_pivot_for_ui: Optional[Tuple[float, float]] = None  # Fixed pivot point for UI rendering during rotation

        # Warp rotation state (Edit Warp mode)
        self.rotating_warp = False
        self.warp_rotate_start_angle = 0.0
        self.warp_rotate_center: Optional[Tuple[float, float]] = None

        # Marquee selection state
        self.marquee_start: Optional[Tuple[float, float]] = None  # Start point (base coords)
        self.marquee_end: Optional[Tuple[float, float]] = None  # Current end point (base coords)
        self.marquee_active = False  # True while dragging marquee

        self.clipboard: Optional[Shape] = None

        # Shape naming
        self.shape_counter = 0  # For generating unique shape names

        # Undo/redo manager
        self.undo_manager = UndoManager()

        # Toast notifications
        self.toasts: List[Toast] = []

        # Keyboard navigation
        self.focused_button_index = -1  # -1 means no button focused

        # Properties panel controls (initialized after fonts)
        self.properties_sliders = []
        self.name_input = None
        self.last_selected_shape = None  # Track selection changes
        self.last_selected_item = None  # Track selection changes for Groups

        # UI scaling
        self.ui_scale = 1.5  # UI scale factor (0.5 to 2.0)
        self.ui_scale_slider = None  # Bottom-right scale control

        # Fonts (scaled)
        self.recreate_fonts()

        # Create UI buttons
        self.create_ui()

        # Create properties panel
        self.create_properties_panel()

        # Create UI scale slider
        self.create_scale_slider()

        # Create hierarchy panel and context menu
        self.hierarchy_panel = HierarchyPanel(self)
        self.context_menu = ContextMenu()

        # Set initial button states based on current context
        self.update_mode_button_states()

    def show_toast(self, message, duration=3.0, toast_type="info"):
        """Show a toast notification"""
        self.toasts.append(Toast(message, duration, toast_type))

    def recreate_fonts(self):
        """Recreate fonts with current UI scale"""
        base_font_size = 24
        base_font_small_size = 20
        self.font = pygame.font.Font(None, int(base_font_size * self.ui_scale))
        self.font_small = pygame.font.Font(None, int(base_font_small_size * self.ui_scale))

    def create_scale_slider(self):
        """Create the UI scale slider in DISPLAY group (lower-left)"""
        slider_width = int(100 * self.ui_scale)  # Match button width
        slider_height = int(35 * self.ui_scale)
        # Use position calculated in create_ui()
        x = getattr(self, 'display_slider_x', 15)
        y = getattr(self, 'display_slider_y', HEIGHT - slider_height - 15)
        self.ui_scale_slider = Slider(
            x, y,
            slider_width, slider_height,
            0.1, 5.0, self.ui_scale,
            "UI Scale", self.on_ui_scale_change, "{:.1f}x",
            tooltip="Change the size of all UI elements"
        )

    def on_ui_scale_change(self, value):
        """Callback when UI scale slider is changed"""
        self.ui_scale = value
        self.recreate_ui_elements()

    def recreate_ui_elements(self):
        """Recreate all UI elements with current scale"""
        self.recreate_fonts()
        self.create_ui()
        self.create_properties_panel()
        self.create_scale_slider()  # Recreate at new position/size
        self.update_mode_button_states()  # Restore button states

    def generate_shape_name(self):
        """Generate a unique name for a new shape"""
        self.shape_counter += 1
        return f"Shape_{self.shape_counter:03d}"

    def generate_group_name(self):
        """Generate a unique name for a new group"""
        self.group_counter += 1
        return f"Group_{self.group_counter:03d}"

    # ========== Hierarchy Helper Methods ==========

    def rebuild_shapes_list(self):
        """Rebuild flat shapes list from scene_root hierarchy."""
        self.shapes = self.get_all_shapes_flat()

    def get_all_shapes_flat(self) -> List[Shape]:
        """Get all shapes in draw order (depth-first traversal, bottom of hierarchy = on top)."""
        shapes = []
        def collect(items: List[SceneItem]):
            for item in items:
                if isinstance(item, Shape):
                    shapes.append(item)
                elif isinstance(item, Group):
                    collect(item.children)
        collect(self.scene_root)
        return shapes

    def get_all_items_flat(self) -> List[SceneItem]:
        """Get all items (groups and shapes) in tree order."""
        items = []
        def collect(children: List[SceneItem]):
            for item in children:
                items.append(item)
                if isinstance(item, Group):
                    collect(item.children)
        collect(self.scene_root)
        return items

    def rebuild_parent_refs(self):
        """Rebuild _parent references after deserialization."""
        def set_parents(items: List[SceneItem], parent: Optional[Group]):
            for item in items:
                item._parent = parent
                if isinstance(item, Group):
                    set_parents(item.children, item)
        set_parents(self.scene_root, None)

    def get_top_level_parent(self, item: SceneItem) -> SceneItem:
        """Traverse _parent chain to get the root-level ancestor of an item."""
        current = item
        while current._parent is not None:
            current = current._parent
        return current

    def get_selection_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the combined bounding box of all selected items in world coordinates."""
        if not self.selected_items:
            return None

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for item in self.selected_items:
            if isinstance(item, Shape):
                bounds = item.get_world_bounds()
            elif isinstance(item, Group):
                bounds = item.get_bounds()
            else:
                continue

            if bounds == (0, 0, 0, 0):
                continue

            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        if min_x == float('inf'):
            return None

        return (min_x, min_y, max_x, max_y)

    def get_selection_center(self) -> Optional[Tuple[float, float]]:
        """Get the center of the combined selection bounds."""
        bounds = self.get_selection_bounds()
        if bounds is None:
            return None
        return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)

    def add_item_to_scene(self, item: SceneItem, parent: Optional[Group] = None):
        """Add an item to the scene (at root or under a parent)."""
        if parent is not None:
            parent.add_child(item)
        else:
            if item._parent is not None:
                item._parent.children.remove(item)
            item._parent = None
            self.scene_root.append(item)
        self.rebuild_shapes_list()

    def remove_item_from_scene(self, item: SceneItem):
        """Remove an item from its current location in the scene."""
        if item._parent is not None:
            item._parent.remove_child(item)
        elif item in self.scene_root:
            self.scene_root.remove(item)
        item._parent = None
        self.rebuild_shapes_list()

    def reparent_item(self, item: SceneItem, new_parent: Optional[Group],
                      index: Optional[int] = None, preserve_world_pos: bool = True):
        """
        Move an item to a new parent (or root if new_parent is None).
        Optionally preserves world position by adjusting local position.
        """
        # Prevent circular references: check if new_parent is a descendant of item
        if new_parent is not None:
            ancestor = new_parent
            while ancestor is not None:
                if ancestor is item:
                    # Would create a cycle - abort
                    self.show_toast("Cannot move item into its own descendant", 2.0, "error")
                    return
                ancestor = ancestor._parent

        old_parent = item._parent
        if preserve_world_pos and isinstance(item, (Shape, Group)):
            # Get current world position
            if isinstance(item, Shape):
                old_world = item.get_center()
            else:
                # For groups, preserve the world position of the group's origin (not its center)
                if old_parent is not None:
                    old_world = old_parent.transform_point(item.position[0], item.position[1])
                else:
                    old_world = item.position

        # Remove from old location
        if old_parent is not None:
            old_parent.children.remove(item)
        elif item in self.scene_root:
            self.scene_root.remove(item)

        # Add to new location
        if new_parent is not None:
            item._parent = new_parent
            if index is not None:
                new_parent.children.insert(index, item)
            else:
                new_parent.children.append(item)
        else:
            item._parent = None
            if index is not None:
                self.scene_root.insert(index, item)
            else:
                self.scene_root.append(item)

        # Adjust position to preserve world location
        if preserve_world_pos and isinstance(item, (Shape, Group)):
            if new_parent is not None:
                # Convert old world pos to new parent's local space
                new_local = new_parent.inverse_transform_point(old_world[0], old_world[1])
            else:
                new_local = old_world

            # Adjust position
            if isinstance(item, Shape):
                current_center = item._get_local_center()
                item.position = (
                    new_local[0] - current_center[0],
                    new_local[1] - current_center[1]
                )
            else:
                item.position = new_local

        self.rebuild_shapes_list()

    def group_selected_items(self):
        """Create a new group containing all selected items."""
        if len(self.selected_items) < 1:
            return None

        self.save_undo_state()

        # Find common parent (or None if items are at different levels)
        # Use id() for set since Group objects aren't hashable
        parent_ids = set(id(item._parent) for item in self.selected_items)
        if len(parent_ids) == 1:
            common_parent = self.selected_items[0]._parent
        else:
            common_parent = None

        # Create new group
        new_group = Group(name=self.generate_group_name())

        # Find insertion index (position of first selected item)
        if common_parent is not None:
            container = common_parent.children
        else:
            container = self.scene_root

        min_index = len(container)
        for item in self.selected_items:
            if item in container:
                min_index = min(min_index, container.index(item))

        # Remove items from their current locations
        for item in self.selected_items:
            if item._parent is not None:
                item._parent.children.remove(item)
                item._parent = None
            elif item in self.scene_root:
                self.scene_root.remove(item)

        # Sort items by their position in hierarchy (top-to-bottom order)
        # This preserves visual ordering regardless of selection order
        def get_hierarchy_index(item):
            """Get item's position in depth-first traversal of scene tree"""
            index = [0]
            def search(items):
                for i in items:
                    if i is item:
                        return True
                    index[0] += 1
                    if isinstance(i, Group):
                        if search(i.children):
                            return True
                return False
            search(self.scene_root)
            return index[0]

        sorted_items = sorted(self.selected_items, key=get_hierarchy_index)

        # Add items to new group (in hierarchy order)
        for item in sorted_items:
            new_group.add_child(item)

        # Insert group at common parent
        if common_parent is not None:
            new_group._parent = common_parent
            common_parent.children.insert(min_index, new_group)
        else:
            new_group._parent = None
            self.scene_root.insert(min_index, new_group)

        self.rebuild_shapes_list()

        # Select the new group
        self.selected_items = [new_group]
        self.selected_shape = None
        self.show_toast(f"Created {new_group.name}", 1.5, "success")
        return new_group

    def ungroup_selected(self):
        """
        Ungroup selected items:
        - If a Group is selected: dissolve it, moving children to parent level
        - If a Shape/Group inside a parent group is selected: remove it from parent
        Empty groups are automatically deleted.
        """
        if not self.selected_items:
            return

        self.save_undo_state()
        new_selection = []
        groups_to_check_empty = set()  # Track groups that might become empty
        extracted_from_groups = set()  # Track group names items were extracted from

        # Separate selected groups (to dissolve) from items inside groups (to extract)
        groups_to_dissolve = [item for item in self.selected_items if isinstance(item, Group)]
        items_to_extract = [item for item in self.selected_items
                           if item._parent is not None and item not in groups_to_dissolve]

        # Extract individual items from their parent groups
        for item in items_to_extract:
            extracted_from_groups.add(item._parent.name)
            parent_group = item._parent
            grandparent = parent_group._parent

            if grandparent is not None:
                container = grandparent.children
            else:
                container = self.scene_root

            # Capture world position before changing parent
            if isinstance(item, Shape):
                item_world_pos = item.get_center()
            else:  # Group
                item_world_pos = parent_group.transform_point(item.position[0], item.position[1])

            # Find insertion index (after parent group)
            idx = container.index(parent_group) + 1

            # Remove from parent group
            parent_group.children.remove(item)
            item._parent = grandparent
            container.insert(idx, item)

            # Convert world position to new parent's local space
            if grandparent is not None:
                new_local = grandparent.inverse_transform_point(item_world_pos[0], item_world_pos[1])
            else:
                new_local = item_world_pos

            # Adjust position to preserve world location
            if isinstance(item, Shape):
                local_center = item._get_local_center()
                item.position = (new_local[0] - local_center[0], new_local[1] - local_center[1])
            else:  # Group
                item.position = new_local

            # Bake the parent group's rotation and scale into the item
            item.rotation = (item.rotation + parent_group.rotation) % 360
            item.scale = item.scale * parent_group.scale

            new_selection.append(item)
            groups_to_check_empty.add(id(parent_group))

        # Dissolve selected groups
        for group in groups_to_dissolve:
            parent = group._parent
            if parent is not None:
                container = parent.children
            else:
                container = self.scene_root

            # Find group's index - skip if group was already moved/removed
            if group not in container:
                # Group may have been moved when dissolving a parent group earlier
                continue
            idx = container.index(group)

            # Get the group's transform
            group_rotation = group.rotation
            group_scale = group.scale

            # Move children to parent level, preserving world transforms
            children = list(group.children)
            for i, child in enumerate(children):
                if isinstance(child, Shape):
                    child_world_pos = child.get_center()
                else:
                    child_world_pos = group.transform_point(child.position[0], child.position[1])

                group.children.remove(child)
                child._parent = parent
                container.insert(idx + i, child)

                if parent is not None:
                    new_local = parent.inverse_transform_point(child_world_pos[0], child_world_pos[1])
                else:
                    new_local = child_world_pos

                if isinstance(child, Shape):
                    local_center = child._get_local_center()
                    child.position = (new_local[0] - local_center[0], new_local[1] - local_center[1])
                else:
                    child.position = new_local

                child.rotation = (child.rotation + group_rotation) % 360
                child.scale = child.scale * group_scale

                new_selection.append(child)

            # Remove the dissolved group
            container.remove(group)

        # Clean up empty groups (check groups that had items extracted)
        self._remove_empty_groups()

        self.rebuild_shapes_list()
        self.selected_items = new_selection
        self.selected_shape = next((s for s in new_selection if isinstance(s, Shape)), None)

        if items_to_extract and not groups_to_dissolve:
            group_names = ", ".join(sorted(extracted_from_groups))
            self.show_toast(f"Removed from {group_names}", 1.5, "success")
        elif groups_to_dissolve:
            group_names = ", ".join(g.name for g in groups_to_dissolve)
            self.show_toast(f"Dissolved {group_names}", 1.5, "success")

    def _remove_empty_groups(self):
        """Recursively remove any empty groups from the scene."""
        def remove_empty_recursive(container, parent=None):
            items_to_remove = []
            for item in container:
                if isinstance(item, Group):
                    # First recurse into children
                    remove_empty_recursive(item.children, item)
                    # Then check if this group is now empty
                    if len(item.children) == 0:
                        items_to_remove.append(item)
            for item in items_to_remove:
                container.remove(item)
                item._parent = None

        remove_empty_recursive(self.scene_root)

    def update_selection_from_items(self):
        """Update selected_shape based on selected_items."""
        shapes_in_selection = [s for s in self.selected_items if isinstance(s, Shape)]
        self.selected_shape = shapes_in_selection[0] if shapes_in_selection else None

    def select_item(self, item: SceneItem, add_to_selection: bool = False,
                    range_select: bool = False, force_individual: bool = False):
        """
        Select an item, optionally adding to current selection.

        Args:
            item: The item to select
            add_to_selection: If True, toggle item in selection (Shift-click behavior)
            range_select: If True, select all items between last selected and this one
            force_individual: If False (default), select top-level parent group.
                             If True (Ctrl-click), select just this specific item.
        """
        # Determine what to actually select
        if force_individual:
            # Ctrl-click: select just the clicked item
            target_item = item
        else:
            # Normal click: select the top-level parent (whole group)
            target_item = self.get_top_level_parent(item)

        if range_select and self.selected_items:
            # Range selection: select all items between last selected and this one
            all_items = self.get_all_items_flat()
            if self.selected_items[-1] in all_items and target_item in all_items:
                start_idx = all_items.index(self.selected_items[-1])
                end_idx = all_items.index(target_item)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                for i in range(start_idx, end_idx + 1):
                    if all_items[i] not in self.selected_items:
                        self.selected_items.append(all_items[i])
        elif add_to_selection:
            # Toggle selection
            if target_item in self.selected_items:
                self.selected_items.remove(target_item)
            else:
                self.selected_items.append(target_item)
        else:
            # Single selection
            self.selected_items = [target_item]

        self.update_selection_from_items()
        self.update_mode_button_states()

    def clear_selection(self):
        """Clear all selection."""
        self.selected_items = []
        self.selected_shape = None
        self.update_mode_button_states()

    def complete_marquee_selection(self, force_individual: bool = False):
        """
        Select all shapes that intersect with the marquee rectangle.
        Uses partial intersection (shapes partially or fully inside are selected).

        Args:
            force_individual: If False (default), select top-level parent groups.
                             If True (Ctrl+drag), select individual shapes.
        """
        if not self.marquee_start or not self.marquee_end:
            return

        # Calculate marquee bounds (normalize to min/max)
        x1, y1 = self.marquee_start
        x2, y2 = self.marquee_end
        marquee_min_x = min(x1, x2)
        marquee_max_x = max(x1, x2)
        marquee_min_y = min(y1, y2)
        marquee_max_y = max(y1, y2)

        # Find shapes that intersect with marquee rectangle
        for shape in self.shapes:
            bounds = shape.get_world_bounds()
            if bounds:
                shape_min_x, shape_min_y, shape_max_x, shape_max_y = bounds

                # Check for rectangle intersection (partial overlap)
                intersects = (
                    marquee_min_x <= shape_max_x and
                    marquee_max_x >= shape_min_x and
                    marquee_min_y <= shape_max_y and
                    marquee_max_y >= shape_min_y
                )

                if intersects:
                    # Determine what to select: individual shape or parent group
                    if force_individual:
                        target_item = shape
                    else:
                        target_item = self.get_top_level_parent(shape)

                    if target_item not in self.selected_items:
                        self.selected_items.append(target_item)

        self.update_selection_from_items()
        self.update_mode_button_states()

    def move_selection(self, dx: float, dy: float):
        """Move all selected items by the given delta."""
        for item in self.selected_items:
            item.move(dx, dy)

    def rotate_selection(self, degrees: float, center: Optional[Tuple[float, float]] = None):
        """
        Rotate all selected items by the given degrees around a center point.
        If center is None, uses the selection center.
        """
        if center is None:
            center = self.get_selection_center()
            if center is None:
                return

        for item in self.selected_items:
            if isinstance(item, Shape):
                # Update shape's rotation
                item.rotation = (item.rotation + degrees) % 360
                # Rotate position around the center
                px, py = item.position
                cx, cy = center
                angle = math.radians(degrees)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                # Get the shape's world center for proper rotation
                shape_center = item.get_center()
                # Rotate shape center around selection center
                dx = shape_center[0] - cx
                dy = shape_center[1] - cy
                new_cx = cx + dx * cos_a - dy * sin_a
                new_cy = cy + dx * sin_a + dy * cos_a
                # Adjust position to move shape center to new location
                # The shape center in world coords is local_center transformed
                local_center = item._get_local_center()
                # Calculate what position gives us the new center
                # new_world_center = local_center * scale rotated by rotation + position
                # We need to adjust position so that get_center() returns (new_cx, new_cy)
                offset_x = new_cx - shape_center[0]
                offset_y = new_cy - shape_center[1]
                item.position = (px + offset_x, py + offset_y)
            elif isinstance(item, Group):
                item.rotate(degrees, center)

    def scale_selection(self, scale_x: float, scale_y: float, fixed_point: Tuple[float, float]):
        """
        Scale all selected items around a fixed point.
        scale_x and scale_y are multipliers (1.0 = no change).
        """
        for item in self.selected_items:
            if isinstance(item, Shape):
                item.scale_geometry(scale_x, scale_y, fixed_point)
            elif isinstance(item, Group):
                # For groups, recursively scale all shapes
                for shape in item.get_all_shapes():
                    shape.scale_geometry(scale_x, scale_y, fixed_point)

    def delete_selected_items(self):
        """Delete all selected items."""
        if not self.selected_items:
            return

        self.save_undo_state()

        for item in list(self.selected_items):
            self.remove_item_from_scene(item)

        self.clear_selection()
        self.show_toast("Items deleted", 1.5, "info")

    def save_undo_state(self):
        """Save current state for undo"""
        self.undo_manager.save_state(self.scene_root)

    def undo(self):
        """Undo last action"""
        restored = self.undo_manager.undo(self.scene_root)
        if restored is not None:
            self.scene_root = restored
            self.rebuild_parent_refs()
            self.rebuild_shapes_list()
            self.clear_selection()
            self.selected_vertex = None
            self.show_toast("Undo", 1.5, "info")
        else:
            self.show_toast("Nothing to undo", 1.5, "info")

    def redo(self):
        """Redo last undone action"""
        restored = self.undo_manager.redo(self.scene_root)
        if restored is not None:
            self.scene_root = restored
            self.rebuild_parent_refs()
            self.rebuild_shapes_list()
            self.clear_selection()
            self.selected_vertex = None
            self.show_toast("Redo", 1.5, "info")
        else:
            self.show_toast("Nothing to redo", 1.5, "info")

    def focus_next_button(self):
        """Move focus to the next button"""
        if not self.buttons:
            return
        # Clear current focus
        if self.focused_button_index >= 0:
            self.buttons[self.focused_button_index].focused = False
        # Move to next button
        self.focused_button_index = (self.focused_button_index + 1) % len(self.buttons)
        self.buttons[self.focused_button_index].focused = True

    def focus_prev_button(self):
        """Move focus to the previous button"""
        if not self.buttons:
            return
        # Clear current focus
        if self.focused_button_index >= 0:
            self.buttons[self.focused_button_index].focused = False
        # Move to previous button
        self.focused_button_index = (self.focused_button_index - 1) % len(self.buttons)
        self.buttons[self.focused_button_index].focused = True

    def activate_focused_button(self):
        """Activate the currently focused button"""
        if 0 <= self.focused_button_index < len(self.buttons):
            button = self.buttons[self.focused_button_index]
            if button.callback:
                button.callback()
                return True
        return False

    def clear_button_focus(self):
        """Clear focus from all buttons"""
        if self.focused_button_index >= 0:
            self.buttons[self.focused_button_index].focused = False
        self.focused_button_index = -1

    def get_scale(self):
        """Get scale factors for current window size vs base size"""
        return WIDTH / BASE_WIDTH, HEIGHT / BASE_HEIGHT

    def scale_point(self, x, y):
        """Scale a point from base coordinates to current screen coordinates"""
        sx, sy = self.get_scale()
        return x * sx, y * sy

    def unscale_point(self, x, y):
        """Convert screen coordinates back to base coordinates"""
        sx, sy = self.get_scale()
        return x / sx, y / sy

    def create_ui(self):
        # Scale all dimensions by ui_scale
        btn_w = int(100 * self.ui_scale)
        btn_h = int(30 * self.ui_scale)
        gap = int(5 * self.ui_scale)
        panel_header_height = int(28 * self.ui_scale)  # Consistent header height for all panels
        panel_padding = int(10 * self.ui_scale)

        # Panel margin constants (used consistently across all panels except Hierarchy)
        self.panel_margin = int(10 * self.ui_scale)  # Horizontal margin from screen edges
        self.panel_margin_v = int(10 * self.ui_scale)  # Vertical margin from screen edges
        self.section_gap = int(8 * self.ui_scale)  # Extra spacing between sections
        margin = self.panel_margin  # Local alias for backward compatibility

        self.buttons = []

        # ===== TOOLS PANEL (CENTER TOP) =====
        tools_panel_width = int(385 * self.ui_scale)
        tools_panel_x = (WIDTH - tools_panel_width) // 2
        tools_panel_y = self.panel_margin_v
        subtitle_height = int(18 * self.ui_scale)  # Height for section subtitles

        # Row 1: Shape creation buttons (below panel header + "Create" subtitle)
        create_btn_w = int(80 * self.ui_scale)
        corner_btn_w = int(90 * self.ui_scale)
        slider_w = int(100 * self.ui_scale)

        create_total_width = create_btn_w * 2 + slider_w + corner_btn_w + gap * 3
        create_x = (WIDTH - create_total_width) // 2
        # Store subtitle Y position for drawing
        self.tools_create_subtitle_y = tools_panel_y + panel_header_height
        create_y = self.tools_create_subtitle_y + subtitle_height

        self.btn_create_freeform = Button(create_x, create_y, create_btn_w, btn_h, "Freeform",
                                          self.start_poly_mode,
                                          tooltip="Draw custom polygon")
        self.buttons.append(self.btn_create_freeform)

        create_x += create_btn_w + gap
        self.btn_create_regular = Button(create_x, create_y, create_btn_w, btn_h, "Regular",
                                         self.start_regular_placement,
                                         tooltip="Place regular polygon")
        self.buttons.append(self.btn_create_regular)

        create_x += create_btn_w + gap
        entry_h = int(24 * self.ui_scale)
        entry_y = create_y + (btn_h - entry_h) // 2  # Vertically center with buttons
        self.sides_entry = NumericEntry(create_x, entry_y, slider_w, entry_h,
                                        3, 120, self.regular_polygon_sides,
                                        "Sides", self.on_sides_change,
                                        tooltip="Number of regular polygon sides")

        create_x += slider_w + gap
        self.btn_on_corner = Button(create_x, create_y, corner_btn_w, btn_h,
                                    "Corner: ON" if self.on_corner else "Corner: OFF",
                                    self.toggle_on_corner,
                                    tooltip="Toggle vertex vs edge at bottom")
        self.buttons.append(self.btn_on_corner)

        # Sides entry and corner button disabled until Regular mode is selected
        self.sides_entry.disabled = True
        self.btn_on_corner.disabled = True

        # ===== CENTER TOP ROW 2: MODE buttons =====
        # Mode buttons: Edit Shape, Edit Warp, Move Shape (centered below creation row + "Edit" subtitle)
        mode_btn_count = 3
        mode_total_width = mode_btn_count * btn_w + (mode_btn_count - 1) * gap
        mode_x = (WIDTH - mode_total_width) // 2
        # Store subtitle Y position for drawing (with section_gap before it)
        self.tools_edit_subtitle_y = create_y + btn_h + gap + self.section_gap
        mode_y = self.tools_edit_subtitle_y + subtitle_height

        # Store Edit Shape position for sub-buttons
        edit_shape_x = mode_x
        self.btn_edit_shape = Button(mode_x, mode_y, btn_w, btn_h, "Edit Shape",
                                     lambda: self.set_mode(self.MODE_EDIT_SHAPE),
                                     tooltip="Modify polygon vertices")
        self.buttons.append(self.btn_edit_shape)

        mode_x += btn_w + gap
        # Store Edit Warp position for sub-buttons
        edit_warp_x = mode_x
        self.btn_edit_warp = Button(mode_x, mode_y, btn_w, btn_h, "Edit Warp",
                                    lambda: self.set_mode(self.MODE_EDIT_WARP),
                                    tooltip="Adjust perspective corners")
        self.buttons.append(self.btn_edit_warp)

        mode_x += btn_w + gap
        # Store Move Shape position for sub-buttons
        move_shape_x = mode_x
        self.btn_move_shape = Button(mode_x, mode_y, btn_w, btn_h, "Move Shape",
                                     lambda: self.set_mode(self.MODE_MOVE_SHAPE),
                                     tooltip="Move and rotate shapes")
        self.buttons.append(self.btn_move_shape)

        # ===== SUB-BUTTONS UNDER MODE BUTTONS =====
        sub_btn_y = mode_y + btn_h + gap
        half_w = (btn_w - gap) // 2

        # fx/fy under Move Shape
        self.btn_flip_x = Button(move_shape_x, sub_btn_y, half_w, btn_h, "fx", self.flip_x,
                                 tooltip="Flip Horizontal")
        self.btn_flip_x.disabled = True  # Initially disabled (not in Move Shape mode)
        self.buttons.append(self.btn_flip_x)
        self.btn_flip_y = Button(move_shape_x + half_w + gap, sub_btn_y, half_w, btn_h, "fy", self.flip_y,
                                 tooltip="Flip Vertical")
        self.btn_flip_y.disabled = True
        self.buttons.append(self.btn_flip_y)

        # Set Image / Fit Warp under Edit Warp
        self.btn_set_image = Button(edit_warp_x, sub_btn_y, half_w, btn_h, "Image", self.set_image,
                                    tooltip="Load image or GIF for shape")
        self.btn_set_image.disabled = True  # Initially disabled (not in Edit Warp mode)
        self.buttons.append(self.btn_set_image)
        self.btn_fit_warp = Button(edit_warp_x + half_w + gap, sub_btn_y, half_w, btn_h, "Fit", self.fit_warp,
                                   tooltip="Reset warp to shape bounds")
        self.btn_fit_warp.disabled = True
        self.buttons.append(self.btn_fit_warp)

        # ===== SCENE PANEL (LEFT SIDE, TOP) =====
        scene_panel_x = margin + int(HierarchyPanel.COLLAPSED_WIDTH * self.ui_scale)
        scene_panel_y = self.panel_margin_v
        scene_btn_x = scene_panel_x + panel_padding
        scene_btn_y = scene_panel_y + panel_header_height

        self.btn_save = Button(scene_btn_x, scene_btn_y, btn_w, btn_h, "Save Scene", self.save_scene,
                               tooltip="Save project to file")
        self.buttons.append(self.btn_save)

        scene_btn_y += btn_h + gap
        self.btn_load = Button(scene_btn_x, scene_btn_y, btn_w, btn_h, "Load Scene", self.load_scene,
                               tooltip="Open saved project")
        self.buttons.append(self.btn_load)

        scene_btn_y += btn_h + gap
        self.btn_new = Button(scene_btn_x, scene_btn_y, btn_w, btn_h, "New Scene", self.new_scene,
                              tooltip="Clear and start fresh")
        self.buttons.append(self.btn_new)

        # ===== DISPLAY PANEL (LEFT SIDE, BOTTOM) =====
        display_panel_width = int(120 * self.ui_scale)
        display_panel_height = int(285 * self.ui_scale)  # 6 buttons + slider + padding
        display_panel_x = margin + int(HierarchyPanel.COLLAPSED_WIDTH * self.ui_scale)
        display_panel_y = HEIGHT - self.panel_margin_v - display_panel_height
        display_btn_x = display_panel_x + panel_padding
        display_btn_y = display_panel_y + panel_header_height

        # Geometry toggle
        self.btn_geometry_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                       "Geom: ON" if self.show_geom else "Geom: OFF",
                                       self.toggle_geometry, tooltip="Toggle geometry display")
        self.btn_geometry_toggle.active = not self.show_geom
        self.buttons.append(self.btn_geometry_toggle)

        display_btn_y += btn_h + gap
        # Mask toggle
        self.btn_mask_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                      "Mask: ON" if self.show_mask else "Mask: OFF",
                                      self.toggle_mask, tooltip="Toggle shape masking")
        self.btn_mask_toggle.active = not self.show_mask
        self.buttons.append(self.btn_mask_toggle)

        display_btn_y += btn_h + gap
        # Cursor toggle
        self.btn_cursor_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                        "Cursor: ON" if self.show_cursor else "Cursor: OFF",
                                        self.toggle_cursor, tooltip="Toggle system cursor visibility")
        self.btn_cursor_toggle.active = not self.show_cursor
        self.buttons.append(self.btn_cursor_toggle)

        display_btn_y += btn_h + gap
        # Crosshair toggle
        self.btn_crosshair_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                           "Cross: ON" if self.show_crosshair else "Cross: OFF",
                                           self.toggle_crosshair, tooltip="Toggle cursor crosshair")
        self.btn_crosshair_toggle.active = not self.show_crosshair
        self.buttons.append(self.btn_crosshair_toggle)

        display_btn_y += btn_h + gap
        # Grid toggle
        self.btn_grid_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                      "Grid: ON" if self.show_grid else "Grid: OFF",
                                      self.toggle_grid, tooltip="Toggle background grid")
        self.btn_grid_toggle.active = not self.show_grid
        self.buttons.append(self.btn_grid_toggle)

        display_btn_y += btn_h + gap
        # Controls toggle
        self.btn_controls_toggle = Button(display_btn_x, display_btn_y, btn_w, btn_h,
                                          "Controls: ON" if self.show_controls else "Controls: OFF",
                                          self.toggle_controls, tooltip="Toggle controls panel")
        self.btn_controls_toggle.active = not self.show_controls
        self.buttons.append(self.btn_controls_toggle)

        # UI Scale slider position (will be created in create_scale_slider)
        self.display_slider_y = display_btn_y + btn_h + int(12 * self.ui_scale)
        self.display_slider_x = display_btn_x

    def create_properties_panel(self):
        """Create properties panel controls on right side"""
        # Scale dimensions
        panel_width = int(220 * self.ui_scale)
        panel_x = WIDTH - panel_width - self.panel_margin
        panel_y = self.panel_margin_v
        slider_width = int(180 * self.ui_scale)
        slider_height = int(35 * self.ui_scale)
        input_height = int(24 * self.ui_scale)
        padding = int(10 * self.ui_scale)
        row_spacing = int(38 * self.ui_scale)  # Consistent spacing between rows

        # Name input (with label above, so position accounts for label space)
        # Panel header is 28px, then label needs 18px above input
        y_offset = panel_y + int(48 * self.ui_scale)
        self.name_input = TextInput(panel_x + padding, y_offset,
                                    slider_width, input_height,
                                    label="Name", on_change=self.on_name_change)

        # Image section position (drawn in draw_properties_panel)
        y_offset += row_spacing
        self.image_label_y = y_offset
        self.image_field_y = y_offset + int(18 * self.ui_scale)

        # Sliders with consistent spacing (section_gap before new section)
        y_offset += row_spacing + self.section_gap
        self.alpha_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                   0.0, 1.0, 1.0, "Alpha", self.on_alpha_slider_change)

        y_offset += row_spacing
        self.hue_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                 0.0, 360.0, 0.0, "Hue", self.on_hue_change, "{:.0f}")

        y_offset += row_spacing
        self.sat_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                 0.0, 2.0, 1.0, "Saturation", self.on_sat_change)

        y_offset += row_spacing
        self.val_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                 0.0, 2.0, 1.0, "Brightness", self.on_val_change)

        y_offset += row_spacing
        self.speed_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                   0.01, 10.0, 1.0, "Anim Speed", self.on_speed_change)

        y_offset += row_spacing
        self.persp_x_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                     -1.0, 1.0, 0.0, "Perspective X", self.on_persp_x_change)

        y_offset += row_spacing
        self.persp_y_slider = Slider(panel_x + padding, y_offset, slider_width, slider_height,
                                     -1.0, 1.0, 0.0, "Perspective Y", self.on_persp_y_change)

        self.properties_sliders = [self.alpha_slider, self.hue_slider,
                                   self.sat_slider, self.val_slider, self.speed_slider,
                                   self.persp_x_slider, self.persp_y_slider]

        # Store panel dimensions for drawing
        self.properties_panel_x = panel_x
        self.properties_panel_y = panel_y
        self.properties_panel_width = panel_width
        self.properties_panel_padding = padding

    def on_name_change(self, new_name):
        """Callback when shape or group name is changed"""
        if self.selected_shape:
            self.selected_shape.name = new_name
        elif len(self.selected_items) == 1 and isinstance(self.selected_items[0], Group):
            self.selected_items[0].name = new_name

    def on_alpha_slider_change(self, value):
        """Callback when alpha slider is changed"""
        if self.selected_shape:
            self.selected_shape.alpha = value

    def on_hue_change(self, value):
        """Callback when hue slider is changed"""
        if self.selected_shape:
            self.selected_shape.hue_shift = value

    def on_sat_change(self, value):
        """Callback when saturation slider is changed"""
        if self.selected_shape:
            self.selected_shape.saturation = value

    def on_val_change(self, value):
        """Callback when value/brightness slider is changed"""
        if self.selected_shape:
            self.selected_shape.color_value = value

    def on_speed_change(self, value):
        """Callback when playback speed slider is changed"""
        if self.selected_shape and self.selected_shape.animation_frames:
            self.selected_shape.playback_speed = value

    def on_persp_x_change(self, value):
        """Callback when perspective X slider is changed"""
        if self.selected_shape:
            self.selected_shape.perspective_x = value

    def on_persp_y_change(self, value):
        """Callback when perspective Y slider is changed"""
        if self.selected_shape:
            self.selected_shape.perspective_y = value

    def update_properties_panel(self):
        """Sync properties panel with selected shape or group"""
        # Determine the selected item for properties panel
        selected_item = None
        if self.selected_shape:
            selected_item = self.selected_shape
        elif len(self.selected_items) == 1 and isinstance(self.selected_items[0], Group):
            selected_item = self.selected_items[0]

        if selected_item and selected_item != self.last_selected_item:
            # Update name input with appropriate label
            if isinstance(selected_item, Shape):
                self.name_input.label = "Shape Name"
                self.name_input.text = selected_item.name
                self.name_input.cursor_pos = len(self.name_input.text)
                # Update sliders for Shape
                self.alpha_slider.value = selected_item.alpha
                self.hue_slider.value = selected_item.hue_shift
                self.sat_slider.value = selected_item.saturation
                self.val_slider.value = selected_item.color_value
                self.speed_slider.value = selected_item.playback_speed
                # Disable speed slider for non-animated images
                has_animation = len(selected_item.animation_frames) > 1
                self.speed_slider.disabled = not has_animation
                # Sync perspective sliders
                self.persp_x_slider.value = selected_item.perspective_x
                self.persp_y_slider.value = selected_item.perspective_y
                self.last_selected_shape = selected_item
            else:  # Group
                self.name_input.label = "Group Name"
                self.name_input.text = selected_item.name
                self.name_input.cursor_pos = len(self.name_input.text)
                self.last_selected_shape = None

            self.last_selected_item = selected_item

    def set_mode(self, mode):
        self.mode = mode
        self.current_contour = []
        self.hover_vertex = None
        self.hover_edge = None
        self.dragging_vertex = None
        self.selected_vertex = None
        self.placing_shape = False
        self.pending_shape_type = None
        # Update mode button states
        self.btn_move_shape.active = (mode == self.MODE_MOVE_SHAPE)
        self.btn_edit_shape.active = (mode == self.MODE_EDIT_SHAPE)
        self.btn_edit_warp.active = (mode == self.MODE_EDIT_WARP)
        # Clear create buttons when switching to non-make modes
        self.btn_create_freeform.active = False
        self.btn_create_regular.active = False
        # Disable sides/corner controls when not in regular mode
        self.sides_entry.disabled = True
        self.btn_on_corner.disabled = True
        # Update sub-button disabled states based on mode
        self.btn_flip_x.disabled = (mode != self.MODE_MOVE_SHAPE)
        self.btn_flip_y.disabled = (mode != self.MODE_MOVE_SHAPE)
        # Update contextual button states (includes Image/Fit buttons)
        self.update_mode_button_states()

    def update_mode_button_states(self):
        """Update mode button disabled states based on current context."""
        # Check if shapes exist
        has_shapes = len(self.shapes) > 0

        # Check if freeform creation is in progress (unclosed)
        freeform_in_progress = (self.mode == self.MODE_MAKE_SHAPE and
                                len(self.current_contour) > 0)

        # Move Shape: disabled if no shapes OR freeform in progress
        self.btn_move_shape.disabled = not has_shapes or freeform_in_progress

        # Check if a closed shape is selected
        has_selected_shape = self.selected_shape is not None

        # Edit Shape/Warp: disabled if no shape selected
        self.btn_edit_shape.disabled = not has_selected_shape
        self.btn_edit_warp.disabled = not has_selected_shape

        # Image/Fit buttons: enabled only in Edit Warp mode with a shape selected
        edit_warp_active = (self.mode == self.MODE_EDIT_WARP and has_selected_shape)
        self.btn_set_image.disabled = not edit_warp_active
        self.btn_fit_warp.disabled = not edit_warp_active

    def toggle_geometry(self):
        self.show_geom = not self.show_geom
        self.btn_geometry_toggle.text = "Geom: ON" if self.show_geom else "Geom: OFF"
        self.btn_geometry_toggle.active = not self.show_geom

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.btn_mask_toggle.text = "Mask: ON" if self.show_mask else "Mask: OFF"
        self.btn_mask_toggle.active = not self.show_mask

    def toggle_crosshair(self):
        self.show_crosshair = not self.show_crosshair
        self.btn_crosshair_toggle.text = "Cross: ON" if self.show_crosshair else "Cross: OFF"
        self.btn_crosshair_toggle.active = not self.show_crosshair

    def toggle_cursor(self):
        self.show_cursor = not self.show_cursor
        self.btn_cursor_toggle.text = "Cursor: ON" if self.show_cursor else "Cursor: OFF"
        self.btn_cursor_toggle.active = not self.show_cursor
        pygame.mouse.set_visible(not self.play_mode and self.show_cursor)

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        self.btn_grid_toggle.text = "Grid: ON" if self.show_grid else "Grid: OFF"
        self.btn_grid_toggle.active = not self.show_grid

    def toggle_controls(self):
        self.show_controls = not self.show_controls
        self.btn_controls_toggle.text = "Controls: ON" if self.show_controls else "Controls: OFF"
        self.btn_controls_toggle.active = not self.show_controls

    def toggle_play_mode(self):
        self.play_mode = not self.play_mode
        # In Play mode, hide cursor; in Edit mode, restore cursor setting
        pygame.mouse.set_visible(not self.play_mode and self.show_cursor)

    def create_regular_polygon(self, cx, cy, sides, radius=100, on_corner=False):
        """Create regular polygon vertices centered at (cx, cy)

        Args:
            on_corner: If True, a vertex points down. If False, edge is at bottom.
        """
        points = []
        if on_corner:
            start_angle = math.pi / 2  # vertex at bottom
        else:
            start_angle = math.pi / 2 - math.pi / sides  # edge at bottom

        for i in range(sides):
            angle = start_angle + (2 * math.pi * i / sides)
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        return points

    def create_axis_aligned_rect(self, cx, cy, size=160):
        """Create axis-aligned rect centered at (cx, cy)"""
        half = size / 2
        return [(cx - half, cy - half), (cx + half, cy - half),
                (cx + half, cy + half), (cx - half, cy + half)]

    def start_regular_placement(self):
        """Start regular polygon placement mode using current sides setting"""
        self.pending_shape_type = "regular"
        self.placing_shape = True
        self.mode = self.MODE_MAKE_SHAPE
        self.current_contour = []  # Clear any partial freeform polygon
        # Update button states
        self.btn_move_shape.active = False
        self.btn_edit_shape.active = False
        self.btn_edit_warp.active = False
        self.btn_create_freeform.active = False
        self.btn_create_regular.active = True
        # Enable sides/corner controls in regular mode
        self.sides_entry.disabled = False
        self.btn_on_corner.disabled = False
        # Update contextual button states
        self.update_mode_button_states()

    def start_poly_mode(self):
        """Start free polygon drawing mode"""
        self.pending_shape_type = None
        self.placing_shape = False
        self.mode = self.MODE_MAKE_SHAPE
        self.current_contour = []
        # Update button states
        self.btn_move_shape.active = False
        self.btn_edit_shape.active = False
        self.btn_edit_warp.active = False
        self.btn_create_freeform.active = True
        self.btn_create_regular.active = False
        # Disable sides/corner controls in freeform mode
        self.sides_entry.disabled = True
        self.btn_on_corner.disabled = True
        # Update contextual button states
        self.update_mode_button_states()

    def on_sides_change(self, value):
        """Callback when sides entry changes"""
        self.regular_polygon_sides = int(value)

    def toggle_on_corner(self):
        """Toggle on-corner mode for regular polygons"""
        self.on_corner = not self.on_corner
        self.btn_on_corner.text = "Corner: ON" if self.on_corner else "Corner: OFF"
        self.btn_on_corner.active = self.on_corner

    def toggle_fullscreen(self):
        global WIDTH, HEIGHT

        self.fullscreen = not self.fullscreen

        # Invalidate OpenGL resources as context may be recreated
        gl_renderer.invalidate()

        # Clear all shape textures as they'll be invalid after context switch
        for shape in self.shapes:
            shape.gl_texture_id = 0
            shape.gl_texture_dirty = True

        # Use pygame's built-in fullscreen toggle
        pygame.display.toggle_fullscreen()
        self.screen = pygame.display.get_surface()

        # Get actual window size (framebuffer size) for OpenGL viewport
        # This differs from surface.get_size() which returns logical size
        WIDTH, HEIGHT = pygame.display.get_window_size()

        gl_renderer.init_gl(WIDTH, HEIGHT)

        # Recreate UI surface
        self.ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        # Recreate UI elements at new positions
        self.create_ui()
        self.create_properties_panel()
        self.create_scale_slider()

        # Restore button states based on current mode
        if self.mode == self.MODE_MAKE_SHAPE:
            if self.placing_shape and self.pending_shape_type == "regular":
                # Regular polygon placement
                self.btn_create_freeform.active = False
                self.btn_create_regular.active = True
            else:
                # Freeform drawing mode
                self.btn_create_freeform.active = True
                self.btn_create_regular.active = False
        else:
            # Non-MAKE_SHAPE modes - deactivate create buttons
            self.btn_create_freeform.active = False
            self.btn_create_regular.active = False
            # Set the correct mode button active
            self.btn_move_shape.active = (self.mode == self.MODE_MOVE_SHAPE)
            self.btn_edit_shape.active = (self.mode == self.MODE_EDIT_SHAPE)
            self.btn_edit_warp.active = (self.mode == self.MODE_EDIT_WARP)

        # Update sub-button disabled states (only enabled in Move Shape mode)
        self.btn_flip_x.disabled = (self.mode != self.MODE_MOVE_SHAPE)
        self.btn_flip_y.disabled = (self.mode != self.MODE_MOVE_SHAPE)

        # Restore on_corner button state
        self.btn_on_corner.text = "Corner: ON" if self.on_corner else "Corner: OFF"
        self.btn_on_corner.active = self.on_corner

        # Restore display toggle button states
        self.btn_geometry_toggle.text = "Geometry: ON" if self.show_geom else "Geometry: OFF"
        self.btn_geometry_toggle.active = not self.show_geom
        self.btn_mask_toggle.text = "Mask: ON" if self.show_mask else "Mask: OFF"
        self.btn_mask_toggle.active = not self.show_mask
        self.btn_cursor_toggle.text = "Cursor: ON" if self.show_cursor else "Cursor: OFF"
        self.btn_cursor_toggle.active = not self.show_cursor
        self.btn_crosshair_toggle.text = "Cross: ON" if self.show_crosshair else "Cross: OFF"
        self.btn_crosshair_toggle.active = not self.show_crosshair
        self.btn_grid_toggle.text = "Grid: ON" if self.show_grid else "Grid: OFF"
        self.btn_grid_toggle.active = not self.show_grid

    def set_image(self):
        if not self.selected_shape:
            self.show_toast("Select a shape first", 2.0, "error")
            return
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Image or Animation",
            filetypes=[
                ("Image files", "*.gif *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        if path:
            error = self.selected_shape.load_image(path)
            if error:
                self.show_toast(f"Failed to load image: {error}", 4.0, "error")
            else:
                self.show_toast("Image loaded", 2.0, "success")
                # Update speed slider disabled state based on animation
                has_animation = len(self.selected_shape.animation_frames) > 1
                self.speed_slider.disabled = not has_animation

    def fit_warp(self):
        if self.selected_shape:
            self.selected_shape.fit_warp_to_contour()

    def flip_x(self):
        """Flip all selected items horizontally around selection center."""
        if not self.selected_items:
            return

        pivot = self.get_selection_center()
        if not pivot:
            return

        warp_only = (self.mode == self.MODE_EDIT_WARP)

        for item in self.selected_items:
            if isinstance(item, Shape):
                item.flip_x_global(pivot, warp_only=warp_only)
            elif isinstance(item, Group):
                item.flip_x(pivot, warp_only=warp_only)

    def flip_y(self):
        """Flip all selected items vertically around selection center."""
        if not self.selected_items:
            return

        pivot = self.get_selection_center()
        if not pivot:
            return

        warp_only = (self.mode == self.MODE_EDIT_WARP)

        for item in self.selected_items:
            if isinstance(item, Shape):
                item.flip_y_global(pivot, warp_only=warp_only)
            elif isinstance(item, Group):
                item.flip_y(pivot, warp_only=warp_only)

    def rotate(self, clockwise, degrees=90):
        if self.selected_shape:
            # In Edit Warp mode, only transform the warp; otherwise transform both
            warp_only = (self.mode == self.MODE_EDIT_WARP)
            self.selected_shape.rotate(clockwise, degrees, warp_only=warp_only)

    def rotate_with_modifiers(self, clockwise):
        mods = pygame.key.get_mods()
        if mods & pygame.KMOD_CTRL:
            degrees = 1
        elif mods & pygame.KMOD_SHIFT:
            degrees = 10
        else:
            degrees = 5
        self.rotate(clockwise, degrees)

    def adjust_alpha(self, delta):
        """Adjust alpha of selected shape by delta amount"""
        if self.selected_shape:
            self.save_undo_state()
            new_alpha = max(0.0, min(1.0, self.selected_shape.alpha + delta))
            self.selected_shape.alpha = new_alpha

    def adjust_scale(self, delta):
        """Adjust scale of selected shape by delta amount"""
        if self.selected_shape:
            self.save_undo_state()
            self.selected_shape.set_scale(self.selected_shape.scale + delta)

    def adjust_playback_speed(self, delta):
        """Adjust playback speed of selected animated shape"""
        if self.selected_shape and self.selected_shape.animation_frames:
            self.save_undo_state()
            new_speed = max(0.1, min(5.0, self.selected_shape.playback_speed + delta))
            self.selected_shape.playback_speed = new_speed

    def save_scene(self):
        root = Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(
            title="Save Scene",
            defaultextension=".hayai",
            filetypes=[("Hayai Scene", "*.hayai"), ("JSON", "*.json")]
        )
        root.destroy()
        if path:
            # Serialize scene_root (hierarchy format)
            def serialize_items(items: List[SceneItem]) -> List[dict]:
                result = []
                for item in items:
                    result.append(item.to_dict())
                return result

            data = {
                'version': 2,  # Version 2 = hierarchy format
                'scene': serialize_items(self.scene_root)
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.show_toast("Scene saved", 2.0, "success")

    def load_scene(self):
        root = Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Load Scene",
            filetypes=[("Hayai Scene", "*.hayai"), ("JSON", "*.json")]
        )
        root.destroy()
        if path:
            with open(path, 'r') as f:
                data = json.load(f)

            # Check for version to determine format
            version = data.get('version', 1)

            if version >= 2:
                # New hierarchy format
                def deserialize_items(items_data: List[dict]) -> List[SceneItem]:
                    result = []
                    for item_data in items_data:
                        if item_data.get('type') == 'group':
                            result.append(Group.from_dict(item_data))
                        else:
                            result.append(Shape.from_dict(item_data))
                    return result

                self.scene_root = deserialize_items(data.get('scene', []))
                self.rebuild_parent_refs()
            else:
                # Old flat format (backward compatibility)
                shapes = [Shape.from_dict(s) for s in data.get('shapes', [])]
                self.scene_root = shapes  # Shapes at root level
                for shape in shapes:
                    shape._parent = None

            # Rebuild flat shapes list
            self.rebuild_shapes_list()
            self.clear_selection()

            # Update counters and assign names to unnamed items
            self._update_counters_from_scene()
            self.show_toast("Scene loaded", 2.0, "success")

    def _update_counters_from_scene(self):
        """Update shape and group counters based on loaded scene."""
        max_shape_num = 0
        max_group_num = 0

        def process_items(items: List[SceneItem]):
            nonlocal max_shape_num, max_group_num
            for item in items:
                if isinstance(item, Shape):
                    if item.name:
                        if item.name.startswith("Shape_"):
                            try:
                                num = int(item.name.split("_")[1])
                                max_shape_num = max(max_shape_num, num)
                            except (IndexError, ValueError):
                                pass
                    else:
                        item.name = self.generate_shape_name()
                elif isinstance(item, Group):
                    if item.name:
                        if item.name.startswith("Group_"):
                            try:
                                num = int(item.name.split("_")[1])
                                max_group_num = max(max_group_num, num)
                            except (IndexError, ValueError):
                                pass
                    else:
                        item.name = self.generate_group_name()
                    process_items(item.children)

        process_items(self.scene_root)
        self.shape_counter = max(self.shape_counter, max_shape_num)
        self.group_counter = max(self.group_counter, max_group_num)

    def new_scene(self):
        """Clear all shapes and start fresh"""
        # Show confirmation dialog if there are shapes
        if self.scene_root:
            root = Tk()
            root.withdraw()
            result = messagebox.askyesno(
                "New Scene",
                "Are you sure you want to clear all shapes?\nUnsaved changes will be lost.",
                icon='warning'
            )
            root.destroy()
            if not result:
                return
        self.scene_root = []
        self.shapes = []
        self.clear_selection()
        self.current_contour = []
        self.shape_counter = 0  # Reset shape counter
        self.group_counter = 0  # Reset group counter
        self.show_toast("New scene created", 2.0, "info")

    def copy_shape(self):
        if self.selected_shape:
            # Store shape data without the surface (which can't be pickled)
            self.clipboard = {
                'contour': self.selected_shape.contour.copy(),
                'warp_points': self.selected_shape.warp_points.copy(),
                'image_path': self.selected_shape.image_path,
                'perspective_x': self.selected_shape.perspective_x,
                'perspective_y': self.selected_shape.perspective_y,
                'perspective_axis': self.selected_shape.perspective_axis,
                'alpha': self.selected_shape.alpha,
                'hue_shift': self.selected_shape.hue_shift,
                'saturation': self.selected_shape.saturation,
                'color_value': self.selected_shape.color_value,
                'playback_speed': self.selected_shape.playback_speed,
            }
            self.show_toast("Shape copied", 1.5, "info")

    def paste_shape(self):
        if self.clipboard:
            self.save_undo_state()
            new_shape = Shape(
                contour=[(x, y) for x, y in self.clipboard['contour']],
                warp_points=[(x, y) for x, y in self.clipboard['warp_points']],
                image_path=self.clipboard['image_path'],
                perspective_x=self.clipboard['perspective_x'],
                perspective_y=self.clipboard['perspective_y'],
                perspective_axis=self.clipboard['perspective_axis'],
                alpha=self.clipboard.get('alpha', 1.0),
                hue_shift=self.clipboard.get('hue_shift', 0.0),
                saturation=self.clipboard.get('saturation', 1.0),
                color_value=self.clipboard.get('color_value', 1.0),
                playback_speed=self.clipboard.get('playback_speed', 1.0),
            )
            # Assign a unique name
            new_shape.name = self.generate_shape_name()
            # Reload the image if there was one
            if new_shape.image_path and os.path.exists(new_shape.image_path):
                new_shape.load_image(new_shape.image_path)
            new_shape.move(20, 20)  # Offset slightly
            self.add_item_to_scene(new_shape)
            self.select_item(new_shape)
            self.show_toast("Shape pasted", 1.5, "info")

    def delete_shape(self):
        """Delete all selected items (shapes and groups)."""
        if not self.selected_items:
            return

        self.save_undo_state()

        # Count items for toast message
        num_shapes = sum(1 for item in self.selected_items if isinstance(item, Shape))
        num_groups = sum(1 for item in self.selected_items if isinstance(item, Group))

        # Delete all selected items
        for item in list(self.selected_items):
            self.remove_item_from_scene(item)

        # Clear selection
        self.clear_selection()

        # Show appropriate toast
        if num_groups > 0 and num_shapes > 0:
            self.show_toast(f"Deleted {num_shapes} shape(s) and {num_groups} group(s)", 1.5, "info")
        elif num_groups > 0:
            self.show_toast(f"Deleted {num_groups} group(s)", 1.5, "info")
        else:
            self.show_toast(f"Deleted {num_shapes} shape(s)", 1.5, "info")

    def get_warp_point_at(self, pos):
        if not self.selected_shape:
            return None
        # Convert world pos to local coords
        local_pos = self.selected_shape.get_local_point(pos[0], pos[1])
        # Use world scale for hit detection threshold
        world_scale = self.selected_shape._get_world_scale()
        for i, wp in enumerate(self.selected_shape.warp_points):
            if math.dist(local_pos, wp) < 15 / world_scale:
                return i
        return None

    def get_vertex_at(self, pos):
        """Find vertex index near position (pos is in world/base coords)"""
        if not self.selected_shape:
            return None
        # Convert world pos to local coords
        local_pos = self.selected_shape.get_local_point(pos[0], pos[1])
        # Use world scale for hit detection threshold
        world_scale = self.selected_shape._get_world_scale()
        for i, v in enumerate(self.selected_shape.contour):
            if math.dist(local_pos, v) < 12 / world_scale:
                return i
        return None

    def get_edge_at(self, pos):
        """Find edge near position, returns (edge_start_index, closest_point_on_edge)"""
        if not self.selected_shape or len(self.selected_shape.contour) < 2:
            return None

        # Convert world pos to local coords
        local_pos = self.selected_shape.get_local_point(pos[0], pos[1])
        contour = self.selected_shape.contour
        n = len(contour)
        # Use world scale for hit detection threshold
        world_scale = self.selected_shape._get_world_scale()

        for i in range(n):
            p1 = contour[i]
            p2 = contour[(i + 1) % n]

            # Calculate closest point on line segment
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length_sq = dx * dx + dy * dy

            if length_sq == 0:
                closest = p1
            else:
                t = max(0, min(1, ((local_pos[0] - p1[0]) * dx + (local_pos[1] - p1[1]) * dy) / length_sq))
                closest = (p1[0] + t * dx, p1[1] + t * dy)

            dist = math.dist(local_pos, closest)
            if dist < 10 / world_scale:
                # Make sure we're not too close to the vertices (those take priority)
                if math.dist(local_pos, p1) > 12 / world_scale and math.dist(local_pos, p2) > 12 / world_scale:
                    return (i, closest)  # Returns local coords

        return None

    def delete_vertex(self):
        """Delete the currently selected vertex"""
        if self.selected_shape and self.selected_vertex is not None:
            self.save_undo_state()
            if len(self.selected_shape.contour) > 1:
                del self.selected_shape.contour[self.selected_vertex]
                # Adjust selected_vertex if needed
                if self.selected_vertex >= len(self.selected_shape.contour):
                    self.selected_vertex = len(self.selected_shape.contour) - 1
                self.hover_vertex = None
            elif len(self.selected_shape.contour) == 1:
                # Last vertex - delete the shape
                self.shapes.remove(self.selected_shape)
                self.selected_shape = None
                self.selected_vertex = None

    def add_vertex_on_edge(self):
        """Add a vertex on the currently hovered edge"""
        if self.selected_shape and self.hover_edge is not None:
            self.save_undo_state()
            edge_idx, point = self.hover_edge
            # Insert new vertex after edge_idx
            self.selected_shape.contour.insert(edge_idx + 1, point)
            self.hover_edge = None

    def perspective_transform(self, src_points, dst_points, point):
        """Simple perspective transform approximation"""
        # Normalize point within source quad
        x, y = point
        x0, y0 = src_points[0]
        x1, y1 = src_points[1]
        x2, y2 = src_points[2]
        x3, y3 = src_points[3]

        # Bilinear interpolation for quad mapping
        # This is a simplified version
        dx0, dy0 = dst_points[0]
        dx1, dy1 = dst_points[1]
        dx2, dy2 = dst_points[2]
        dx3, dy3 = dst_points[3]

        # Calculate normalized coordinates
        width = max(abs(x1 - x0), abs(x2 - x3), 1)
        height = max(abs(y3 - y0), abs(y2 - y1), 1)

        u = (x - x0) / width
        v = (y - y0) / height

        u = max(0, min(1, u))
        v = max(0, min(1, v))

        # Bilinear interpolation
        out_x = (1-u)*(1-v)*dx0 + u*(1-v)*dx1 + u*v*dx2 + (1-u)*v*dx3
        out_y = (1-u)*(1-v)*dy0 + u*(1-v)*dy1 + u*v*dy2 + (1-u)*v*dy3

        return (out_x, out_y)

    def draw_shape_gl(self, shape, selected=False):  # noqa: ARG002
        """Draw shape using OpenGL for proper perspective warping"""
        if len(shape.contour) < 3:
            return

        # Get image to draw (supports animation) - call only once to avoid timing issues
        img = shape.get_current_image()
        if img is None:
            img = DEFAULT_PATTERN

        # Draw warped image if we have 4 warp points
        if len(shape.warp_points) == 4:
            # Get world coordinates and scale for screen
            world_warp = shape.get_world_warp_points()
            wp = [self.scale_point(x, y) for x, y in world_warp]

            # For animated images, always update texture each frame
            is_animated = len(shape.animation_frames) > 1

            # Create/update texture if needed
            if shape.gl_texture_dirty or shape.gl_texture_id == 0 or is_animated:
                if shape.gl_texture_id:
                    gl_renderer.delete_texture(shape.gl_texture_id)
                shape.gl_texture_id, _, _ = gl_renderer.surface_to_texture(img)
                shape.gl_texture_dirty = False

            # Draw shape with texture (pass alpha and HSV for color adjustments)
            if not self.show_mask:
                # Mask OFF: Draw warp quad directly as two triangles
                gl_renderer.draw_textured_quad_direct(
                    shape.gl_texture_id, wp,
                    img.get_width(), img.get_height(),
                    shape.perspective_x, shape.perspective_y,
                    shape.alpha, shape.hue_shift, shape.saturation, shape.color_value
                )
            else:
                # Mask ON: Draw triangulated polygon with texture
                world_contour = shape.get_world_contour()
                scaled_contour = [self.scale_point(x, y) for x, y in world_contour]
                triangles = triangulate_polygon(scaled_contour)
                gl_renderer.draw_textured_triangles(
                    shape.gl_texture_id, triangles, wp,
                    img.get_width(), img.get_height(),
                    shape.perspective_x, shape.perspective_y,
                    shape.alpha, shape.hue_shift, shape.saturation, shape.color_value
                )

    def draw_shape_overlay(self, shape, selected=False):
        """Draw shape overlays (contours, points) to pygame surface"""
        if len(shape.contour) < 3:
            return

        # Get world coordinates for rendering
        world_contour = shape.get_world_contour()
        world_warp = shape.get_world_warp_points()

        # Only draw geometry (verts and edges) if toggle is on and not in Play mode
        if not self.play_mode and self.show_geom:
            # Draw contour outline (scaled from world coords)
            outline_color = SELECTED_COLOR if selected else SHAPE_OUTLINE_COLOR
            points = [self.scale_point(x, y) for x, y in world_contour]
            points = [(int(x), int(y)) for x, y in points]
            pygame.draw.polygon(self.ui_surface, outline_color, points, 2)

            # Draw contour points (scaled from world coords)
            for i, point in enumerate(world_contour):
                sx, sy = self.scale_point(point[0], point[1])
                px, py = int(sx), int(sy)
                if selected and self.mode == self.MODE_EDIT_SHAPE:
                    # Selected vertex (persistent selection) - cyan with thick white outline
                    if i == self.selected_vertex:
                        pygame.draw.circle(self.ui_surface, (80, 200, 255), (px, py), 8)
                        pygame.draw.circle(self.ui_surface, (255, 255, 255), (px, py), 8, 2)
                    # Hovered vertex - blue highlight (colorblind-friendly)
                    elif i == self.hover_vertex:
                        pygame.draw.circle(self.ui_surface, (100, 150, 255), (px, py), 7)
                        pygame.draw.circle(self.ui_surface, (255, 255, 255), (px, py), 7, 1)
                    else:
                        pygame.draw.circle(self.ui_surface, POINT_COLOR, (px, py), 5)
                else:
                    pygame.draw.circle(self.ui_surface, POINT_COLOR, (px, py), 5)

            # Draw hover indicator on edge in edit shape mode (scaled)
            if selected and self.mode == self.MODE_EDIT_SHAPE and self.hover_edge is not None:
                _, hover_point = self.hover_edge
                # hover_point is in local coordinates from get_edge_at - convert to world
                world_hover = self.selected_shape.get_world_point(hover_point[0], hover_point[1])
                sx, sy = self.scale_point(world_hover[0], world_hover[1])
                px, py = int(sx), int(sy)
                # Use cyan for edge indicator (colorblind-friendly)
                pygame.draw.circle(self.ui_surface, (80, 200, 255), (px, py), 8)
                pygame.draw.line(self.ui_surface, (255, 255, 255), (px - 6, py), (px + 6, py), 2)
                pygame.draw.line(self.ui_surface, (255, 255, 255), (px, py - 6), (px, py + 6), 2)

            # Draw warp points if selected and in edit warp mode (scaled from world coords)
            # Use squares instead of circles to differentiate from vertices
            if selected and self.mode == self.MODE_EDIT_WARP:
                # Draw marching ants around warp quad
                self.draw_marching_ants_quad(world_warp)
                for i, wp in enumerate(world_warp):
                    sx, sy = self.scale_point(wp[0], wp[1])
                    px, py = int(sx), int(sy)
                    # Draw blue squares for warp points (colorblind-friendly)
                    rect = pygame.Rect(px - 7, py - 7, 14, 14)
                    pygame.draw.rect(self.ui_surface, (80, 150, 255), rect)
                    pygame.draw.rect(self.ui_surface, (255, 255, 255), rect, 2)

    def draw_current_contour(self):
        if len(self.current_contour) > 0:
            # Scale contour points for display
            points = [self.scale_point(x, y) for x, y in self.current_contour]
            points = [(int(x), int(y)) for x, y in points]
            if len(points) > 1:
                pygame.draw.lines(self.ui_surface, (255, 255, 100), False, points, 2)
            # Draw line from last point to cursor (mouse_pos is already in screen coords)
            last_point = points[-1]
            pygame.draw.line(self.ui_surface, (200, 200, 200, 150), last_point,
                           (int(self.mouse_pos[0]), int(self.mouse_pos[1])), 1)
            # Draw vertices (scaled)
            for point in self.current_contour:
                sx, sy = self.scale_point(point[0], point[1])
                pygame.draw.circle(self.ui_surface, (255, 100, 100), (int(sx), int(sy)), 6)

    def draw_group_bounds(self, group: Group):
        """Draw dashed bounding rectangle around a selected group"""
        bounds = group.get_bounds()
        if bounds == (0, 0, 0, 0):
            return  # No shapes in group

        # Scale bounds to screen coordinates
        min_x, min_y = self.scale_point(bounds[0], bounds[1])
        max_x, max_y = self.scale_point(bounds[2], bounds[3])

        # Draw dashed rectangle
        color = (255, 200, 75)  # Orange-yellow
        dash_length = 8
        gap_length = 4
        width = 2

        def draw_dashed_line(start, end):
            """Draw a dashed line between two points"""
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return
            dx /= length
            dy /= length

            x, y = start
            drawn = 0
            while drawn < length:
                # Start of dash
                x1, y1 = x, y
                # End of dash
                dash_end = min(drawn + dash_length, length)
                x2 = start[0] + dx * dash_end
                y2 = start[1] + dy * dash_end
                pygame.draw.line(self.ui_surface, color, (int(x1), int(y1)), (int(x2), int(y2)), width)
                # Move to next dash
                drawn = dash_end + gap_length
                x = start[0] + dx * drawn
                y = start[1] + dy * drawn

        # Draw the four sides
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        for i in range(4):
            draw_dashed_line(corners[i], corners[(i + 1) % 4])

        # Draw corner circles
        for cx, cy in corners:
            pygame.draw.circle(self.ui_surface, color, (int(cx), int(cy)), 4)

    def draw_marching_ants_rect(self, bounds: Tuple[float, float, float, float]):
        """
        Draw marching ants (animated dashed border) around the selection bounds.
        Uses white dashes that animate over time.
        """
        # Update marching ants animation
        current_time = time.time()
        elapsed = current_time - self.last_march_time
        self.march_offset = (self.march_offset + elapsed * MARCH_SPEED) % (MARCH_DASH + MARCH_GAP)
        self.last_march_time = current_time

        # Scale bounds to screen coordinates
        min_x, min_y = self.scale_point(bounds[0], bounds[1])
        max_x, max_y = self.scale_point(bounds[2], bounds[3])

        color = HANDLE_COLOR  # White
        width = 1

        def draw_marching_line(start: Tuple[float, float], end: Tuple[float, float]):
            """Draw an animated dashed line between two points"""
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return
            dx /= length
            dy /= length

            # Start drawing offset by march_offset for animation
            pos = -self.march_offset
            while pos < length:
                # Start of dash
                dash_start = max(0, pos)
                dash_end = min(pos + MARCH_DASH, length)
                if dash_end > 0 and dash_start < length:
                    x1 = start[0] + dx * dash_start
                    y1 = start[1] + dy * dash_start
                    x2 = start[0] + dx * dash_end
                    y2 = start[1] + dy * dash_end
                    pygame.draw.line(self.ui_surface, color,
                                   (int(x1), int(y1)), (int(x2), int(y2)), width)
                pos += MARCH_DASH + MARCH_GAP

        # Draw the four sides
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        for i in range(4):
            draw_marching_line(corners[i], corners[(i + 1) % 4])

    def draw_marquee_rect(self):
        """
        Draw the marquee selection rectangle during drag.
        Uses a semi-transparent blue fill with a dashed border.
        """
        if not self.marquee_start or not self.marquee_end:
            return

        # Convert to screen coordinates
        x1, y1 = self.scale_point(self.marquee_start[0], self.marquee_start[1])
        x2, y2 = self.scale_point(self.marquee_end[0], self.marquee_end[1])

        # Normalize to min/max
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        # Draw semi-transparent fill
        fill_color = (100, 150, 255, 40)  # Light blue, very transparent
        rect = pygame.Rect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
        fill_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        fill_surface.fill(fill_color)
        self.ui_surface.blit(fill_surface, rect.topleft)

        # Draw dashed border
        border_color = (100, 150, 255)  # Light blue
        dash_length = 6
        gap_length = 4

        def draw_dashed_line(start: Tuple[float, float], end: Tuple[float, float]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return
            dx /= length
            dy /= length

            pos = 0
            while pos < length:
                dash_start = pos
                dash_end = min(pos + dash_length, length)
                x1 = start[0] + dx * dash_start
                y1 = start[1] + dy * dash_start
                x2 = start[0] + dx * dash_end
                y2 = start[1] + dy * dash_end
                pygame.draw.line(self.ui_surface, border_color,
                               (int(x1), int(y1)), (int(x2), int(y2)), 1)
                pos += dash_length + gap_length

        # Draw the four sides
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        for i in range(4):
            draw_dashed_line(corners[i], corners[(i + 1) % 4])

    def draw_marching_ants_quad(self, world_warp: List[Tuple[float, float]]):
        """Draw marching ants around warp quad edges"""
        if len(world_warp) != 4:
            return

        current_time = time.time()
        elapsed = current_time - self.last_march_time
        self.march_offset = (self.march_offset + elapsed * MARCH_SPEED) % (MARCH_DASH + MARCH_GAP)
        self.last_march_time = current_time

        screen_warp = [self.scale_point(wp[0], wp[1]) for wp in world_warp]
        color = HANDLE_COLOR

        def draw_marching_line(start, end):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length == 0:
                return
            dx, dy = dx / length, dy / length

            pos = -self.march_offset
            while pos < length:
                dash_start = max(0, pos)
                dash_end = min(pos + MARCH_DASH, length)
                if dash_end > 0 and dash_start < length:
                    x1, y1 = start[0] + dx * dash_start, start[1] + dy * dash_start
                    x2, y2 = start[0] + dx * dash_end, start[1] + dy * dash_end
                    pygame.draw.line(self.ui_surface, color, (int(x1), int(y1)), (int(x2), int(y2)), 1)
                pos += MARCH_DASH + MARCH_GAP

        for i in range(4):
            draw_marching_line(screen_warp[i], screen_warp[(i + 1) % 4])

    def get_handle_at(self, pos: Tuple[float, float]) -> Optional[str]:
        """
        Check if the given position is over a grasp handle.
        Returns handle ID ("tl", "tr", "br", "bl", "t", "r", "b", "l") or None.
        """
        bounds = self.get_selection_bounds()
        if bounds is None:
            return None

        min_x, min_y, max_x, max_y = bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        # Handle positions (in world coords)
        handles = {
            'tl': (min_x, min_y), 'tr': (max_x, min_y),
            'bl': (min_x, max_y), 'br': (max_x, max_y),
            't': (mid_x, min_y), 'b': (mid_x, max_y),
            'l': (min_x, mid_y), 'r': (max_x, mid_y),
        }

        # Check corner handles first (larger hit area)
        for handle_id in ('tl', 'tr', 'bl', 'br'):
            hx, hy = handles[handle_id]
            if math.dist(pos, (hx, hy)) <= HANDLE_SIZE + 2:
                return handle_id

        # Check edge handles
        for handle_id in ('t', 'b', 'l', 'r'):
            hx, hy = handles[handle_id]
            if math.dist(pos, (hx, hy)) <= EDGE_HANDLE_SIZE + 2:
                return handle_id

        return None

    def draw_grasp_handles(self, bounds: Tuple[float, float, float, float]):
        """
        Draw grasp handles (squares at corners, circles at edge midpoints).
        Highlights the hovered handle.
        """
        min_x, min_y, max_x, max_y = bounds
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        # Convert to screen coordinates
        def to_screen(x, y):
            sx, sy = self.scale_point(x, y)
            return (int(sx), int(sy))

        # Handle positions
        corners = {
            'tl': to_screen(min_x, min_y), 'tr': to_screen(max_x, min_y),
            'bl': to_screen(min_x, max_y), 'br': to_screen(max_x, max_y),
        }
        edges = {
            't': to_screen(mid_x, min_y), 'b': to_screen(mid_x, max_y),
            'l': to_screen(min_x, mid_y), 'r': to_screen(max_x, mid_y),
        }

        # Draw corner handles (squares)
        for handle_id, (cx, cy) in corners.items():
            color = HANDLE_HOVER_COLOR if handle_id == self.hover_handle else HANDLE_COLOR
            rect = pygame.Rect(cx - HANDLE_SIZE, cy - HANDLE_SIZE,
                             HANDLE_SIZE * 2, HANDLE_SIZE * 2)
            pygame.draw.rect(self.ui_surface, color, rect)
            pygame.draw.rect(self.ui_surface, (0, 0, 0), rect, 1)

        # Draw edge handles (circles)
        for handle_id, (cx, cy) in edges.items():
            color = HANDLE_HOVER_COLOR if handle_id == self.hover_handle else HANDLE_COLOR
            pygame.draw.circle(self.ui_surface, color, (cx, cy), EDGE_HANDLE_SIZE)
            pygame.draw.circle(self.ui_surface, (0, 0, 0), (cx, cy), EDGE_HANDLE_SIZE, 1)

    def draw_ui(self):
        # Always draw toasts even when UI is hidden
        self.draw_toasts()

        # Draw crosshair at cursor position (hidden in Play mode)
        if not self.play_mode and self.show_crosshair:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            crosshair_color = (100, 100, 100, 180)  # Semi-transparent gray
            # Horizontal line
            pygame.draw.line(self.ui_surface, crosshair_color, (0, mouse_y), (WIDTH, mouse_y), 1)
            # Vertical line
            pygame.draw.line(self.ui_surface, crosshair_color, (mouse_x, 0), (mouse_x, HEIGHT), 1)

        if self.play_mode:
            return

        # Scaled dimensions (matching create_ui)
        base_margin = self.panel_margin
        hierarchy_offset = int(self.hierarchy_panel.COLLAPSED_WIDTH * self.ui_scale)
        padding = int(10 * self.ui_scale)
        border_radius = int(8 * self.ui_scale)

        # Helper function to draw styled panel with header
        def draw_panel(x, y, width, height, title):
            panel_rect = pygame.Rect(x, y, width, height)
            # Background
            pygame.draw.rect(self.ui_surface, (30, 30, 35, 230), panel_rect, border_radius=border_radius)
            # Border
            pygame.draw.rect(self.ui_surface, (60, 60, 70), panel_rect, width=2, border_radius=border_radius)
            # Inner highlight
            highlight_rect = pygame.Rect(x + 2, y + 2, width - 4, 1)
            pygame.draw.rect(self.ui_surface, (80, 80, 90, 100), highlight_rect)
            # Header
            header = self.font_small.render(title, True, (120, 120, 130))
            self.ui_surface.blit(header, (x + padding, y + int(8 * self.ui_scale)))

        # TOOLS panel (center top) - includes subtitles, creation row, mode buttons, sub-buttons
        tools_panel_width = int(385 * self.ui_scale)
        tools_panel_height = int(179 * self.ui_scale)
        tools_panel_x = (WIDTH - tools_panel_width) // 2
        tools_panel_y = self.panel_margin_v
        draw_panel(tools_panel_x, tools_panel_y, tools_panel_width, tools_panel_height, "TOOLS")

        # Draw section subtitles in Tools panel
        subtitle_color = (150, 150, 160)
        create_subtitle = self.font_small.render("Create", True, subtitle_color)
        self.ui_surface.blit(create_subtitle, (tools_panel_x + padding, self.tools_create_subtitle_y))
        edit_subtitle = self.font_small.render("Edit", True, subtitle_color)
        self.ui_surface.blit(edit_subtitle, (tools_panel_x + padding, self.tools_edit_subtitle_y))

        # SCENE panel (left side, top)
        scene_panel_width = int(120 * self.ui_scale)
        scene_panel_height = int(138 * self.ui_scale)
        scene_panel_x = base_margin + hierarchy_offset
        scene_panel_y = self.panel_margin_v
        draw_panel(scene_panel_x, scene_panel_y, scene_panel_width, scene_panel_height, "SCENE")

        # DISPLAY panel (left side, bottom) - 6 buttons + slider
        display_panel_width = int(120 * self.ui_scale)
        display_panel_height = int(285 * self.ui_scale)
        display_panel_x = base_margin + hierarchy_offset
        display_panel_y = HEIGHT - self.panel_margin_v - display_panel_height
        draw_panel(display_panel_x, display_panel_y, display_panel_width, display_panel_height, "DISPLAY")

        # Draw buttons
        for button in self.buttons:
            button.draw(self.ui_surface, self.font_small)

        # Draw sides entry (for regular polygon creation)
        if self.sides_entry:
            self.sides_entry.draw(self.ui_surface, self.font_small)

        # Draw contextual instructions based on current mode
        self.draw_contextual_instructions()

        # Draw properties panel (right side)
        self.draw_properties_panel()

        # Draw UI scale slider (bottom-right corner)
        if self.ui_scale_slider:
            self.ui_scale_slider.draw(self.ui_surface, self.font_small)

        # NOTE: Tooltips are drawn after hierarchy panel in draw() to ensure
        # they appear on top of the panel

    def draw_properties_panel(self):
        """Draw the properties panel on right side"""
        # Don't show selection UI when geometry display is off
        if not self.show_geom:
            return

        # Determine what to show properties for
        selected_item = None
        is_group = False
        is_multi = False

        if len(self.selected_items) > 1:
            # Multiple items selected
            is_multi = True
        elif self.selected_shape:
            selected_item = self.selected_shape
        elif len(self.selected_items) == 1 and isinstance(self.selected_items[0], Group):
            selected_item = self.selected_items[0]
            is_group = True

        if not selected_item and not is_multi:
            return

        # Update panel values when selection changes
        if selected_item:
            self.update_properties_panel()

        # Scale dimensions
        panel_width = int(220 * self.ui_scale)
        panel_x = WIDTH - panel_width - self.panel_margin
        panel_y = self.panel_margin_v
        padding = int(10 * self.ui_scale)
        line_height = int(18 * self.ui_scale)
        border_radius = int(8 * self.ui_scale)
        slider_width = int(180 * self.ui_scale)

        # Panel height varies by content
        if is_multi:
            count = len(self.selected_items)
            max_visible = 10  # Max items to show before truncating
            visible_count = min(count, max_visible)
            # Base height (header + count line) + name lines + potential "more" line
            extra_lines = 1 if count > max_visible else 0
            panel_height = int((55 + (visible_count + extra_lines) * 18) * self.ui_scale)
        elif is_group:
            panel_height = int(120 * self.ui_scale)
        else:
            panel_height = int(420 * self.ui_scale)

        # Draw panel background with rounded corners and border
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        # Background
        pygame.draw.rect(self.ui_surface, (30, 30, 35, 230), panel_rect, border_radius=border_radius)
        # Border - subtle gradient effect with inner highlight
        pygame.draw.rect(self.ui_surface, (60, 60, 70), panel_rect, width=2, border_radius=border_radius)
        # Inner highlight line at top
        highlight_rect = pygame.Rect(panel_x + 2, panel_y + 2, panel_width - 4, 1)
        pygame.draw.rect(self.ui_surface, (80, 80, 90, 100), highlight_rect)

        # Draw header
        header = self.font_small.render("PROPERTIES", True, (120, 120, 130))
        self.ui_surface.blit(header, (panel_x + padding, panel_y + int(8 * self.ui_scale)))

        if is_multi:
            # Multiple items selected - show count and list names
            count = len(self.selected_items)
            max_visible = 10
            info_y = panel_y + int(35 * self.ui_scale)

            # Header line with count
            count_text = f"{count} items selected"
            self.ui_surface.blit(self.font_small.render(count_text, True, (180, 180, 180)),
                                 (panel_x + padding, info_y))
            info_y += line_height

            # List each item with type indicator
            max_name_width = slider_width - int(10 * self.ui_scale)
            for i, item in enumerate(self.selected_items[:max_visible]):
                # Type indicator prefix
                prefix = "[G] " if isinstance(item, Group) else "[S] "
                name = prefix + item.name
                # Truncate if needed
                while self.font_small.size(name)[0] > max_name_width and len(name) > 8:
                    name = name[:-4] + "..."
                self.ui_surface.blit(self.font_small.render(name, True, (160, 160, 160)),
                                     (panel_x + padding, info_y))
                info_y += line_height

            # Show overflow indicator if more items than visible limit
            if count > max_visible:
                more_text = f"... and {count - max_visible} more"
                self.ui_surface.blit(self.font_small.render(more_text, True, (120, 120, 130)),
                                     (panel_x + padding, info_y))
        elif is_group:
            # For Groups: show name and transform info
            self.name_input.draw(self.ui_surface, self.font_small, self.ui_scale)
            group = selected_item
            info_y = panel_y + int(78 * self.ui_scale)
            pos_text = f"Pos: ({group.position[0]:.0f}, {group.position[1]:.0f})"
            rot_text = f"Rot: {group.rotation:.1f}°  Scale: {group.scale:.2f}"
            self.ui_surface.blit(self.font_small.render(pos_text, True, (160, 160, 160)),
                                 (panel_x + padding, info_y))
            self.ui_surface.blit(self.font_small.render(rot_text, True, (160, 160, 160)),
                                 (panel_x + padding, info_y + line_height))
        else:
            # For Shapes: show full panel
            shape = selected_item

            # Draw name input
            self.name_input.draw(self.ui_surface, self.font_small, self.ui_scale)

            # Draw image section with label and field box
            label_text = self.font_small.render("Image", True, (180, 180, 180))
            self.ui_surface.blit(label_text, (panel_x + padding, self.image_label_y))

            # Draw field box for image filename
            field_height = int(22 * self.ui_scale)
            field_rect = pygame.Rect(panel_x + padding, self.image_field_y, slider_width, field_height)
            pygame.draw.rect(self.ui_surface, (40, 40, 48), field_rect, border_radius=4)
            pygame.draw.rect(self.ui_surface, (60, 60, 70), field_rect, 1, border_radius=4)

            if shape.image_path:
                # Show only filename, truncate if too long
                filename = os.path.basename(shape.image_path)
                max_width = slider_width - int(10 * self.ui_scale)
                while self.font_small.size(filename)[0] > max_width and len(filename) > 4:
                    filename = "..." + filename[4:]
                path_text = self.font_small.render(filename, True, (160, 160, 160))
            else:
                path_text = self.font_small.render("(no image)", True, (100, 100, 100))
            text_y = self.image_field_y + (field_height - path_text.get_height()) // 2
            self.ui_surface.blit(path_text, (panel_x + padding + int(5 * self.ui_scale), text_y))

            # Draw sliders
            for slider in self.properties_sliders:
                slider.draw(self.ui_surface, self.font_small)

    def draw_toasts(self):
        """Draw and manage toast notifications"""
        # Remove expired toasts
        self.toasts = [t for t in self.toasts if not t.is_expired()]

        # Draw toasts from bottom, stacking up
        y_offset = 0
        for toast in self.toasts:
            height = toast.draw(self.ui_surface, self.font_small, y_offset, self.ui_scale)
            y_offset += height

    def draw_contextual_instructions(self):
        """Draw mode-specific instructions in a panel"""
        if not self.show_controls:
            return

        # Common instructions
        common = [
            "Ctrl+Z/Y: Undo/Redo",
            "SPACE: Toggle UI",
            "F11: Fullscreen",
        ]

        # Mode-specific instructions
        if self.mode == self.MODE_MAKE_SHAPE:
            if self.placing_shape and self.pending_shape_type:
                mode_instructions = [
                    f"Click to place {self.pending_shape_type}",
                ]
            else:
                mode_instructions = [
                    "Click to add vertices",
                    "Click near start to close",
                    "ENTER: Close polygon",
                ]
        elif self.mode == self.MODE_MOVE_SHAPE:
            mode_instructions = [
                "Click shape to select",
                "Click selects entire group",
                "Shift+Click: Select individual",
                "Ctrl+Click: Add to selection",
                "Drag to move selection",
                "Right-drag: Rotate selection",
                "Wheel: Scale (Ctrl=1%, Shift=10%)",
                "DEL: Delete selected",
                "Ctrl+C/V: Copy/Paste",
                "Ctrl+G: Group selection",
                "Ctrl+U: Ungroup",
                "Arrows: Move (Shift=10x)",
            ]
        elif self.mode == self.MODE_EDIT_SHAPE:
            mode_instructions = [
                "Click vertex to select",
                "Drag vertex to move",
                "Click edge to add point",
                "DEL: Delete vertex",
                "Arrows: Move vertex",
            ]
        elif self.mode == self.MODE_EDIT_WARP:
            mode_instructions = [
                "Drag corners to warp",
                "Right-drag: Rotate warp",
                "Wheel: Perspective amount",
                "Middle-click: Switch axis",
                "Arrows: Move warp points",
            ]
        else:
            mode_instructions = []

        # Combine instructions
        instructions = mode_instructions + [""] + common

        # Calculate panel dimensions
        line_height = int(18 * self.ui_scale)
        padding = int(10 * self.ui_scale)
        border_radius = int(8 * self.ui_scale)
        header_height = int(28 * self.ui_scale)

        # Find widest instruction for panel width
        max_width = 0
        for inst in instructions:
            if inst:
                w = self.font_small.size(inst)[0]
                if w > max_width:
                    max_width = w

        panel_width = max_width + padding * 2 + int(10 * self.ui_scale)
        panel_height = header_height + len(instructions) * line_height + padding
        panel_x = WIDTH - panel_width - self.panel_margin
        panel_y = HEIGHT - panel_height - self.panel_margin_v

        # Draw panel background
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.ui_surface, (30, 30, 35, 230), panel_rect, border_radius=border_radius)
        pygame.draw.rect(self.ui_surface, (60, 60, 70), panel_rect, width=2, border_radius=border_radius)
        highlight_rect = pygame.Rect(panel_x + 2, panel_y + 2, panel_width - 4, 1)
        pygame.draw.rect(self.ui_surface, (80, 80, 90, 100), highlight_rect)

        # Draw header
        header = self.font_small.render("CONTROLS", True, (120, 120, 130))
        self.ui_surface.blit(header, (panel_x + padding, panel_y + int(8 * self.ui_scale)))

        # Draw instructions
        y = panel_y + header_height
        for inst in instructions:
            if inst:  # Skip empty lines but use them for spacing
                text = self.font_small.render(inst, True, (180, 180, 180))
                self.ui_surface.blit(text, (panel_x + padding, y))
            y += line_height

    def handle_events(self):
        global WIDTH, HEIGHT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Handle window resize
            if event.type == pygame.VIDEORESIZE and not self.fullscreen:
                WIDTH, HEIGHT = event.w, event.h
                # Context may be recreated, reinitialize
                gl_renderer.invalidate()
                for shape in self.shapes:
                    shape.gl_texture_id = 0
                    shape.gl_texture_dirty = True
                pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT),
                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                gl_renderer.init_gl(WIDTH, HEIGHT)
                self.ui_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                # Recreate UI elements at new positions
                self.create_ui()
                self.create_properties_panel()
                self.create_scale_slider()

            # Handle context menu events first (highest priority)
            if self.context_menu.handle_event(event):
                continue

            # Handle hierarchy panel events (panel handles its own collapsed/expanded state)
            if self.hierarchy_panel.handle_event(event):
                continue

            # Handle button events first if UI is visible
            if not self.play_mode:
                for button in self.buttons:
                    if button.handle_event(event):
                        return

            # Handle UI scale slider events (always available when UI is shown)
            if not self.play_mode and self.ui_scale_slider:
                if self.ui_scale_slider.handle_event(event):
                    return

            # Handle sides entry events (for regular polygon creation)
            if not self.play_mode and self.sides_entry:
                if self.sides_entry.handle_event(event):
                    return

            # Handle properties panel events (sliders, buttons, and text input)
            # Check for either a selected shape OR a selected group
            has_properties_target = self.selected_shape or (len(self.selected_items) == 1 and isinstance(self.selected_items[0], Group))
            if not self.play_mode and has_properties_target:
                if self.name_input and self.name_input.handle_event(event):
                    return
                # Only handle shape-specific controls when a shape is selected
                if self.selected_shape:
                    for slider in self.properties_sliders:
                        if slider.handle_event(event):
                            return

            # Handle drag-and-drop for images
            if event.type == pygame.DROPFILE:
                dropped_path = event.file
                # Check if it's an image file
                image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
                if dropped_path.lower().endswith(image_extensions):
                    if self.selected_shape:
                        error = self.selected_shape.load_image(dropped_path)
                        if error:
                            self.show_toast(f"Failed to load image: {error}", 4.0, "error")
                        else:
                            self.show_toast("Image loaded via drag-and-drop", 2.0, "success")
                            # Update speed slider disabled state based on animation
                            has_animation = len(self.selected_shape.animation_frames) > 1
                            self.speed_slider.disabled = not has_animation
                    else:
                        self.show_toast("Select a shape first to drop an image", 3.0, "error")
                else:
                    # Check if it's a scene file
                    if dropped_path.lower().endswith(('.hayai', '.json')):
                        try:
                            with open(dropped_path, 'r') as f:
                                data = json.load(f)
                            self.shapes = [Shape.from_dict(s) for s in data.get('shapes', [])]
                            self.selected_shape = None
                            self.show_toast("Scene loaded via drag-and-drop", 2.0, "success")
                        except Exception as e:
                            self.show_toast(f"Failed to load scene: {e}", 4.0, "error")
                    else:
                        self.show_toast("Unsupported file type", 2.0, "error")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.fullscreen:
                        self.toggle_fullscreen()  # Exit fullscreen
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_SPACE:
                    self.toggle_play_mode()
                elif event.key == pygame.K_h:
                    # Toggle hierarchy panel collapsed/expanded state with animation
                    import time
                    self.hierarchy_panel.collapsed = not self.hierarchy_panel.collapsed
                    self.hierarchy_panel.animating = True
                    self.hierarchy_panel.animation_start = time.time()
                elif event.key == pygame.K_g and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        # Ctrl+Shift+G: Ungroup
                        self.ungroup_selected()
                    else:
                        # Ctrl+G: Group selected items
                        self.group_selected_items()
                elif event.key == pygame.K_u and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    # Ctrl+U: Ungroup (alternative shortcut)
                    self.ungroup_selected()
                elif event.key == pygame.K_F2:
                    # Rename selected item via hierarchy panel
                    if self.selected_items:
                        self.hierarchy_panel._start_rename()
                elif event.key == pygame.K_DELETE:
                    if self.mode == self.MODE_EDIT_SHAPE:
                        # Only delete vertex if one is selected
                        if self.selected_vertex is not None:
                            self.delete_vertex()
                        # Don't delete shape in edit mode unless it's via last vertex deletion
                    elif self.mode == self.MODE_MOVE_SHAPE:
                        self.delete_shape()
                    elif self.mode == self.MODE_MAKE_SHAPE:
                        # Delete the last created shape
                        if self.shapes:
                            self.save_undo_state()
                            last_shape = self.shapes[-1]
                            self.remove_item_from_scene(last_shape)
                            self.clear_selection()
                            self.show_toast(f"Deleted {last_shape.name}", 1.5, "info")
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.copy_shape()
                elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.paste_shape()
                elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.undo()
                elif event.key == pygame.K_y and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    self.redo()
                elif event.key == pygame.K_TAB and not self.play_mode:
                    # Tab navigation for buttons
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.focus_prev_button()
                    else:
                        self.focus_next_button()
                elif event.key == pygame.K_RETURN:
                    # Close polygon with Enter in poly drawing mode
                    if self.mode == self.MODE_MAKE_SHAPE and not self.placing_shape:
                        if len(self.current_contour) >= 3:
                            self.save_undo_state()
                            new_shape = Shape(contour=self.current_contour.copy())
                            new_shape.name = self.generate_shape_name()
                            new_shape.fit_warp_to_contour()
                            self.add_item_to_scene(new_shape)
                            self.current_contour = []  # Clear before select_item so button states update correctly
                            self.select_item(new_shape)
                            self.show_toast("Shape created", 1.5, "success")
                    # Activate focused button with Enter
                    elif self.focused_button_index >= 0:
                        self.activate_focused_button()
                elif event.key in (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT):
                    if self.selected_items:  # Works for shapes AND groups
                        amount = 10 if pygame.key.get_mods() & pygame.KMOD_SHIFT else 1
                        dx = dy = 0
                        if event.key == pygame.K_UP: dy = -amount
                        elif event.key == pygame.K_DOWN: dy = amount
                        elif event.key == pygame.K_LEFT: dx = -amount
                        elif event.key == pygame.K_RIGHT: dx = amount
                        # In edit shape mode with selected vertex, move just the vertex
                        if self.mode == self.MODE_EDIT_SHAPE and self.selected_shape and self.selected_vertex is not None:
                            # Convert world delta to local delta for screen-direction movement
                            local_dx, local_dy = self.selected_shape.world_delta_to_local(dx, dy)
                            vx, vy = self.selected_shape.contour[self.selected_vertex]
                            self.selected_shape.contour[self.selected_vertex] = (vx + local_dx, vy + local_dy)
                        # In edit warp mode, only move the warp points
                        elif self.mode == self.MODE_EDIT_WARP and self.selected_shape:
                            # Convert world delta to local delta for screen-direction movement
                            local_dx, local_dy = self.selected_shape.world_delta_to_local(dx, dy)
                            self.selected_shape.warp_points = [
                                (x + local_dx, y + local_dy) for x, y in self.selected_shape.warp_points
                            ]
                        else:
                            # Move all selected items (supports multi-selection and groups)
                            self.move_selection(dx, dy)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                # Clear button focus when clicking with mouse
                self.clear_button_focus()
                # Convert to base coordinates for shape interactions
                base_pos = self.unscale_point(pos[0], pos[1])

                if event.button == 1:  # Left click
                    if self.mode == self.MODE_MAKE_SHAPE:
                        # Shape placement mode (Create Regular)
                        if self.placing_shape and self.pending_shape_type == "regular":
                            self.save_undo_state()
                            contour = self.create_regular_polygon(base_pos[0], base_pos[1],
                                                                  self.regular_polygon_sides,
                                                                  radius=80,
                                                                  on_corner=self.on_corner)
                            new_shape = Shape(contour=contour)
                            new_shape.name = self.generate_shape_name()
                            new_shape.fit_warp_to_contour()
                            self.add_item_to_scene(new_shape)
                            self.select_item(new_shape)
                            self.show_toast("Shape created", 1.5, "success")
                            # Stay in placement mode for easy repeated placement
                            return
                        # Free polygon drawing mode (Poly button)
                        # Check if close to start point to close shape (use scaled threshold)
                        if len(self.current_contour) >= 3:
                            start_scaled = self.scale_point(self.current_contour[0][0], self.current_contour[0][1])
                            if math.dist(pos, start_scaled) < 20:
                                # Close shape
                                self.save_undo_state()
                                new_shape = Shape(contour=self.current_contour.copy())
                                new_shape.name = self.generate_shape_name()
                                new_shape.fit_warp_to_contour()
                                self.add_item_to_scene(new_shape)
                                self.current_contour = []  # Clear before select_item so button states update correctly
                                self.select_item(new_shape)
                                self.show_toast("Shape created", 1.5, "success")
                                return
                        self.current_contour.append(base_pos)
                        self.update_mode_button_states()

                    elif self.mode == self.MODE_MOVE_SHAPE:
                        # Get modifier key states
                        mods = pygame.key.get_mods()
                        shift_held = bool(mods & pygame.KMOD_SHIFT)
                        ctrl_held = bool(mods & pygame.KMOD_CTRL)

                        # Check for handle hit first (for scaling)
                        handle = self.get_handle_at(base_pos)
                        if handle is not None and self.selected_items:
                            # Start handle drag for scaling
                            self.active_handle = handle
                            self.handle_drag_start = base_pos
                            self.handle_bounds_start = self.get_selection_bounds()
                            # Calculate fixed point (opposite corner/edge)
                            if self.handle_bounds_start:
                                min_x, min_y, max_x, max_y = self.handle_bounds_start
                                handle_fixed_points = {
                                    'tl': (max_x, max_y), 'tr': (min_x, max_y),
                                    'bl': (max_x, min_y), 'br': (min_x, min_y),
                                    't': ((min_x + max_x) / 2, max_y),
                                    'b': ((min_x + max_x) / 2, min_y),
                                    'l': (max_x, (min_y + max_y) / 2),
                                    'r': (min_x, (min_y + max_y) / 2),
                                }
                                self.handle_fixed_point = handle_fixed_points.get(handle)
                            self.save_undo_state()
                            self.dragging = True
                        else:
                            # Select or start dragging
                            clicked_shape = None
                            for shape in reversed(self.shapes):
                                if shape.contains_point(base_pos):
                                    clicked_shape = shape
                                    break

                            if clicked_shape:
                                # Shift: add to existing selection (multi-select)
                                # Ctrl: select individual shape (bypass group hierarchy)
                                self.select_item(clicked_shape,
                                               add_to_selection=shift_held,
                                               force_individual=ctrl_held)
                                self.save_undo_state()
                                self.dragging = True
                                self.drag_start = base_pos
                            else:
                                # Start marquee selection on empty area
                                if not shift_held:
                                    self.clear_selection()
                                self.marquee_start = base_pos
                                self.marquee_end = base_pos
                                self.marquee_active = True

                    elif self.mode == self.MODE_EDIT_SHAPE:
                        # Check if clicking on a vertex to select/drag it
                        if self.hover_vertex is not None:
                            self.selected_vertex = self.hover_vertex
                            self.dragging_vertex = self.hover_vertex
                            # Freeze pivot to prevent center drift during editing
                            self.selected_shape.freeze_pivot()
                            # Store drag start for delta-based dragging
                            self.drag_start_world = base_pos
                            self.drag_start_local = self.selected_shape.contour[self.hover_vertex]
                            self.save_undo_state()
                            self.dragging = True
                        # Check if clicking on an edge to add a vertex
                        elif self.hover_edge is not None:
                            self.add_vertex_on_edge()
                        # Otherwise try to select a shape (and deselect vertex)
                        else:
                            self.selected_vertex = None
                            clicked_shape = None
                            for shape in reversed(self.shapes):
                                if shape.contains_point(base_pos):
                                    clicked_shape = shape
                                    break
                            if clicked_shape:
                                self.select_item(clicked_shape)

                    elif self.mode == self.MODE_EDIT_WARP:
                        # Check for warp point drag
                        wp_idx = self.get_warp_point_at(base_pos)
                        if wp_idx is not None:
                            self.dragging_warp_point = wp_idx
                            # Freeze pivot to prevent center drift during editing
                            self.selected_shape.freeze_pivot()
                            # Store drag start for delta-based dragging
                            self.drag_start_world = base_pos
                            self.drag_start_local = self.selected_shape.warp_points[wp_idx]
                            self.save_undo_state()
                            self.dragging = True

                elif event.button == 2:  # Middle click - switch perspective axis
                    if self.selected_shape and self.mode == self.MODE_EDIT_WARP:
                        self.selected_shape.perspective_axis = 1 - self.selected_shape.perspective_axis

                elif event.button == 3:  # Right click - start rotation
                    if self.mode == self.MODE_MOVE_SHAPE and self.selected_items:
                        center = self.get_selection_center()
                        if center:
                            self.rotating = True
                            self.rotate_start_center = center
                            self.rotation_pivot_for_ui = center  # Store for UI rendering
                            # Calculate initial angle from center to mouse
                            self.rotate_start_angle = math.degrees(math.atan2(
                                base_pos[1] - center[1],
                                base_pos[0] - center[0]
                            ))
                            self.save_undo_state()
                    elif self.mode == self.MODE_EDIT_WARP and self.selected_shape:
                        warp_world = self.selected_shape.get_world_warp_points()
                        if len(warp_world) == 4:
                            cx = sum(wp[0] for wp in warp_world) / 4
                            cy = sum(wp[1] for wp in warp_world) / 4
                            self.rotating_warp = True
                            self.warp_rotate_center = (cx, cy)
                            self.warp_rotate_start_angle = math.degrees(math.atan2(
                                base_pos[1] - cy, base_pos[0] - cx))
                            self.save_undo_state()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # Unfreeze pivot when done editing vertices/warp points
                    if self.selected_shape and (self.dragging_vertex is not None or self.dragging_warp_point is not None):
                        self.selected_shape.unfreeze_pivot()
                    self.dragging = False
                    self.drag_start = None
                    self.dragging_vertex = None
                    self.dragging_warp_point = None
                    self.drag_start_world = None
                    self.drag_start_local = None
                    # Clear handle scaling state
                    self.active_handle = None
                    self.handle_drag_start = None
                    self.handle_bounds_start = None
                    self.handle_fixed_point = None

                    # Complete marquee selection
                    if self.marquee_active and self.marquee_start and self.marquee_end:
                        # Ctrl+drag selects individual shapes, normal drag selects parent groups
                        mods = pygame.key.get_mods()
                        ctrl_held = bool(mods & pygame.KMOD_CTRL)
                        self.complete_marquee_selection(force_individual=ctrl_held)
                    self.marquee_active = False
                    self.marquee_start = None
                    self.marquee_end = None

                elif event.button == 3:  # Right click release - stop rotation
                    self.rotating = False
                    self.rotate_start_angle = 0.0
                    self.rotate_start_center = None
                    self.rotation_pivot_for_ui = None  # Clear stored pivot
                    self.rotating_warp = False
                    self.warp_rotate_start_angle = 0.0
                    self.warp_rotate_center = None

            elif event.type == pygame.MOUSEMOTION:
                pos = event.pos
                self.mouse_pos = pos  # Track for shape preview (screen coords)
                # Convert to base coordinates for shape interactions
                base_pos = self.unscale_point(pos[0], pos[1])

                # Update hover state for edit shape mode
                if self.mode == self.MODE_EDIT_SHAPE and not self.dragging:
                    self.hover_vertex = self.get_vertex_at(base_pos)
                    if self.hover_vertex is None:
                        self.hover_edge = self.get_edge_at(base_pos)
                    else:
                        self.hover_edge = None

                    # Update cursor for edit shape mode
                    if self.hover_edge is not None:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
                    elif self.hover_vertex is not None:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    else:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                elif self.mode == self.MODE_MAKE_SHAPE:
                    # Crosshair cursor for drawing mode
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
                elif self.mode == self.MODE_MOVE_SHAPE:
                    # Check if hovering over a grasp handle
                    self.hover_handle = self.get_handle_at(base_pos)

                    if self.hover_handle:
                        # Set cursor based on handle type
                        cursor_map = {
                            'tl': pygame.SYSTEM_CURSOR_SIZENWSE,
                            'br': pygame.SYSTEM_CURSOR_SIZENWSE,
                            'tr': pygame.SYSTEM_CURSOR_SIZENESW,
                            'bl': pygame.SYSTEM_CURSOR_SIZENESW,
                            't': pygame.SYSTEM_CURSOR_SIZENS,
                            'b': pygame.SYSTEM_CURSOR_SIZENS,
                            'l': pygame.SYSTEM_CURSOR_SIZEWE,
                            'r': pygame.SYSTEM_CURSOR_SIZEWE,
                        }
                        pygame.mouse.set_cursor(cursor_map.get(self.hover_handle, pygame.SYSTEM_CURSOR_ARROW))
                    elif self.marquee_active:
                        # Crosshair cursor during marquee selection
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
                    else:
                        # Check if hovering over a shape
                        over_shape = any(shape.contains_point(base_pos) for shape in self.shapes)
                        if over_shape:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                elif self.mode == self.MODE_EDIT_WARP:
                    # Check if hovering over a warp point
                    base_pos = self.unscale_point(pos[0], pos[1])
                    wp_idx = self.get_warp_point_at(base_pos)
                    if wp_idx is not None:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    else:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                else:
                    pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

                # Update marquee selection during drag
                if self.marquee_active:
                    self.marquee_end = base_pos

                # Handle right-click rotation
                if self.rotating and self.rotate_start_center:
                    # Calculate angle from center to current mouse position
                    center = self.rotate_start_center
                    current_angle = math.degrees(math.atan2(
                        base_pos[1] - center[1],
                        base_pos[0] - center[0]
                    ))
                    delta_angle = current_angle - self.rotate_start_angle
                    self.rotate_start_angle = current_angle
                    self.rotate_selection(delta_angle, center)

                # Handle warp rotation (Edit Warp mode)
                if self.rotating_warp and self.warp_rotate_center and self.selected_shape:
                    center = self.warp_rotate_center
                    current_angle = math.degrees(math.atan2(
                        base_pos[1] - center[1], base_pos[0] - center[0]))
                    delta_angle = current_angle - self.warp_rotate_start_angle
                    self.warp_rotate_start_angle = current_angle

                    angle_rad = math.radians(delta_angle)
                    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

                    new_warp_points = []
                    for wp in self.selected_shape.warp_points:
                        world_wp = self.selected_shape.get_world_point(wp[0], wp[1])
                        dx, dy = world_wp[0] - center[0], world_wp[1] - center[1]
                        new_world_x = center[0] + dx * cos_a - dy * sin_a
                        new_world_y = center[1] + dx * sin_a + dy * cos_a
                        new_local = self.selected_shape.get_local_point(new_world_x, new_world_y)
                        new_warp_points.append(new_local)
                    self.selected_shape.warp_points = new_warp_points

                if self.dragging:
                    if self.mode == self.MODE_MOVE_SHAPE and self.active_handle:
                        # Handle scaling drag
                        if self.handle_bounds_start and self.handle_fixed_point and self.handle_drag_start:
                            min_x, min_y, max_x, max_y = self.handle_bounds_start
                            orig_w = max_x - min_x
                            orig_h = max_y - min_y

                            # Calculate new bounds based on mouse position
                            fx, fy = self.handle_fixed_point
                            mx, my = base_pos

                            # Calculate scale factors based on handle type
                            if self.active_handle in ('tl', 'tr', 'bl', 'br'):
                                # Corner handles: non-uniform scaling
                                if orig_w > 0:
                                    scale_x = abs(mx - fx) / orig_w
                                else:
                                    scale_x = 1.0
                                if orig_h > 0:
                                    scale_y = abs(my - fy) / orig_h
                                else:
                                    scale_y = 1.0
                            elif self.active_handle in ('l', 'r'):
                                # Left/right edge: horizontal scaling only
                                if orig_w > 0:
                                    scale_x = abs(mx - fx) / orig_w
                                else:
                                    scale_x = 1.0
                                scale_y = 1.0
                            else:  # 't', 'b'
                                # Top/bottom edge: vertical scaling only
                                scale_x = 1.0
                                if orig_h > 0:
                                    scale_y = abs(my - fy) / orig_h
                                else:
                                    scale_y = 1.0

                            # Clamp scale factors to reasonable range
                            scale_x = max(0.01, min(10.0, scale_x))
                            scale_y = max(0.01, min(10.0, scale_y))

                            # Calculate relative scale from last frame
                            # We need to scale relative to original, so restore and apply new scale
                            self.scale_selection(scale_x, scale_y, self.handle_fixed_point)
                            # Update bounds for next frame calculation
                            self.handle_bounds_start = self.get_selection_bounds()

                    elif self.mode == self.MODE_MOVE_SHAPE and self.selected_items and self.drag_start:
                        # Normal move dragging - use move_selection for all selected items
                        dx = base_pos[0] - self.drag_start[0]
                        dy = base_pos[1] - self.drag_start[1]
                        self.move_selection(dx, dy)
                        self.drag_start = base_pos

                    elif self.mode == self.MODE_EDIT_SHAPE and self.dragging_vertex is not None:
                        if self.selected_shape and self.drag_start_world and self.drag_start_local:
                            # Use delta-based dragging to avoid center drift
                            world_dx = base_pos[0] - self.drag_start_world[0]
                            world_dy = base_pos[1] - self.drag_start_world[1]
                            local_dx, local_dy = self.selected_shape.world_delta_to_local(world_dx, world_dy)
                            new_local = (self.drag_start_local[0] + local_dx, self.drag_start_local[1] + local_dy)
                            self.selected_shape.contour[self.dragging_vertex] = new_local

                    elif self.mode == self.MODE_EDIT_WARP and self.dragging_warp_point is not None:
                        if self.selected_shape and self.drag_start_world and self.drag_start_local:
                            # Use delta-based dragging to avoid center drift
                            world_dx = base_pos[0] - self.drag_start_world[0]
                            world_dy = base_pos[1] - self.drag_start_world[1]
                            local_dx, local_dy = self.selected_shape.world_delta_to_local(world_dx, world_dy)
                            new_local = (self.drag_start_local[0] + local_dx, self.drag_start_local[1] + local_dy)
                            self.selected_shape.warp_points[self.dragging_warp_point] = new_local

            elif event.type == pygame.MOUSEWHEEL:
                if self.selected_items:
                    if self.mode == self.MODE_MOVE_SHAPE:
                        # Scale all selected items with mouse wheel
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_CTRL:
                            scale_step = 0.01  # Fine: 1%
                        elif mods & pygame.KMOD_SHIFT:
                            scale_step = 0.1   # Coarse: 10%
                        else:
                            scale_step = 0.05  # Normal: 5%

                        if event.y > 0:
                            scale_factor = 1.0 + scale_step
                        else:
                            scale_factor = 1.0 - scale_step

                        center = self.get_selection_center()
                        if center:
                            self.save_undo_state()
                            self.scale_selection(scale_factor, scale_factor, center)
                    elif self.mode == self.MODE_EDIT_WARP and self.selected_shape:
                        # Adjust perspective on the selected axis
                        delta = event.y * 0.05
                        if self.selected_shape.perspective_axis == 0:
                            self.selected_shape.perspective_x = max(-2.0, min(2.0,
                                self.selected_shape.perspective_x + delta))
                            # Sync slider
                            self.persp_x_slider.value = self.selected_shape.perspective_x
                        else:
                            self.selected_shape.perspective_y = max(-2.0, min(2.0,
                                self.selected_shape.perspective_y + delta))
                            # Sync slider
                            self.persp_y_slider.value = self.selected_shape.perspective_y

    def draw(self):
        bg = BG_COLOR if self.play_mode else BG_COLOR_UI

        # Clear with OpenGL
        gl_renderer.clear(bg[0], bg[1], bg[2])

        # Draw shapes using OpenGL
        for shape in self.shapes:
            self.draw_shape_gl(shape, selected=(shape in self.selected_items))

        # Clear UI surface
        self.ui_surface.fill((0, 0, 0, 0))

        # Draw background grid (hidden in Play mode)
        if not self.play_mode and self.show_grid:
            grid_color = (60, 60, 60, 100)  # Semi-transparent dark gray
            grid_spacing = 20  # Pixels between grid lines
            # Vertical lines
            for x in range(0, WIDTH, grid_spacing):
                pygame.draw.line(self.ui_surface, grid_color, (x, 0), (x, HEIGHT), 1)
            # Horizontal lines
            for y in range(0, HEIGHT, grid_spacing):
                pygame.draw.line(self.ui_surface, grid_color, (0, y), (WIDTH, y), 1)

        # Draw shape overlays to UI surface
        for shape in self.shapes:
            self.draw_shape_overlay(shape, selected=(shape in self.selected_items))

        # Draw selection markers only when geometry is visible
        if self.show_geom:
            # Draw group bounding boxes for selected groups
            for item in self.selected_items:
                if isinstance(item, Group):
                    self.draw_group_bounds(item)

            # Draw marching ants and grasp handles for selection in move mode
            if self.selected_items and self.mode == self.MODE_MOVE_SHAPE:
                bounds = self.get_selection_bounds()
                if bounds:
                    self.draw_marching_ants_rect(bounds)
                    self.draw_grasp_handles(bounds)

                # Draw pivot/center marker (use stored pivot during rotation to prevent drift)
                if self.rotating and self.rotation_pivot_for_ui:
                    center = self.rotation_pivot_for_ui
                else:
                    center = self.get_selection_center()
                if center:
                    cx, cy = self.scale_point(center[0], center[1])
                    cx, cy = int(cx), int(cy)
                    size = int(10 * self.ui_scale)
                    # Draw crosshair
                    pygame.draw.line(self.ui_surface, (255, 255, 0), (cx - size, cy), (cx + size, cy), 2)
                    pygame.draw.line(self.ui_surface, (255, 255, 0), (cx, cy - size), (cx, cy + size), 2)
                    pygame.draw.circle(self.ui_surface, (255, 255, 0), (cx, cy), int(4 * self.ui_scale))

        # Draw marquee selection rectangle
        if self.marquee_active and self.marquee_start and self.marquee_end:
            self.draw_marquee_rect()

        # Draw current contour being created
        if self.mode == self.MODE_MAKE_SHAPE:
            self.draw_current_contour()

        # Draw UI elements
        self.draw_ui()

        # Draw watermark (hidden in play mode)
        if not self.play_mode:
            watermark_text = "Hayai"
            watermark_surface = self.font_small.render(watermark_text, True, (80, 80, 90))
            watermark_surface.set_alpha(120)
            watermark_x = WIDTH - watermark_surface.get_width() - int(10 * self.ui_scale)
            watermark_y = HEIGHT - watermark_surface.get_height() - int(10 * self.ui_scale)
            self.ui_surface.blit(watermark_surface, (watermark_x, watermark_y))

        # Draw hierarchy panel (hidden in play mode)
        if not self.play_mode:
            self.hierarchy_panel.draw(self.ui_surface)

        # Draw tooltips last (on top of everything including hierarchy panel)
        # Suppress UI element tooltips when cursor is over hierarchy panel
        mouse_pos = pygame.mouse.get_pos()
        panel_rect = self.hierarchy_panel.get_rect()
        over_panel = panel_rect.collidepoint(mouse_pos) and not self.play_mode

        if not over_panel:
            for button in self.buttons:
                button.draw_tooltip(self.ui_surface, self.font_small, self.ui_scale)
            # Draw slider and numeric entry tooltips
            self.ui_scale_slider.draw_tooltip(self.ui_surface, self.font_small, self.ui_scale)
            self.sides_entry.draw_tooltip(self.ui_surface, self.font_small, self.ui_scale)

        # Draw hierarchy panel tooltip (always shown when appropriate)
        if not self.play_mode:
            self.hierarchy_panel.draw_tooltip(self.ui_surface, self.font_small, self.ui_scale)

        # Draw context menu (always on top)
        self.context_menu.draw(self.ui_surface, self.font)

        # Render pygame UI surface as OpenGL texture overlay (flip Y for correct orientation)
        # Disable shader and set up fixed-function pipeline for UI overlay
        glUseProgram(0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        ui_tex, _, _ = gl_renderer.surface_to_texture(self.ui_surface)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, ui_tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(WIDTH, 0)
        glTexCoord2f(1, 0); glVertex2f(WIDTH, HEIGHT)
        glTexCoord2f(0, 0); glVertex2f(0, HEIGHT)
        glEnd()
        gl_renderer.delete_texture(ui_tex)

        pygame.display.flip()

    def run(self):
        self.set_mode(self.MODE_MAKE_SHAPE)
        self.btn_create_freeform.active = True

        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


def main():
    app = Hayai()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
