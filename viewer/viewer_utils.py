"""
Common utilities for MDL and WMB viewers.
Provides text rendering, file list overlay, and file navigation.
"""

import pygame
from OpenGL.GL import *
import os
import glob


class TextRenderer:
    """Handles text rendering in OpenGL context"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24)

    def render(self, text, x, y, color=(255, 255, 255), large=False):
        """Render text at screen position (x, y)"""
        font = self.font_large if large else self.font
        # Sanitize text - remove null characters and non-printable chars
        text = ''.join(c if c.isprintable() else '' for c in text)
        if not text:
            return

        surface = font.render(text, True, color)
        text_data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()

        # Save OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()

        # Switch to 2D
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Draw quad
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1); glVertex2f(x, y + height)
        glEnd()

        glDeleteTextures([tex_id])

        # Restore
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()


class FileNavigator:
    """Manages file list and navigation for viewers"""

    def __init__(self, folder, extension, width, height):
        """
        Initialize file navigator.

        Args:
            folder: Directory to search for files
            extension: File extension to search for (e.g., '.wmb', '.mdl')
            width: Screen width for overlay
            height: Screen height for overlay
        """
        self.folder = folder if folder else os.getcwd()
        self.extension = extension
        self.width = width
        self.height = height

        # Find all matching files
        pattern = os.path.join(self.folder, f"*{extension}")
        self.files = sorted(glob.glob(pattern))

        self.current_index = 0
        self.show_file_list = False
        self.list_scroll_offset = 0

        self.text_renderer = TextRenderer(width, height)

    @property
    def current_file(self):
        """Get current file path"""
        if not self.files:
            return None
        return self.files[self.current_index]

    @property
    def current_filename(self):
        """Get current file name (without path)"""
        if not self.files:
            return None
        return os.path.basename(self.files[self.current_index])

    @property
    def file_count(self):
        """Get total number of files"""
        return len(self.files)

    def switch_file(self, delta):
        """Switch to next/previous file. Returns True if changed."""
        if not self.files:
            return False
        new_index = (self.current_index + delta) % len(self.files)
        if new_index != self.current_index:
            self.current_index = new_index
            return True
        return False

    def switch_to_index(self, index):
        """Switch to file at specific index. Returns True if changed."""
        if not self.files:
            return False
        if 0 <= index < len(self.files) and index != self.current_index:
            self.current_index = index
            return True
        return False

    def handle_event(self, event):
        """
        Handle pygame event for file navigation.
        Returns: (action, data) tuple where action is:
            - 'switch': file was switched, data is new index
            - 'toggle_list': list visibility toggled
            - None: no navigation action
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE and self.show_file_list:
                self.show_file_list = False
                return ('toggle_list', False)
            elif event.key == pygame.K_LEFT:
                if self.switch_file(-1):
                    return ('switch', self.current_index)
            elif event.key == pygame.K_RIGHT:
                if self.switch_file(1):
                    return ('switch', self.current_index)
            elif event.key == pygame.K_l:
                self.show_file_list = not self.show_file_list
                self.list_scroll_offset = max(0, self.current_index - 10)
                return ('toggle_list', self.show_file_list)
            elif event.key == pygame.K_UP and self.show_file_list:
                self.list_scroll_offset = max(0, self.list_scroll_offset - 1)
            elif event.key == pygame.K_DOWN and self.show_file_list:
                max_scroll = max(0, len(self.files) - 20)
                self.list_scroll_offset = min(max_scroll, self.list_scroll_offset + 1)
            elif event.key == pygame.K_RETURN and self.show_file_list:
                self.show_file_list = False
                return ('toggle_list', False)
            elif event.key == pygame.K_HOME:
                if self.switch_to_index(0):
                    return ('switch', self.current_index)
            elif event.key == pygame.K_END:
                if self.switch_to_index(len(self.files) - 1):
                    return ('switch', self.current_index)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4 and self.show_file_list:  # Scroll up
                self.list_scroll_offset = max(0, self.list_scroll_offset - 3)
            elif event.button == 5 and self.show_file_list:  # Scroll down
                max_scroll = max(0, len(self.files) - 20)
                self.list_scroll_offset = min(max_scroll, self.list_scroll_offset + 3)
            elif self.show_file_list and event.button == 1 and event.pos[0] < 320:
                # Click on file list
                # File list starts at pygame y=100 (below title at y=70)
                # Each item is 22 pixels tall, pygame y increases downward
                # Max 20 visible items, so valid area is y=100 to y=100+20*22=540
                visible_files = 20
                max_y = 100 + visible_files * 22
                if 100 <= event.pos[1] < max_y:
                    clicked_idx = self.list_scroll_offset + (event.pos[1] - 100) // 22
                    if 0 <= clicked_idx < len(self.files):
                        if self.switch_to_index(clicked_idx):
                            self.show_file_list = False
                            return ('switch', self.current_index)

        return (None, None)

    def draw_overlay(self, extra_info=None, error=None, controls_hint=None):
        """
        Draw the file info overlay.

        Args:
            extra_info: Additional info line to show below filename
            error: Error message to show in red
            controls_hint: Custom controls hint (default shows navigation controls)
        """
        if not self.files:
            self.text_renderer.render("No files found!", 10, self.height - 30, (255, 100, 100), large=True)
            return

        # Current file name at top
        filename = os.path.basename(self.files[self.current_index])
        info_text = f"{filename} ({self.current_index + 1}/{len(self.files)})"
        self.text_renderer.render(info_text, 10, self.height - 30, (255, 255, 255), large=True)

        # Controls hint at bottom
        if controls_hint is None:
            controls_hint = "Left/Right: navigate | L: file list | Home/End: first/last"
        self.text_renderer.render(controls_hint, 10, 10, (180, 180, 180))

        # Error or extra info
        if error:
            self.text_renderer.render(f"Error: {error[:80]}", 10, self.height - 55, (255, 100, 100))
        elif extra_info:
            self.text_renderer.render(extra_info, 10, self.height - 55, (200, 200, 200))

        # File list overlay
        if self.show_file_list:
            self.draw_file_list()

    def draw_file_list(self):
        """Draw the file list overlay"""
        # Background
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPushMatrix()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Semi-transparent background
        glColor4f(0.1, 0.1, 0.1, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(320, 0)
        glVertex2f(320, self.height)
        glVertex2f(0, self.height)
        glEnd()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

        # Title
        ext_name = self.extension.upper().replace('.', '')
        self.text_renderer.render(f"{ext_name} Files (scroll/arrows, click to select):", 10, self.height - 70, (255, 255, 100))

        # File list
        visible_files = 20
        y_start = self.height - 100
        for i in range(visible_files):
            idx = self.list_scroll_offset + i
            if idx >= len(self.files):
                break
            name = os.path.basename(self.files[idx])
            if len(name) > 35:
                name = name[:32] + "..."
            if idx == self.current_index:
                color = (100, 255, 100)
                prefix = "> "
            else:
                color = (220, 220, 220)
                prefix = "  "
            self.text_renderer.render(f"{prefix}{idx + 1}. {name}", 10, y_start - i * 22, color)

        # Scroll indicator
        if len(self.files) > visible_files:
            scroll_info = f"Showing {self.list_scroll_offset + 1}-{min(self.list_scroll_offset + visible_files, len(self.files))} of {len(self.files)}"
            self.text_renderer.render(scroll_info, 10, 35, (150, 150, 150))
