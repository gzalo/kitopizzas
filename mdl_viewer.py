import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from mdl_loader import MDL
import sys
import os
import glob

class MDLViewer:
    def __init__(self, folder=None):
        self.width = 800
        self.height = 600
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("MDL Viewer")
        self.font = pygame.font.SysFont('Arial', 18)
        self.font_large = pygame.font.SysFont('Arial', 24)

        # Find all MDL files in folder
        if folder is None:
            folder = os.getcwd()
        self.folder = folder
        self.mdl_files = sorted(glob.glob(os.path.join(folder, "*.mdl")))
        if not self.mdl_files:
            print(f"No MDL files found in {folder}")
            sys.exit(1)

        self.current_index = 0
        self.show_file_list = False
        self.list_scroll_offset = 0

        # Load first file
        self.mdl = None
        self.texture_id = None

        # Setup OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        # Setup Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 10000.0)
        glMatrixMode(GL_MODELVIEW)

        self.frame_index = 0
        self.last_time = pygame.time.get_ticks()
        self.animation_speed = 10 # frames per second

        self.rotate_x = 0
        self.rotate_y = 0
        self.zoom = 50.0
        self.dragging = False
        self.last_mouse_pos = (0, 0)

        self.load_current_file()

    def load_current_file(self):
        """Load the MDL file at current_index"""
        if self.texture_id:
            glDeleteTextures([self.texture_id])
            self.texture_id = None

        filename = self.mdl_files[self.current_index]
        print(f"\nLoading: {os.path.basename(filename)} ({self.current_index + 1}/{len(self.mdl_files)})")

        self.mdl = MDL()
        self.load_error = None
        try:
            self.mdl.load(filename)
            self.texture_id = self.load_texture()
            self.zoom = self.calculate_auto_zoom()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            self.load_error = str(e)
            self.mdl.frames = [{'verts': [], 'name': 'error'}]
            self.mdl.triangles = []
            self.zoom = 50.0

        self.frame_index = 0
        self.rotate_x = 0
        self.rotate_y = 0
        pygame.display.set_caption(f"MDL Viewer - {os.path.basename(filename)}")

    def switch_file(self, delta):
        """Switch to next/previous file"""
        self.current_index = (self.current_index + delta) % len(self.mdl_files)
        self.load_current_file()

    def switch_to_index(self, index):
        """Switch to file at specific index"""
        if 0 <= index < len(self.mdl_files):
            self.current_index = index
            self.load_current_file()
    
    def calculate_auto_zoom(self):
        """Calculate appropriate zoom level based on model bounding box"""
        if not self.mdl.frames:
            return 50.0
        
        verts = self.mdl.frames[0]['verts']
        if not verts:
            return 50.0
        
        # Calculate bounding box
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for vx, vy, vz in verts:
            min_x = min(min_x, vx)
            min_y = min(min_y, vy)
            min_z = min(min_z, vz)
            max_x = max(max_x, vx)
            max_y = max(max_y, vy)
            max_z = max(max_z, vz)
        
        # Calculate model size
        size_x = max_x - min_x
        size_y = max_y - min_y
        size_z = max_z - min_z
        max_size = max(size_x, size_y, size_z)
        
        # Calculate center offset for later use
        self.model_center = (
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2
        )
        
        print(f"Model bounding box: ({min_x:.2f}, {min_y:.2f}, {min_z:.2f}) to ({max_x:.2f}, {max_y:.2f}, {max_z:.2f})")
        print(f"Model size: {size_x:.2f} x {size_y:.2f} x {size_z:.2f}, max: {max_size:.2f}")
        print(f"Model center: {self.model_center}")
        
        # Set zoom to fit model in view (with some padding)
        zoom = max_size * 2.0
        if zoom < 1.0:
            zoom = 1.0
        
        print(f"Auto zoom: {zoom:.2f}")
        return zoom

    def load_texture(self):
        if not self.mdl.skins:
            return None
            
        skin = self.mdl.skins[0]
        
        # Get width/height - MDL3/4/5 store per-skin, IDPO uses header
        if 'width' in skin:
            width = skin['width']
            height = skin['height']
        else:
            width = self.mdl.header['skinwidth']
            height = self.mdl.header['skinheight']
        
        skin_type = skin['type']
        
        if skin_type in ['single_16bit', 'single_16bit_565']:
            # Convert RGB565 to RGB
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint16)
            
            # RGB565: RRRRR GGGGGG BBBBB
            # R: (x >> 11) & 0x1F
            # G: (x >> 5) & 0x3F
            # B: x & 0x1F
            
            r = ((arr >> 11) & 0x1F) * 255 // 31
            g = ((arr >> 5) & 0x3F) * 255 // 63
            b = (arr & 0x1F) * 255 // 31
            
            # Stack to (N, 3)
            rgb = np.dstack((r, g, b)).astype(np.uint8).flatten()
            texture_data = rgb.tobytes()
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id
        
        elif skin_type == 'single_16bit_4444':
            # Convert ARGB4444 to RGBA
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint16)
            
            # ARGB4444: AAAA RRRR GGGG BBBB
            a = ((arr >> 12) & 0xF) * 255 // 15
            r = ((arr >> 8) & 0xF) * 255 // 15
            g = ((arr >> 4) & 0xF) * 255 // 15
            b = (arr & 0xF) * 255 // 15
            
            rgba = np.dstack((r, g, b, a)).astype(np.uint8).flatten()
            texture_data = rgba.tobytes()
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id
        
        elif skin_type == 'single_24bit_888':
            # 24-bit RGB (BGR in file, Intel byte order)
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            
            # BGR -> RGB
            rgb = arr[:, ::-1].flatten()
            texture_data = rgb.tobytes()
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id
        
        elif skin_type == 'single_32bit_8888':
            # 32-bit ARGB (BGRA in file, Intel byte order: B, G, R, A)
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
            
            # BGRA -> RGBA
            rgba = arr[:, [2, 1, 0, 3]].flatten()
            texture_data = rgba.tobytes()
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id
            
        elif skin_type == 'single_8bit':
            print("8-bit skin not fully supported yet (using grayscale)")
            return None
        
        else:
            print(f"Unsupported skin type: {skin_type}")
            return None

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_file_list:
                        self.show_file_list = False
                    else:
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_LEFT:
                    self.switch_file(-1)
                elif event.key == pygame.K_RIGHT:
                    self.switch_file(1)
                elif event.key == pygame.K_l:
                    self.show_file_list = not self.show_file_list
                    self.list_scroll_offset = max(0, self.current_index - 10)
                elif event.key == pygame.K_UP:
                    if self.show_file_list:
                        self.list_scroll_offset = max(0, self.list_scroll_offset - 1)
                elif event.key == pygame.K_DOWN:
                    if self.show_file_list:
                        max_scroll = max(0, len(self.mdl_files) - 20)
                        self.list_scroll_offset = min(max_scroll, self.list_scroll_offset + 1)
                elif event.key == pygame.K_RETURN:
                    if self.show_file_list:
                        self.show_file_list = False
                elif event.key == pygame.K_HOME:
                    self.switch_to_index(0)
                elif event.key == pygame.K_END:
                    self.switch_to_index(len(self.mdl_files) - 1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    if self.show_file_list:
                        self.list_scroll_offset = max(0, self.list_scroll_offset - 3)
                    else:
                        self.zoom = max(1.0, self.zoom - 2.0)
                elif event.button == 5:  # Scroll down
                    if self.show_file_list:
                        max_scroll = max(0, len(self.mdl_files) - 20)
                        self.list_scroll_offset = min(max_scroll, self.list_scroll_offset + 3)
                    else:
                        self.zoom += 2.0
                elif self.show_file_list and event.button == 1 and event.pos[0] < 300:
                    # Click on file list
                    clicked_idx = self.list_scroll_offset + (event.pos[1] - 100) // 22
                    if 0 <= clicked_idx < len(self.mdl_files):
                        self.switch_to_index(clicked_idx)
                        self.show_file_list = False
                elif event.button == 1:
                    self.dragging = True
                    self.last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.rotate_y += dx * 0.5
                    self.rotate_x += dy * 0.5
                    self.last_mouse_pos = event.pos

    def render_text(self, text, x, y, color=(255, 255, 255), font=None):
        """Render text at screen position (x, y)"""
        if font is None:
            font = self.font
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

    def draw_overlay(self):
        """Draw text overlay with current file info"""
        # Current file name at top
        filename = os.path.basename(self.mdl_files[self.current_index])
        info_text = f"{filename} ({self.current_index + 1}/{len(self.mdl_files)})"
        self.render_text(info_text, 10, self.height - 30, (255, 255, 255), self.font_large)

        # Controls hint at bottom
        controls = "Left/Right: navigate | L: file list | Home/End: first/last"
        self.render_text(controls, 10, 10, (180, 180, 180))

        # Error message if load failed
        if self.load_error:
            self.render_text(f"Error: {self.load_error[:80]}", 10, self.height - 55, (255, 100, 100))
        # Frame info
        elif self.mdl.frames:
            frame_idx = int(self.frame_index) % len(self.mdl.frames)
            frame_name = self.mdl.frames[frame_idx].get('name', '')
            frame_text = f"Frame: {frame_idx + 1}/{len(self.mdl.frames)}"
            if frame_name:
                frame_text += f" ({frame_name})"
            self.render_text(frame_text, 10, self.height - 55, (200, 200, 200))

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
        self.render_text("MDL Files (scroll with mouse/arrows, click to select):", 10, self.height - 70, (255, 255, 100), self.font)

        # File list
        visible_files = 20
        y_start = self.height - 100
        for i in range(visible_files):
            idx = self.list_scroll_offset + i
            if idx >= len(self.mdl_files):
                break
            name = os.path.basename(self.mdl_files[idx])
            if len(name) > 35:
                name = name[:32] + "..."
            if idx == self.current_index:
                color = (100, 255, 100)
                prefix = "> "
            else:
                color = (220, 220, 220)
                prefix = "  "
            self.render_text(f"{prefix}{idx + 1}. {name}", 10, y_start - i * 22, color)

        # Scroll indicator
        if len(self.mdl_files) > visible_files:
            scroll_info = f"Showing {self.list_scroll_offset + 1}-{min(self.list_scroll_offset + visible_files, len(self.mdl_files))} of {len(self.mdl_files)}"
            self.render_text(scroll_info, 10, 35, (150, 150, 150))

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self.zoom)
        glRotatef(-90, 1, 0, 0) # Quake Z-up to OpenGL Y-up
        
        glRotatef(self.rotate_x, 1, 0, 0)
        glRotatef(self.rotate_y, 0, 0, 1) # Rotate around Z (Quake up)
        
        # Center the model
        if hasattr(self, 'model_center'):
            cx, cy, cz = self.model_center
            glTranslatef(-cx, -cy, -cz)
        
        if self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        else:
            glDisable(GL_TEXTURE_2D)
            
        glBegin(GL_TRIANGLES)
        
        # Interpolation
        frame_idx_1 = int(self.frame_index)
        frame_idx_2 = (frame_idx_1 + 1) % len(self.mdl.frames)
        alpha = self.frame_index - frame_idx_1
        
        frame1 = self.mdl.frames[frame_idx_1]
        frame2 = self.mdl.frames[frame_idx_2]
        
        verts1 = frame1['verts']
        verts2 = frame2['verts']
        
        # Get skin dimensions - MDL3/4/5 store per-skin, IDPO uses header
        if self.mdl.skins and 'width' in self.mdl.skins[0]:
            skin_width = self.mdl.skins[0]['width']
            skin_height = self.mdl.skins[0]['height']
        else:
            skin_width = self.mdl.header['skinwidth']
            skin_height = self.mdl.header['skinheight']
        
        # Check format type
        ident = self.mdl.header['ident']
        is_idpo = ident == b'IDPO'
        is_mdl7 = ident == b'MDL7'
        
        for tri in self.mdl.triangles:
            if is_idpo:
                # IDPO format: (facesfront, v0, v1, v2)
                facesfront = tri[0]
                vert_indices = tri[1:4]
                uv_indices = vert_indices  # Same indices for UV
                
                for i, v_idx in enumerate(vert_indices):
                    # Texture coords
                    st = self.mdl.texcoords[v_idx]
                    onseam = st[0]
                    s = st[1]
                    t = st[2]
                    
                    if not facesfront and onseam:
                        s += skin_width // 2
                    
                    u = s / skin_width
                    v = t / skin_height
                    
                    glTexCoord2f(u, v)
                    
                    # Vertex Interpolation
                    v1 = verts1[v_idx]
                    v2 = verts2[v_idx]
                    
                    vx = v1[0] * (1 - alpha) + v2[0] * alpha
                    vy = v1[1] * (1 - alpha) + v2[1] * alpha
                    vz = v1[2] * (1 - alpha) + v2[2] * alpha
                    
                    glVertex3f(vx, vy, vz)
            elif is_mdl7:
                # MDL7 format: (xyz0, xyz1, xyz2, uv0, uv1, uv2)
                # UV coords are already floats 0.0-1.0
                vert_indices = tri[0:3]
                uv_indices = tri[3:6]
                
                for i in range(3):
                    v_idx = vert_indices[i]
                    uv_idx = uv_indices[i]
                    
                    # Texture coords from skinverts - already normalized floats
                    st = self.mdl.skinverts[uv_idx]
                    u = st[0]
                    v = st[1]
                    
                    glTexCoord2f(u, v)
                    
                    # Vertex Interpolation
                    v1 = verts1[v_idx]
                    v2 = verts2[v_idx]
                    
                    vx = v1[0] * (1 - alpha) + v2[0] * alpha
                    vy = v1[1] * (1 - alpha) + v2[1] * alpha
                    vz = v1[2] * (1 - alpha) + v2[2] * alpha
                    
                    glVertex3f(vx, vy, vz)
            else:
                # MDL3/4/5 format: (xyz0, xyz1, xyz2, uv0, uv1, uv2)
                vert_indices = tri[0:3]
                uv_indices = tri[3:6]
                
                for i in range(3):
                    v_idx = vert_indices[i]
                    uv_idx = uv_indices[i]
                    
                    # Texture coords from skinverts
                    st = self.mdl.skinverts[uv_idx]
                    s = st[0]
                    t = st[1]
                    
                    u = s / skin_width
                    v = t / skin_height
                    
                    glTexCoord2f(u, v)
                    
                    # Vertex Interpolation
                    v1 = verts1[v_idx]
                    v2 = verts2[v_idx]
                    
                    vx = v1[0] * (1 - alpha) + v2[0] * alpha
                    vy = v1[1] * (1 - alpha) + v2[1] * alpha
                    vz = v1[2] * (1 - alpha) + v2[2] * alpha
                    
                    glVertex3f(vx, vy, vz)
                
        glEnd()

        # Draw text overlay
        self.draw_overlay()

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            self.handle_input()
            
            # Animation
            current_time = pygame.time.get_ticks()
            dt = (current_time - self.last_time) / 1000.0
            self.frame_index += self.animation_speed * dt
            if self.frame_index >= len(self.mdl.frames):
                self.frame_index = 0
            self.last_time = current_time
            
            self.draw()
            clock.tick(60)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "C:\\Users\\Gzalo\\Desktop\\KitoPizzasVol1"
    viewer = MDLViewer(folder)
    viewer.run()
