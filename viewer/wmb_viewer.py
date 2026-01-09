import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from wmb_loader import WMB
from viewer_utils import FileNavigator
import sys
import os
import math


class WMBViewer:
    def __init__(self, folder=None):
        self.width = 1024
        self.height = 768
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("WMB Viewer")

        # File navigation
        if folder is None:
            folder = os.getcwd()
        self.file_nav = FileNavigator(folder, '.wmb', self.width, self.height)

        if not self.file_nav.files:
            print(f"No WMB files found in {folder}")
            sys.exit(1)

        # Setup OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Setup Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, (self.width / self.height), 1.0, 50000.0)
        glMatrixMode(GL_MODELVIEW)

        # WMB data
        self.wmb = None
        self.texture_ids = []
        self.lightmap_ids = []
        self.triangulated_faces = []
        self.load_error = None

        # Camera state
        self.camera_pos = [0.0, 0.0, 100.0]
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.move_speed = 100.0
        self.mouse_sensitivity = 0.2

        # Mouse capture state
        self.mouse_captured = True
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        # Wireframe mode
        self.wireframe = False
        self.show_lightmaps = True

        # Load first file
        self.load_current_file()

    def load_current_file(self):
        """Load the WMB file at current index"""
        # Clean up previous textures
        self.cleanup_textures()

        filename = self.file_nav.current_file
        print(f"\nLoading: {os.path.basename(filename)} ({self.file_nav.current_index + 1}/{self.file_nav.file_count})")

        self.wmb = WMB()
        self.load_error = None
        self.triangulated_faces = []

        try:
            self.wmb.load(filename)
            self.texture_ids = self.load_textures()
            self.lightmap_ids = self.load_lightmaps()
            self.camera_pos = self.calculate_start_position()

            # Pre-triangulate WMB4/WMB6 faces
            if self.wmb.version in [b'WMB4', b'WMB6']:
                self.triangulated_faces = self.triangulate_faces()

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            self.load_error = str(e)
            self.texture_ids = []
            self.lightmap_ids = []
            self.camera_pos = [0.0, 0.0, 100.0]

        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        pygame.display.set_caption(f"WMB Viewer - {os.path.basename(filename)}")

    def cleanup_textures(self):
        """Delete OpenGL textures"""
        for tex_id in self.texture_ids:
            if tex_id is not None:
                glDeleteTextures([tex_id])
        for lm_id in self.lightmap_ids:
            if lm_id is not None:
                glDeleteTextures([lm_id])
        self.texture_ids = []
        self.lightmap_ids = []

    def calculate_start_position(self):
        """Calculate a good starting camera position"""
        # Check for POSITION objects as spawn points
        for obj in self.wmb.objects:
            if obj.get('name') == 'POSITION':
                pos = list(obj['origin'])
                pos[2] += 50  # Raise camera a bit
                return pos

        # Find world bounds from vertices or blocks
        if self.wmb.version in [b'WMB4', b'WMB6'] and self.wmb.vertices:
            min_x = min(v[0] for v in self.wmb.vertices)
            max_x = max(v[0] for v in self.wmb.vertices)
            min_y = min(v[1] for v in self.wmb.vertices)
            max_y = max(v[1] for v in self.wmb.vertices)
            min_z = min(v[2] for v in self.wmb.vertices)
            max_z = max(v[2] for v in self.wmb.vertices)
        elif self.wmb.blocks:
            min_x = min(b['mins'][0] for b in self.wmb.blocks)
            max_x = max(b['maxs'][0] for b in self.wmb.blocks)
            min_y = min(b['mins'][1] for b in self.wmb.blocks)
            max_y = max(b['maxs'][1] for b in self.wmb.blocks)
            min_z = min(b['mins'][2] for b in self.wmb.blocks)
            max_z = max(b['maxs'][2] for b in self.wmb.blocks)
        else:
            return [0.0, 0.0, 100.0]

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2 + 100

        print(f"World bounds: ({min_x:.1f}, {min_y:.1f}, {min_z:.1f}) to ({max_x:.1f}, {max_y:.1f}, {max_z:.1f})")
        print(f"Starting at: ({center_x:.1f}, {center_y:.1f}, {center_z:.1f})")

        return [center_x, center_y, center_z]

    def triangulate_faces(self):
        """Convert polygon faces to triangles using fan triangulation"""
        triangles = []

        for face in self.wmb.faces:
            verts = face['vertices']
            if len(verts) < 3:
                continue

            # Get texinfo for this face
            texinfo_idx = face['tex_idx']
            if texinfo_idx < len(self.wmb.texinfo):
                texinfo = self.wmb.texinfo[texinfo_idx]
                actual_tex = texinfo['texture']
                s_vec = texinfo['s_vec']
                s_off = texinfo['s_off']
                t_vec = texinfo['t_vec']
                t_off = texinfo['t_off']
            else:
                actual_tex = 0
                s_vec = (1, 0, 0)
                s_off = 0
                t_vec = (0, 1, 0)
                t_off = 0

            # Get actual vertex positions and calculate UVs
            positions = []
            uvs = []
            for v_idx in verts:
                if v_idx < len(self.wmb.vertices):
                    pos = self.wmb.vertices[v_idx]
                    positions.append(pos)
                    # Calculate UV using texinfo vectors (Quake-style)
                    u = pos[0] * s_vec[0] + pos[1] * s_vec[1] + pos[2] * s_vec[2] + s_off
                    v = pos[0] * t_vec[0] + pos[1] * t_vec[1] + pos[2] * t_vec[2] + t_off
                    uvs.append((u, v))
                else:
                    positions.append((0, 0, 0))
                    uvs.append((0, 0))

            # Fan triangulation from first vertex
            for i in range(1, len(positions) - 1):
                triangles.append({
                    'vertices': [positions[0], positions[i], positions[i + 1]],
                    'uvs': [uvs[0], uvs[i], uvs[i + 1]],
                    'texture': actual_tex,
                    'flags': face['flags']
                })

        print(f"Triangulated {len(self.wmb.faces)} faces into {len(triangles)} triangles")
        return triangles

    def load_textures(self):
        """Load all textures into OpenGL"""
        texture_ids = []

        for i, tex in enumerate(self.wmb.textures):
            if tex['data'] is None or tex['format'] == 'unknown' or tex['format'] == 'dds':
                texture_ids.append(None)
                continue

            width = tex['width']
            height = tex['height']

            if tex['format'] == 'rgb565':
                arr = np.frombuffer(tex['data'], dtype=np.uint16)
                r = ((arr >> 11) & 0x1F) * 255 // 31
                g = ((arr >> 5) & 0x3F) * 255 // 63
                b = (arr & 0x1F) * 255 // 31
                rgb = np.dstack((r, g, b)).astype(np.uint8).flatten()
                texture_data = rgb.tobytes()
                fmt = GL_RGB

            elif tex['format'] == 'rgba8888':
                arr = np.frombuffer(tex['data'], dtype=np.uint8).reshape(-1, 4)
                rgba = arr[:, [2, 1, 0, 3]].flatten()
                texture_data = rgba.tobytes()
                fmt = GL_RGBA

            elif tex['format'] == 'rgb888':
                arr = np.frombuffer(tex['data'], dtype=np.uint8).reshape(-1, 3)
                rgb = arr[:, ::-1].flatten()
                texture_data = rgb.tobytes()
                fmt = GL_RGB

            elif tex['format'] == 'palette8':
                # For 8-bit, create a simple colorful mapping for visibility
                arr = np.frombuffer(tex['data'], dtype=np.uint8).astype(np.uint16)
                r = (arr).astype(np.uint8)
                g = ((arr * 2) % 256).astype(np.uint8)
                b = ((arr * 3) % 256).astype(np.uint8)
                rgb = np.dstack((r, g, b)).flatten()
                texture_data = rgb.tobytes()
                fmt = GL_RGB

            else:
                texture_ids.append(None)
                continue

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)

            if fmt == GL_RGBA:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            else:
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glGenerateMipmap(GL_TEXTURE_2D)

            texture_ids.append(tex_id)

        print(f"Loaded {len([t for t in texture_ids if t is not None])} textures into OpenGL")
        return texture_ids

    def load_lightmaps(self):
        """Load all lightmaps into OpenGL"""
        lightmap_ids = []

        for i, lm in enumerate(self.wmb.lightmaps):
            width = lm['width']
            height = lm['height']

            arr = np.frombuffer(lm['data'], dtype=np.uint8).reshape(-1, 3)
            rgb = arr[:, ::-1].flatten()
            texture_data = rgb.tobytes()

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

            lightmap_ids.append(tex_id)

        print(f"Loaded {len(lightmap_ids)} lightmaps into OpenGL")
        return lightmap_ids

    def set_mouse_capture(self, captured):
        """Enable or disable mouse capture"""
        self.mouse_captured = captured
        pygame.mouse.set_visible(not captured)
        pygame.event.set_grab(captured)

    def handle_input(self, dt):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Handle file navigation
            action, data = self.file_nav.handle_event(event)
            if action == 'switch':
                self.load_current_file()
                continue
            elif action == 'toggle_list':
                if data:  # List opened
                    self.set_mouse_capture(False)
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.file_nav.show_file_list:
                        self.file_nav.show_file_list = False
                        self.set_mouse_capture(True)
                    elif not self.mouse_captured:
                        self.set_mouse_capture(True)
                    else:
                        pygame.quit()
                        sys.exit()
                elif event.key == pygame.K_TAB:
                    # Toggle mouse capture
                    self.set_mouse_capture(not self.mouse_captured)
                elif event.key == pygame.K_F1:
                    self.wireframe = not self.wireframe
                    print(f"Wireframe: {self.wireframe}")
                elif event.key == pygame.K_F2:
                    self.show_lightmaps = not self.show_lightmaps
                    print(f"Lightmaps: {self.show_lightmaps}")
                elif event.key == pygame.K_F3:
                    # Toggle culling
                    if glIsEnabled(GL_CULL_FACE):
                        glDisable(GL_CULL_FACE)
                        print("Culling: OFF")
                    else:
                        glEnable(GL_CULL_FACE)
                        print("Culling: ON")
            elif event.type == pygame.MOUSEMOTION and self.mouse_captured:
                dx, dy = event.rel
                self.camera_yaw += dx * self.mouse_sensitivity
                self.camera_pitch -= dy * self.mouse_sensitivity
                self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.mouse_captured:
                # Click to recapture mouse (unless on file list)
                if not self.file_nav.show_file_list or event.pos[0] > 320:
                    self.set_mouse_capture(True)

        # Only process movement when mouse is captured
        if not self.mouse_captured:
            return

        # Keyboard movement
        keys = pygame.key.get_pressed()
        speed = self.move_speed * dt

        if keys[pygame.K_LSHIFT]:
            speed *= 3.0

        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)

        forward = [
            math.cos(pitch_rad) * math.sin(yaw_rad),
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad)
        ]
        right = [
            math.cos(yaw_rad),
            -math.sin(yaw_rad),
            0
        ]

        if keys[pygame.K_w]:
            self.camera_pos[0] += forward[0] * speed
            self.camera_pos[1] += forward[1] * speed
            self.camera_pos[2] += forward[2] * speed
        if keys[pygame.K_s]:
            self.camera_pos[0] -= forward[0] * speed
            self.camera_pos[1] -= forward[1] * speed
            self.camera_pos[2] -= forward[2] * speed
        if keys[pygame.K_a]:
            self.camera_pos[0] -= right[0] * speed
            self.camera_pos[1] -= right[1] * speed
        if keys[pygame.K_d]:
            self.camera_pos[0] += right[0] * speed
            self.camera_pos[1] += right[1] * speed
        if keys[pygame.K_SPACE]:
            self.camera_pos[2] += speed
        if keys[pygame.K_LCTRL]:
            self.camera_pos[2] -= speed

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.4, 0.6, 0.8, 1.0)  # Sky blue

        glLoadIdentity()

        # Calculate look direction from yaw and pitch
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)

        # Forward vector (where camera is looking)
        look_x = math.cos(pitch_rad) * math.sin(yaw_rad)
        look_y = math.cos(pitch_rad) * math.cos(yaw_rad)
        look_z = math.sin(pitch_rad)

        # Camera position and look-at point
        cx, cy, cz = self.camera_pos
        gluLookAt(cx, cy, cz,
                  cx + look_x, cy + look_y, cz + look_z,
                  0, 0, 1)  # Z is up

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_TEXTURE_2D)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw based on version
        if self.wmb and self.wmb.version in [b'WMB4', b'WMB6']:
            self.draw_wmb6()
        elif self.wmb:
            self.draw_wmb7()

        # Draw overlay
        self.draw_overlay()

        pygame.display.flip()

    def draw_overlay(self):
        """Draw file info and controls overlay"""
        # Build extra info string
        extra_info = None
        if self.wmb and not self.load_error:
            version = self.wmb.version.decode() if self.wmb.version else 'Unknown'
            if self.wmb.version in [b'WMB4', b'WMB6']:
                extra_info = f"{version} | Faces: {len(self.wmb.faces)} | Textures: {len(self.wmb.textures)}"
            else:
                extra_info = f"{version} | Blocks: {len(self.wmb.blocks)} | Textures: {len(self.wmb.textures)}"

        controls = "WASD: move | Tab: release mouse | L: file list | F1: wireframe"
        if not self.mouse_captured:
            controls = "Click to capture mouse | " + controls

        self.file_nav.draw_overlay(
            extra_info=extra_info,
            error=self.load_error,
            controls_hint=controls
        )

    def draw_wmb6(self):
        """Draw WMB6 level geometry"""
        if not self.triangulated_faces:
            return

        # Group triangles by texture for efficiency
        tex_groups = {}
        for tri in self.triangulated_faces:
            tex_idx = tri['texture']
            if tex_idx not in tex_groups:
                tex_groups[tex_idx] = []
            tex_groups[tex_idx].append(tri)

        # Draw each texture group
        for tex_idx, tris in tex_groups.items():
            # Bind texture
            has_texture = False
            if not self.wireframe and tex_idx < len(self.texture_ids):
                tex_id = self.texture_ids[tex_idx]
                if tex_id is not None:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                    has_texture = True

            if not has_texture and not self.wireframe:
                glDisable(GL_TEXTURE_2D)

            # Set color based on texture for visibility
            if not has_texture:
                # Use texture index for color variety
                r = ((tex_idx * 37) % 200 + 55) / 255.0
                g = ((tex_idx * 71) % 200 + 55) / 255.0
                b = ((tex_idx * 113) % 200 + 55) / 255.0
                glColor3f(r, g, b)
            else:
                glColor3f(1.0, 1.0, 1.0)

            # Get texture dimensions for UV scaling
            tex_width = 64
            tex_height = 64
            if has_texture and tex_idx < len(self.wmb.textures):
                tex_width = self.wmb.textures[tex_idx]['width']
                tex_height = self.wmb.textures[tex_idx]['height']

            glBegin(GL_TRIANGLES)
            for tri in tris:
                verts = tri['vertices']
                uvs = tri['uvs']
                for i, v in enumerate(verts):
                    if has_texture:
                        # Use pre-calculated UVs, scaled by texture dimensions
                        u = uvs[i][0] / tex_width
                        vt = uvs[i][1] / tex_height
                        glTexCoord2f(u, vt)
                    glVertex3f(v[0], v[1], v[2])
            glEnd()

    def draw_wmb7(self):
        """Draw WMB7 blocks"""
        for block in self.wmb.blocks:
            self.draw_block(block)

    def draw_block(self, block):
        """Draw a single WMB7 block"""
        vertices = block['vertices']
        triangles = block['triangles']
        skins = block['skins']

        if not triangles:
            return

        skin_tris = {}
        for tri in triangles:
            skin_idx = tri['skin']
            if skin_idx not in skin_tris:
                skin_tris[skin_idx] = []
            skin_tris[skin_idx].append(tri)

        for skin_idx, tris in skin_tris.items():
            if skin_idx < 0 or skin_idx >= len(skins):
                continue

            skin = skins[skin_idx]
            tex_idx = skin['texture']
            flags = skin['flags']

            is_sky = (flags & 2) != 0

            if is_sky:
                glColor3f(0.5, 0.7, 1.0)
            else:
                glColor3f(1.0, 1.0, 1.0)

            has_texture = False
            if not self.wireframe and tex_idx >= 0 and tex_idx < len(self.texture_ids):
                tex_id = self.texture_ids[tex_idx]
                if tex_id is not None:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                    has_texture = True

            if not has_texture and not self.wireframe:
                glDisable(GL_TEXTURE_2D)

            glBegin(GL_TRIANGLES)
            for tri in tris:
                i1, i2, i3 = tri['indices']

                for idx in [i1, i2, i3]:
                    if idx < 0 or idx >= len(vertices):
                        continue

                    vert = vertices[idx]

                    if has_texture:
                        glTexCoord2f(vert['uv'][0], vert['uv'][1])

                    glVertex3f(vert['pos'][0], vert['pos'][1], vert['pos'][2])

            glEnd()

    def run(self):
        clock = pygame.time.Clock()
        print("\nControls:")
        print("  WASD - Move")
        print("  Mouse - Look")
        print("  Space/Ctrl - Up/Down")
        print("  Shift - Fast move")
        print("  Tab - Release/capture mouse")
        print("  Left/Right - Previous/next file")
        print("  L - Toggle file list")
        print("  F1 - Toggle wireframe")
        print("  F2 - Toggle lightmaps")
        print("  F3 - Toggle culling")
        print("  ESC - Quit (or release mouse)")

        while True:
            dt = clock.tick(60) / 1000.0
            self.handle_input(dt)
            self.draw()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    viewer = WMBViewer(folder)
    viewer.run()
