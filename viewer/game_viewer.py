import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from wmb_loader import WMB
from mdl_loader import MDL
from viewer_utils import FileNavigator
import sys
import os
import math


class ModelInstance:
    """Represents a loaded model with its position, rotation, scale and texture"""
    def __init__(self):
        self.mdl = None
        self.texture_id = None
        self.origin = (0, 0, 0)
        self.angle = (0, 0, 0)
        self.scale = (1, 1, 1)
        self.filename = ""
        self.load_error = None

        # Pre-built geometry for fast rendering
        self.texcoords = None      # numpy array of UV coords (static)
        self.frame_verts = []      # list of numpy arrays, one per frame
        self.vertex_count = 0      # number of vertices (triangles * 3)
        self.interpolated_verts = None  # reusable buffer for interpolated positions
        self.min_z = 0.0           # minimum Z of model (for ground placement)


class GameViewer:
    def __init__(self, folder=None):
        self.width = 1024
        self.height = 768
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Game Viewer")

        # File navigation
        if folder is None:
            folder = os.getcwd()
        self.folder = folder
        self.file_nav = FileNavigator(folder, '.wmb', self.width, self.height)

        if not self.file_nav.files:
            print(f"No WMB files found in {folder}")
            sys.exit(1)

        # Setup OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glCullFace(GL_BACK)

        # Setup Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, (self.width / self.height), 1.0, 80000.0)
        glMatrixMode(GL_MODELVIEW)

        # WMB data
        self.wmb = None
        self.texture_ids = []
        self.triangulated_faces = []
        self.render_batches = []
        self.load_error = None

        # Model instances
        self.models = []

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

        # Animation
        self.frame_index = 0.0
        self.animation_speed = 10  # frames per second
        self.last_time = pygame.time.get_ticks()

        # Load first file
        self.load_current_file()

    def load_current_file(self):
        """Load the WMB file and its models"""
        # Clean up previous data
        self.cleanup_textures()
        self.cleanup_models()

        filename = self.file_nav.current_file
        print(f"\nLoading: {os.path.basename(filename)} ({self.file_nav.current_index + 1}/{self.file_nav.file_count})")

        self.wmb = WMB()
        self.load_error = None
        self.triangulated_faces = []
        self.render_batches = []

        try:
            self.wmb.load(filename)
            self.texture_ids = self.load_world_textures()
            self.camera_pos = self.calculate_start_position()

            # Pre-triangulate WMB4/WMB6 faces and build render batches
            if self.wmb.version in [b'WMB4', b'WMB6']:
                self.triangulated_faces = self.triangulate_faces()
                self.render_batches = self.build_render_batches()

            # Load models from entities
            self.load_entity_models()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading {filename}: {e}")
            self.load_error = str(e)
            self.texture_ids = []
            self.camera_pos = [0.0, 0.0, 100.0]

        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.frame_index = 0.0

        # Adjust movement speed for large levels
        base_name = os.path.basename(filename).lower()
        if base_name in ['ciudad.wmb', 'gruta.wmb']:
            self.move_speed = 1000.0  # 10x faster for large levels
        else:
            self.move_speed = 100.0

        pygame.display.set_caption(f"Game Viewer - {os.path.basename(filename)}")

    def cleanup_textures(self):
        """Delete OpenGL textures for world"""
        for tex_id in self.texture_ids:
            if tex_id is not None:
                glDeleteTextures([tex_id])
        self.texture_ids = []

    def cleanup_models(self):
        """Delete OpenGL textures for models"""
        for model in self.models:
            if model.texture_id is not None:
                glDeleteTextures([model.texture_id])
        self.models = []

    def load_entity_models(self):
        """Load MDL models referenced by entities in the WMB"""
        wmb_dir = os.path.dirname(self.file_nav.current_file)

        for obj in self.wmb.objects:
            if obj.get('name') in ['ENTITY', 'OLD_ENTITY']:
                filename = obj.get('filename', '')
                if not filename:
                    continue

                # Try to find the MDL file
                mdl_path = self.find_mdl_file(filename, wmb_dir)
                if not mdl_path:
                    print(f"  Could not find MDL: {filename}")
                    continue

                model = ModelInstance()
                model.filename = filename
                model.origin = obj.get('origin', (0, 0, 0))
                model.angle = obj.get('angle', (0, 0, 0))
                # Handle scale - default to (1,1,1) if zero or not set
                scale = obj.get('scale', (1, 1, 1))
                if scale[0] == 0 and scale[1] == 0 and scale[2] == 0:
                    scale = (1, 1, 1)
                model.scale = scale

                try:
                    model.mdl = MDL()
                    model.mdl.load(mdl_path)
                    model.texture_id = self.load_model_texture(model.mdl)
                    self.build_model_geometry(model)
                    print(f"  Loaded model: {filename}")
                    print(f"    pos={model.origin}, angle={model.angle}, scale={model.scale}")
                    print(f"    {model.vertex_count // 3} tris, {len(model.frame_verts)} frames, min_z={model.min_z:.2f}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                    model.load_error = str(e)

                self.models.append(model)

        print(f"Loaded {len(self.models)} entity models")

    def find_mdl_file(self, filename, wmb_dir):
        """Try to find the MDL file in various locations"""
        # Clean up the filename
        filename = filename.strip()
        if not filename.lower().endswith('.mdl'):
            filename += '.mdl'

        # Try different paths
        search_paths = [
            os.path.join(wmb_dir, filename),
            os.path.join(wmb_dir, os.path.basename(filename)),
            os.path.join(self.folder, filename),
            os.path.join(self.folder, os.path.basename(filename)),
        ]

        # Also try models subdirectory
        search_paths.append(os.path.join(wmb_dir, 'models', filename))
        search_paths.append(os.path.join(self.folder, 'models', filename))

        for path in search_paths:
            if os.path.exists(path):
                return path

        return None

    def load_model_texture(self, mdl):
        """Load texture for an MDL model"""
        if not mdl.skins:
            return None

        skin = mdl.skins[0]

        # Get width/height
        if 'width' in skin:
            width = skin['width']
            height = skin['height']
        else:
            width = mdl.header['skinwidth']
            height = mdl.header['skinheight']

        skin_type = skin['type']

        if skin_type in ['single_16bit', 'single_16bit_565']:
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint16)
            r = ((arr >> 11) & 0x1F) * 255 // 31
            g = ((arr >> 5) & 0x3F) * 255 // 63
            b = (arr & 0x1F) * 255 // 31
            rgb = np.dstack((r, g, b)).astype(np.uint8).flatten()
            texture_data = rgb.tobytes()

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id

        elif skin_type == 'single_16bit_4444':
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint16)
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
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
            rgb = arr[:, ::-1].flatten()
            texture_data = rgb.tobytes()

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id

        elif skin_type == 'single_32bit_8888':
            data = skin['data']
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
            rgba = arr[:, [2, 1, 0, 3]].flatten()
            texture_data = rgba.tobytes()

            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            return tex_id

        return None

    def build_model_geometry(self, model):
        """Pre-build vertex arrays for fast model rendering"""
        mdl = model.mdl
        if not mdl.triangles or not mdl.frames:
            return

        # Get skin dimensions
        if mdl.skins and 'width' in mdl.skins[0]:
            skin_width = mdl.skins[0]['width']
            skin_height = mdl.skins[0]['height']
        else:
            skin_width = mdl.header.get('skinwidth', 64)
            skin_height = mdl.header.get('skinheight', 64)

        ident = mdl.header['ident']
        is_idpo = ident == b'IDPO'
        is_mdl7 = ident == b'MDL7'

        num_tris = len(mdl.triangles)
        num_verts = num_tris * 3

        # Build UV array (static - same for all frames)
        texcoords = np.empty((num_verts, 2), dtype=np.float32)

        # Build index mapping for vertex lookups
        # For each output vertex, store which source vertex index to use
        vert_indices = np.empty(num_verts, dtype=np.int32)

        idx = 0
        for tri in mdl.triangles:
            if is_idpo:
                facesfront = tri[0]
                tri_vert_indices = tri[1:4]

                for v_idx in tri_vert_indices:
                    vert_indices[idx] = v_idx

                    if v_idx < len(mdl.texcoords):
                        st = mdl.texcoords[v_idx]
                        onseam = st[0]
                        s = st[1]
                        t = st[2]

                        if not facesfront and onseam:
                            s += skin_width // 2

                        texcoords[idx, 0] = s / skin_width
                        texcoords[idx, 1] = t / skin_height
                    else:
                        texcoords[idx] = (0, 0)

                    idx += 1

            elif is_mdl7:
                tri_vert_indices = tri[0:3]
                uv_indices = tri[3:6]

                for i in range(3):
                    v_idx = tri_vert_indices[i]
                    uv_idx = uv_indices[i]
                    vert_indices[idx] = v_idx

                    if uv_idx < len(mdl.skinverts):
                        st = mdl.skinverts[uv_idx]
                        texcoords[idx, 0] = st[0]
                        texcoords[idx, 1] = st[1]
                    else:
                        texcoords[idx] = (0, 0)

                    idx += 1

            else:
                # MDL3/4/5
                tri_vert_indices = tri[0:3]
                uv_indices = tri[3:6]

                for i in range(3):
                    v_idx = tri_vert_indices[i]
                    uv_idx = uv_indices[i]
                    vert_indices[idx] = v_idx

                    if uv_idx < len(mdl.skinverts):
                        st = mdl.skinverts[uv_idx]
                        texcoords[idx, 0] = st[0] / skin_width
                        texcoords[idx, 1] = st[1] / skin_height
                    else:
                        texcoords[idx] = (0, 0)

                    idx += 1

        # Build per-frame vertex position arrays
        frame_verts = []
        for frame in mdl.frames:
            src_verts = frame['verts']
            positions = np.empty((num_verts, 3), dtype=np.float32)

            for i in range(num_verts):
                v_idx = vert_indices[i]
                if v_idx < len(src_verts):
                    v = src_verts[v_idx]
                    positions[i] = (v[0], v[1], v[2])
                else:
                    positions[i] = (0, 0, 0)

            frame_verts.append(positions)

        model.texcoords = texcoords
        model.frame_verts = frame_verts
        model.vertex_count = num_verts
        model.interpolated_verts = np.empty((num_verts, 3), dtype=np.float32)

        # Calculate model's minimum Z for ground placement offset
        # Use first frame to determine bounding box
        if frame_verts:
            model.min_z = float(np.min(frame_verts[0][:, 2]))

    def calculate_start_position(self):
        """Calculate a good starting camera position"""
        # Check for POSITION objects as spawn points
        for obj in self.wmb.objects:
            if obj.get('name') == 'POSITION':
                pos = list(obj['origin'])
                pos[2] += 50
                return pos

        # Find world bounds
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

        return [center_x, center_y, center_z]

    def triangulate_faces(self):
        """Convert polygon faces to triangles"""
        triangles = []

        for face in self.wmb.faces:
            verts = face['vertices']
            if len(verts) < 3:
                continue

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

            positions = []
            uvs = []
            for v_idx in verts:
                if v_idx < len(self.wmb.vertices):
                    pos = self.wmb.vertices[v_idx]
                    positions.append(pos)
                    u = pos[0] * s_vec[0] + pos[1] * s_vec[1] + pos[2] * s_vec[2] + s_off
                    v = pos[0] * t_vec[0] + pos[1] * t_vec[1] + pos[2] * t_vec[2] + t_off
                    uvs.append((u, v))
                else:
                    positions.append((0, 0, 0))
                    uvs.append((0, 0))

            for i in range(1, len(positions) - 1):
                triangles.append({
                    'vertices': [positions[0], positions[i], positions[i + 1]],
                    'uvs': [uvs[0], uvs[i], uvs[i + 1]],
                    'texture': actual_tex,
                    'flags': face['flags']
                })

        return triangles

    def build_render_batches(self):
        """Pre-build render batches grouped by texture"""
        tex_groups = {}
        for tri in self.triangulated_faces:
            tex_idx = tri['texture']
            if tex_idx not in tex_groups:
                tex_groups[tex_idx] = []
            tex_groups[tex_idx].append(tri)

        batches = []
        for tex_idx, tris in tex_groups.items():
            tex_width = 64
            tex_height = 64
            if tex_idx < len(self.wmb.textures):
                tex_width = self.wmb.textures[tex_idx]['width']
                tex_height = self.wmb.textures[tex_idx]['height']

            num_verts = len(tris) * 3
            positions = np.empty((num_verts, 3), dtype=np.float32)
            texcoords = np.empty((num_verts, 2), dtype=np.float32)

            idx = 0
            for tri in tris:
                tri_verts = tri['vertices']
                tri_uvs = tri['uvs']
                for i in range(3):
                    v = tri_verts[i]
                    uv = tri_uvs[i]
                    positions[idx] = (v[0], v[1], v[2])
                    texcoords[idx] = (uv[0] / tex_width, uv[1] / tex_height)
                    idx += 1

            batches.append({
                'texture_idx': tex_idx,
                'positions': positions,
                'texcoords': texcoords,
                'vertex_count': num_verts
            })

        return batches

    def load_world_textures(self):
        """Load all world textures into OpenGL"""
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

        return texture_ids

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
                if data:
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
                    self.set_mouse_capture(not self.mouse_captured)
                elif event.key == pygame.K_F1:
                    self.wireframe = not self.wireframe
                    print(f"Wireframe: {self.wireframe}")
                elif event.key == pygame.K_F3:
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
                if not self.file_nav.show_file_list or event.pos[0] > 320:
                    self.set_mouse_capture(True)

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
        glClearColor(0.4, 0.6, 0.8, 1.0)

        glLoadIdentity()

        # Calculate look direction
        yaw_rad = math.radians(self.camera_yaw)
        pitch_rad = math.radians(self.camera_pitch)

        look_x = math.cos(pitch_rad) * math.sin(yaw_rad)
        look_y = math.cos(pitch_rad) * math.cos(yaw_rad)
        look_z = math.sin(pitch_rad)

        cx, cy, cz = self.camera_pos
        gluLookAt(cx, cy, cz,
                  cx + look_x, cy + look_y, cz + look_z,
                  0, 0, 1)

        if self.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_TEXTURE_2D)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Draw world geometry
        if self.wmb and self.wmb.version in [b'WMB4', b'WMB6']:
            self.draw_wmb6()
        elif self.wmb:
            self.draw_wmb7()

        # Draw entity models
        self.draw_models()

        # Draw overlay
        self.draw_overlay()

        pygame.display.flip()

    def draw_wmb6(self):
        """Draw WMB6 level geometry using vertex arrays"""
        if not self.render_batches:
            return

        glEnableClientState(GL_VERTEX_ARRAY)

        for batch in self.render_batches:
            tex_idx = batch['texture_idx']
            positions = batch['positions']
            texcoords = batch['texcoords']
            vertex_count = batch['vertex_count']

            has_texture = False
            if not self.wireframe and tex_idx < len(self.texture_ids):
                tex_id = self.texture_ids[tex_idx]
                if tex_id is not None:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                    has_texture = True

            if not has_texture and not self.wireframe:
                glDisable(GL_TEXTURE_2D)

            if not has_texture:
                r = ((tex_idx * 37) % 200 + 55) / 255.0
                g = ((tex_idx * 71) % 200 + 55) / 255.0
                b = ((tex_idx * 113) % 200 + 55) / 255.0
                glColor3f(r, g, b)
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            else:
                glColor3f(1.0, 1.0, 1.0)
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2, GL_FLOAT, 0, texcoords)

            glVertexPointer(3, GL_FLOAT, 0, positions)
            glDrawArrays(GL_TRIANGLES, 0, vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

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

    def draw_models(self):
        """Draw all entity models"""
        # Temporarily change cull face for models (MDL uses front-face culling)
        glCullFace(GL_FRONT)

        for model in self.models:
            if model.mdl is None or not model.mdl.frames:
                continue

            self.draw_model(model)

        # Restore cull face for world
        glCullFace(GL_BACK)

    def draw_model(self, model):
        """Draw a single model instance with position, rotation, and scale using vertex arrays"""
        if model.vertex_count == 0 or not model.frame_verts:
            return

        glPushMatrix()

        # Apply transform: translate, then rotate, then scale
        ox, oy, oz = model.origin
        glTranslatef(ox, oy, oz)

        # Rotation (pan, tilt, roll) - Acknex/Gamestudio convention
        pan, tilt, roll = model.angle
        glRotatef(pan, 0, 0, 1)    # Pan around Z
        glRotatef(tilt, 0, 1, 0)   # Tilt around Y
        glRotatef(roll, 1, 0, 0)   # Roll around X

        # Scale
        sx, sy, sz = model.scale
        glScalef(sx, sy, sz)

        # Offset model so its bottom (min_z) sits at the entity origin
        # This makes models "stand on" their position rather than being centered on it
        glTranslatef(0, 0, -model.min_z)

        # Bind texture
        has_texture = False
        if model.texture_id and not self.wireframe:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, model.texture_id)
            glColor3f(1.0, 1.0, 1.0)
            has_texture = True
        else:
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.8, 0.6, 0.4)

        # Calculate frame interpolation
        num_frames = len(model.frame_verts)
        frame_idx_1 = int(self.frame_index) % num_frames
        frame_idx_2 = (frame_idx_1 + 1) % num_frames
        alpha = self.frame_index - int(self.frame_index)

        # Interpolate vertex positions using numpy (fast vectorized operation)
        verts1 = model.frame_verts[frame_idx_1]
        verts2 = model.frame_verts[frame_idx_2]

        # Use pre-allocated buffer for interpolated vertices
        np.multiply(verts1, 1.0 - alpha, out=model.interpolated_verts)
        model.interpolated_verts += verts2 * alpha

        # Draw using vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, model.interpolated_verts)

        if has_texture and model.texcoords is not None:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, 0, model.texcoords)

        glDrawArrays(GL_TRIANGLES, 0, model.vertex_count)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

        glPopMatrix()

    def draw_overlay(self):
        """Draw file info and controls overlay"""
        extra_info = None
        if self.wmb and not self.load_error:
            version = self.wmb.version.decode() if self.wmb.version else 'Unknown'
            model_count = len(self.models)
            if self.wmb.version in [b'WMB4', b'WMB6']:
                extra_info = f"{version} | Faces: {len(self.wmb.faces)} | Models: {model_count}"
            else:
                extra_info = f"{version} | Blocks: {len(self.wmb.blocks)} | Models: {model_count}"

        controls = "WASD: move | Tab: release mouse | L: file list | F1: wireframe"
        if not self.mouse_captured:
            controls = "Click to capture mouse | " + controls

        self.file_nav.draw_overlay(
            extra_info=extra_info,
            error=self.load_error,
            controls_hint=controls
        )

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
        print("  F3 - Toggle culling")
        print("  ESC - Quit (or release mouse)")

        while True:
            dt = clock.tick(60) / 1000.0

            # Update animation
            current_time = pygame.time.get_ticks()
            anim_dt = (current_time - self.last_time) / 1000.0
            self.last_time = current_time

            # Find max frames across all models
            max_frames = 1
            for model in self.models:
                if model.mdl and model.mdl.frames:
                    max_frames = max(max_frames, len(model.mdl.frames))

            self.frame_index += self.animation_speed * anim_dt
            if self.frame_index >= max_frames:
                self.frame_index = 0

            self.handle_input(dt)
            self.draw()


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    viewer = GameViewer(folder)
    viewer.run()
