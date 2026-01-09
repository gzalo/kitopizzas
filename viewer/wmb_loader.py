import struct
import numpy as np

class WMB:
    def __init__(self):
        self.header = {}
        self.textures = []
        self.materials = []
        self.blocks = []      # WMB7 blocks
        self.objects = []
        self.lightmaps = []
        self.info = None
        # WMB6 specific
        self.vertices = []    # Raw vertex positions
        self.edges = []       # Edge list (v1, v2) pairs
        self.surfedges = []   # Face edge references (1-indexed signed)
        self.faces = []       # Face data
        self.texinfo = []     # Texture info (UV mapping + texture index)
        self.version = None

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = f.read()

        # Read version identifier
        self.version = struct.unpack_from('<4s', data, 0)[0]
        print(f"WMB version: {self.version}")

        if self.version == b'WMB7':
            self._load_wmb7(data)
        elif self.version == b'WMB6':
            self._load_wmb6(data)
        elif self.version == b'WMB4':
            self._load_wmb4(data)
        else:
            print(f"Unknown WMB version: {self.version}")

    def _load_wmb4(self, data):
        """Load WMB4 format (older A6 engine)"""
        # WMB4 header has 16 lists
        offset = 4
        list_names = [
            'list0', 'palettes', 'textures', 'vertices', 'pvs',
            'bsp_nodes', 'materials', 'faces', 'list8', 'aabb_hulls',
            'bsp_leafs', 'bsp_blocks', 'edges', 'skins', 'info',
            'objects'
        ]

        self.header['version'] = self.version
        self.header['lists'] = {}

        for name in list_names:
            list_offset, list_length = struct.unpack_from('<II', data, offset)
            self.header['lists'][name] = {'offset': list_offset, 'length': list_length}
            offset += 8
            if list_length > 0 and list_offset < len(data):
                print(f"  {name}: offset={list_offset}, length={list_length}")

        # Load textures (same format as WMB6)
        self._load_textures(data)

        # Load texinfo (materials list contains texinfo)
        self._load_wmb6_texinfo(data)

        # Load vertices
        self._load_wmb6_vertices(data)

        # Load edges
        self._load_wmb6_edges(data)

        # Load faces
        self._load_wmb6_faces(data)

        # Load objects
        self._load_objects(data)

        print(f"Loaded WMB4: textures={len(self.textures)}, verts={len(self.vertices)}, faces={len(self.faces)}")

    def _load_wmb6(self, data):
        """Load WMB6 format (A6 engine)"""
        # WMB6 header has 17 lists instead of 20
        offset = 4
        list_names = [
            'list0', 'palettes', 'textures', 'vertices', 'pvs',
            'bsp_nodes', 'materials', 'faces', 'legacy4', 'aabb_hulls',
            'bsp_leafs', 'bsp_blocks', 'edges', 'skins', 'legacy7',
            'objects', 'lightmaps'
        ]

        self.header['version'] = self.version
        self.header['lists'] = {}

        for name in list_names:
            list_offset, list_length = struct.unpack_from('<II', data, offset)
            self.header['lists'][name] = {'offset': list_offset, 'length': list_length}
            offset += 8
            if list_length > 0 and list_offset < len(data):
                print(f"  {name}: offset={list_offset}, length={list_length}")

        # Load textures
        self._load_textures(data)

        # Load texinfo (materials list in WMB6 contains texinfo, not material names)
        self._load_wmb6_texinfo(data)

        # Load vertices (legacy2 in WMB7 terms = 'vertices' here)
        self._load_wmb6_vertices(data)

        # Load edges
        self._load_wmb6_edges(data)

        # Load faces
        self._load_wmb6_faces(data)

        # Load objects
        self._load_objects(data)

        print(f"Loaded WMB6: textures={len(self.textures)}, verts={len(self.vertices)}, faces={len(self.faces)}")

    def _load_wmb6_texinfo(self, data):
        """Load WMB6 texinfo (UV mapping vectors + texture index)"""
        # In WMB6, the materials list contains texinfo structs
        mat_list = self.header['lists']['materials']
        if mat_list['length'] == 0:
            return

        offset = mat_list['offset']
        # Each texinfo is 64 bytes:
        # s_vec[3], s_offset, t_vec[3], t_offset, texture_idx, flags, padding
        num_texinfo = mat_list['length'] // 64

        print(f"Loading {num_texinfo} texinfo entries...")
        for i in range(num_texinfo):
            s_vec = struct.unpack_from('<3f', data, offset)
            s_off = struct.unpack_from('<f', data, offset + 12)[0]
            t_vec = struct.unpack_from('<3f', data, offset + 16)
            t_off = struct.unpack_from('<f', data, offset + 28)[0]
            tex_idx, flags = struct.unpack_from('<2I', data, offset + 32)

            self.texinfo.append({
                's_vec': s_vec,
                's_off': s_off,
                't_vec': t_vec,
                't_off': t_off,
                'texture': tex_idx,
                'flags': flags
            })
            offset += 64

    def _load_wmb6_vertices(self, data):
        """Load WMB6 vertices (3 floats per vertex)"""
        vert_list = self.header['lists']['vertices']
        if vert_list['length'] == 0:
            return

        offset = vert_list['offset']
        num_verts = vert_list['length'] // 12  # 3 floats * 4 bytes

        print(f"Loading {num_verts} vertices...")
        for i in range(num_verts):
            x, y, z = struct.unpack_from('<3f', data, offset)
            self.vertices.append((x, y, z))
            offset += 12

    def _load_wmb6_edges(self, data):
        """Load WMB6 edges (pairs of vertex indices)"""
        edge_list = self.header['lists']['edges']
        if edge_list['length'] == 0:
            return

        offset = edge_list['offset'] + 8  # Skip 8-byte header
        num_edges = (edge_list['length'] - 8) // 8  # 2 ints per edge

        print(f"Loading {num_edges} edges...")
        for i in range(num_edges):
            v1, v2 = struct.unpack_from('<2I', data, offset)
            self.edges.append((v1, v2))
            offset += 8

    def _load_wmb6_surfedges(self, data):
        """Load WMB6 surfedges (face edge references, 1-indexed signed ints)"""
        # In WMB6, 'skins' list is actually surfedges
        surfedge_list = self.header['lists']['skins']
        if surfedge_list['length'] == 0:
            return

        offset = surfedge_list['offset']
        num_surfedges = surfedge_list['length'] // 4

        print(f"Loading {num_surfedges} surfedges...")
        self.surfedges = []
        for i in range(num_surfedges):
            val = struct.unpack_from('<i', data, offset)[0]
            self.surfedges.append(val)
            offset += 4

    def _load_wmb6_faces(self, data):
        """Load WMB6 faces"""
        face_list = self.header['lists']['faces']
        if face_list['length'] == 0:
            return

        # Load surfedges first (needed for face construction)
        self._load_wmb6_surfedges(data)

        offset = face_list['offset']
        num_faces = face_list['length'] // 24  # 24 bytes per face

        print(f"Loading {num_faces} faces...")
        for i in range(num_faces):
            vals = struct.unpack_from('<6I', data, offset)
            flags = vals[0]
            first_surfedge = vals[1]
            tex_info = vals[2]
            skin = vals[5]

            # tex_info low 16 bits = num_verts
            num_verts = tex_info & 0xFFFF
            tex_idx = (tex_info >> 16) & 0xFFFF

            # Build vertex list using surfedges
            # Surfedges are 1-indexed signed ints pointing to edges
            face_verts = []
            for e in range(num_verts):
                surfedge_idx = first_surfedge + e
                if surfedge_idx >= len(self.surfedges):
                    continue

                surfedge = self.surfedges[surfedge_idx]
                # Convert 1-indexed to 0-indexed
                edge_idx = abs(surfedge) - 1
                if edge_idx < 0 or edge_idx >= len(self.edges):
                    continue

                v1, v2 = self.edges[edge_idx]
                # Negative surfedge means reverse edge direction
                if surfedge < 0:
                    v1, v2 = v2, v1

                if len(face_verts) == 0:
                    face_verts.append(v1)
                face_verts.append(v2)

            self.faces.append({
                'flags': flags,
                'first_edge': first_surfedge,
                'num_verts': num_verts,
                'tex_idx': tex_idx,
                'skin': skin,
                'vertices': face_verts
            })
            offset += 24

    def _load_wmb7(self, data):
        """Load WMB7 format"""
        offset = 4
        list_names = [
            'palettes', 'legacy1', 'textures', 'legacy2', 'pvs',
            'bsp_nodes', 'materials', 'legacy3', 'legacy4', 'aabb_hulls',
            'bsp_leafs', 'bsp_blocks', 'legacy5', 'legacy6', 'legacy7',
            'objects', 'lightmaps', 'blocks', 'legacy8', 'lightmaps_terrain'
        ]

        self.header['version'] = self.version
        self.header['lists'] = {}

        for name in list_names:
            list_offset, list_length = struct.unpack_from('<II', data, offset)
            self.header['lists'][name] = {'offset': list_offset, 'length': list_length}
            offset += 8
            if list_length > 0 and list_offset < len(data):
                print(f"  {name}: offset={list_offset}, length={list_length}")

        self._load_textures(data)
        self._load_materials(data)
        self._load_lightmaps(data)
        self._load_blocks(data)
        self._load_objects(data)

        print(f"Loaded WMB7: textures={len(self.textures)}, blocks={len(self.blocks)}")

    def _load_textures(self, data):
        tex_list = self.header['lists']['textures']
        if tex_list['length'] == 0:
            return

        offset = tex_list['offset']
        num_textures = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        print(f"Loading {num_textures} textures...")

        tex_offsets = []
        for i in range(num_textures):
            tex_off = struct.unpack_from('<I', data, offset)[0]
            tex_offsets.append(tex_off)
            offset += 4

        for i, tex_off in enumerate(tex_offsets):
            abs_offset = tex_list['offset'] + tex_off

            name = struct.unpack_from('<16s', data, abs_offset)[0]
            name = name.strip(b'\x00').decode('utf-8', errors='ignore')
            width, height, tex_type = struct.unpack_from('<III', data, abs_offset + 16)
            abs_offset += 16 + 4 * 4  # Skip header (40 bytes total)

            texture = {
                'name': name,
                'width': width,
                'height': height,
                'type': tex_type,
                'data': None
            }

            # WMB4/WMB6 uses different type encoding than WMB7
            if self.version in [b'WMB4', b'WMB6']:
                # WMB6: type 40 = RGB565 with mipmaps, type 8 = RGB565 no mipmaps
                # type 48 = RGB888 with mipmaps, etc.
                if tex_type in [40, 8, 2]:  # RGB565
                    texture['format'] = 'rgb565'
                    size = width * height * 2
                    texture['data'] = data[abs_offset:abs_offset + size]
                elif tex_type in [48, 16, 4]:  # RGB888
                    texture['format'] = 'rgb888'
                    size = width * height * 3
                    texture['data'] = data[abs_offset:abs_offset + size]
                elif tex_type in [56, 24, 5]:  # RGBA8888
                    texture['format'] = 'rgba8888'
                    size = width * height * 4
                    texture['data'] = data[abs_offset:abs_offset + size]
                else:
                    # Try to detect from legacy field (size info)
                    legacy = struct.unpack_from('<I', data, abs_offset - 12)[0]
                    expected_16bit = width * height * 2 + 40
                    if legacy == expected_16bit or abs(legacy - expected_16bit) < 100:
                        texture['format'] = 'rgb565'
                        size = width * height * 2
                        texture['data'] = data[abs_offset:abs_offset + size]
                    else:
                        texture['format'] = 'unknown'
            else:
                # WMB7 format
                has_mipmaps = (tex_type & 8) != 0
                base_type = tex_type & 7

                if base_type == 6:
                    dds_size = width
                    texture['format'] = 'dds'
                    texture['data'] = data[abs_offset:abs_offset + dds_size]
                elif base_type == 5:
                    texture['format'] = 'rgba8888'
                    size = width * height * 4
                    texture['data'] = data[abs_offset:abs_offset + size]
                elif base_type == 4:
                    texture['format'] = 'rgb888'
                    size = width * height * 3
                    texture['data'] = data[abs_offset:abs_offset + size]
                elif base_type == 2:
                    texture['format'] = 'rgb565'
                    size = width * height * 2
                    texture['data'] = data[abs_offset:abs_offset + size]
                elif base_type == 0:
                    texture['format'] = 'palette8'
                    size = width * height
                    texture['data'] = data[abs_offset:abs_offset + size]
                else:
                    texture['format'] = 'unknown'

            self.textures.append(texture)
            print(f"  Texture {i}: '{name}' {width}x{height} {texture['format']}")

    def _load_materials(self, data):
        mat_list = self.header['lists']['materials']
        if mat_list['length'] == 0:
            return

        num_materials = mat_list['length'] // 64
        offset = mat_list['offset']

        for i in range(num_materials):
            mat_name = struct.unpack_from('<20s', data, offset + 44)[0]
            mat_name = mat_name.strip(b'\x00').decode('utf-8', errors='ignore')
            self.materials.append(mat_name)
            offset += 64

    def _load_lightmaps(self, data):
        lm_list = self.header['lists']['lightmaps']
        if lm_list['length'] == 0:
            return

        lm_size = 1024 * 1024 * 3
        num_lightmaps = lm_list['length'] // lm_size
        offset = lm_list['offset']

        print(f"Loading {num_lightmaps} lightmaps...")
        for i in range(num_lightmaps):
            lm_data = data[offset:offset + lm_size]
            self.lightmaps.append({
                'width': 1024,
                'height': 1024,
                'data': lm_data
            })
            offset += lm_size

    def _load_blocks(self, data):
        blk_list = self.header['lists'].get('blocks')
        if not blk_list or blk_list['length'] == 0 or blk_list['offset'] >= len(data):
            return

        offset = blk_list['offset']
        num_blocks = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        print(f"Loading {num_blocks} blocks...")

        for blk_idx in range(num_blocks):
            block_fmt = '<6f4I'
            block_size = struct.calcsize(block_fmt)
            block_data = struct.unpack_from(block_fmt, data, offset)
            offset += block_size

            block = {
                'mins': block_data[0:3],
                'maxs': block_data[3:6],
                'content': block_data[6],
                'num_verts': block_data[7],
                'num_tris': block_data[8],
                'num_skins': block_data[9],
                'vertices': [],
                'triangles': [],
                'skins': []
            }

            for v_idx in range(block['num_verts']):
                vx, vy, vz, tu, tv, su, sv = struct.unpack_from('<7f', data, offset)
                offset += 28
                block['vertices'].append({
                    'pos': (vx, vy, vz),
                    'uv': (tu, tv),
                    'lm_uv': (su, sv)
                })

            for t_idx in range(block['num_tris']):
                v1, v2, v3, skin, unused = struct.unpack_from('<4hI', data, offset)
                offset += 12
                block['triangles'].append({
                    'indices': (v1, v2, v3),
                    'skin': skin
                })

            for s_idx in range(block['num_skins']):
                texture, lightmap, material = struct.unpack_from('<hhI', data, offset)
                ambient, albedo = struct.unpack_from('<ff', data, offset + 8)
                flags = struct.unpack_from('<I', data, offset + 16)[0]
                offset += 20
                block['skins'].append({
                    'texture': texture,
                    'lightmap': lightmap,
                    'material': material,
                    'ambient': ambient,
                    'albedo': albedo,
                    'flags': flags
                })

            self.blocks.append(block)

    def _load_objects(self, data):
        obj_list = self.header['lists']['objects']
        if obj_list['length'] == 0:
            return

        offset = obj_list['offset']
        num_objects = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        print(f"Loading {num_objects} objects...")

        obj_offsets = []
        for i in range(num_objects):
            obj_off = struct.unpack_from('<I', data, offset)[0]
            obj_offsets.append(obj_off)
            offset += 4

        for i, obj_off in enumerate(obj_offsets):
            abs_offset = obj_list['offset'] + obj_off
            obj_type = struct.unpack_from('<I', data, abs_offset)[0]

            obj = {'type': obj_type, 'index': i}

            if obj_type == 5:  # WMB_INFO
                obj['name'] = 'INFO'
                obj['origin'] = struct.unpack_from('<3f', data, abs_offset + 4)
                obj['azimuth'], obj['elevation'] = struct.unpack_from('<2f', data, abs_offset + 16)
                self.info = obj

            elif obj_type == 1:  # WMB_POSITION
                obj['name'] = 'POSITION'
                obj['origin'] = struct.unpack_from('<3f', data, abs_offset + 4)
                obj['angle'] = struct.unpack_from('<3f', data, abs_offset + 16)
                obj['pos_name'] = struct.unpack_from('<20s', data, abs_offset + 36)[0].strip(b'\x00').decode('utf-8', errors='ignore')

            elif obj_type == 2:  # WMB_LIGHT
                obj['name'] = 'LIGHT'
                obj['origin'] = struct.unpack_from('<3f', data, abs_offset + 4)
                obj['color'] = struct.unpack_from('<3f', data, abs_offset + 16)
                obj['range'] = struct.unpack_from('<f', data, abs_offset + 28)[0]

            elif obj_type == 3:  # WMB_OLD_ENTITY
                obj['name'] = 'OLD_ENTITY'
                obj['origin'] = struct.unpack_from('<3f', data, abs_offset + 4)
                obj['angle'] = struct.unpack_from('<3f', data, abs_offset + 16)
                obj['scale'] = struct.unpack_from('<3f', data, abs_offset + 28)
                obj['ent_name'] = struct.unpack_from('<20s', data, abs_offset + 40)[0].strip(b'\x00').decode('utf-8', errors='ignore')
                obj['filename'] = struct.unpack_from('<13s', data, abs_offset + 60)[0].strip(b'\x00').decode('utf-8', errors='ignore')

            elif obj_type == 7:  # WMB_ENTITY
                obj['name'] = 'ENTITY'
                obj['origin'] = struct.unpack_from('<3f', data, abs_offset + 4)
                obj['angle'] = struct.unpack_from('<3f', data, abs_offset + 16)
                obj['scale'] = struct.unpack_from('<3f', data, abs_offset + 28)
                obj['ent_name'] = struct.unpack_from('<33s', data, abs_offset + 40)[0].strip(b'\x00').decode('utf-8', errors='ignore')
                obj['filename'] = struct.unpack_from('<33s', data, abs_offset + 73)[0].strip(b'\x00').decode('utf-8', errors='ignore')

            else:
                obj['name'] = f'TYPE_{obj_type}'

            self.objects.append(obj)
