import struct
import numpy as np

class MDL:
    def __init__(self):
        self.header = {}
        self.skins = []
        self.texcoords = []      # For IDPO: list of (onseam, s, t)
        self.skinverts = []      # For MDL3/4/5: list of (u, v) skin vertices
        self.triangles = []
        self.frames = []

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = f.read()

        # First, read just the ident to determine the format
        ident = struct.unpack_from('<4s', data, 0)[0]
        
        if ident == b'IDPO':
            self._load_idpo(data)
        elif ident in [b'MDL3', b'MDL4', b'MDL5']:
            self._load_mdl345(data)
        elif ident == b'MDL7':
            self._load_mdl7(data)
        else:
            print(f"Not a valid MDL file. Found identifier: {ident}")
            return
    
    def _load_idpo(self, data):
        """Load Quake MDL format (IDPO, version 6)"""
        offset = 0
        
        # Header for IDPO format
        header_fmt = '<4sI 3f 3f f 3f I I I I I I I I f'
        header_size = struct.calcsize(header_fmt)
        unpacked = struct.unpack_from(header_fmt, data, offset)
        offset += header_size

        self.header = {
            'ident': unpacked[0],
            'version': unpacked[1],
            'scale': unpacked[2:5],
            'translate': unpacked[5:8],
            'boundingradius': unpacked[8],
            'eyeposition': unpacked[9:12],
            'num_skins': unpacked[12],
            'skinwidth': unpacked[13],
            'skinheight': unpacked[14],
            'num_verts': unpacked[15],
            'num_tris': unpacked[16],
            'num_frames': unpacked[17],
            'synctype': unpacked[18],
            'flags': unpacked[19],
            'size': unpacked[20]
        }

        print(f"Loaded MDL format: IDPO (version {self.header['version']})")

        # Skins
        for _ in range(self.header['num_skins']):
            skin_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            print(f"Skin type: {skin_type}")

            width = self.header['skinwidth']
            height = self.header['skinheight']

            if skin_type == 0:
                # Single 8-bit palettized
                size = width * height
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_8bit', 'data': skin_data})
            elif skin_type == 2:
                # Single 16-bit 565 RGB
                size = width * height * 2
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_16bit', 'data': skin_data})
            elif skin_type == 3:
                # Single 16-bit 4444 ARGB
                size = width * height * 2
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_16bit_4444', 'data': skin_data})
            elif skin_type == 4:
                # Single 24-bit 888 RGB
                size = width * height * 3
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_24bit_888', 'data': skin_data})
            elif skin_type == 5:
                # Single 32-bit 8888 ARGB
                size = width * height * 4
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_32bit_8888', 'data': skin_data})
            elif skin_type == 1:
                # Group of 8-bit skins
                print(f"Group skin detected (8-bit)")
                nb = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                print(f"Number of skins in group: {nb}")

                times = struct.unpack_from(f'<{nb}f', data, offset)
                offset += nb * 4

                size = width * height
                group_skins = []
                for i in range(nb):
                    skin_data = data[offset:offset+size]
                    offset += size
                    group_skins.append(skin_data)

                self.skins.append({'type': 'group', 'times': times, 'data': group_skins})
            else:
                print(f"Unknown skin type: {skin_type}, skipping skin")

        # Texture Coords: onseam (I), s (I), t (I)
        st_fmt = '<III'
        st_size = struct.calcsize(st_fmt)
        for _ in range(self.header['num_verts']):
            st = struct.unpack_from(st_fmt, data, offset)
            offset += st_size
            self.texcoords.append(st)

        # Triangles: facesfront (I), vertindex (3I)
        tri_fmt = '<I3I'
        tri_size = struct.calcsize(tri_fmt)
        for _ in range(self.header['num_tris']):
            tri = struct.unpack_from(tri_fmt, data, offset)
            offset += tri_size
            self.triangles.append(tri)

        # Frames
        offset = self._load_frames_idpo(data, offset)

        print(f"Loaded MDL: skins={len(self.skins)}, frames={len(self.frames)}")
    
    def _load_frames_idpo(self, data, offset):
        """Load frames for IDPO format"""
        for _ in range(self.header['num_frames']):
            frame_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            if frame_type == 0:
                # Simple frame
                frame_header_fmt = '<4B 4B 16s'
                frame_header_size = struct.calcsize(frame_header_fmt)
                fh = struct.unpack_from(frame_header_fmt, data, offset)
                offset += frame_header_size
                
                bboxmin = fh[0:4]
                bboxmax = fh[4:8]
                name = fh[8].strip(b'\x00').decode('utf-8', errors='ignore')
                
                num_verts = self.header['num_verts']
                vert_size = 4
                verts_data = data[offset : offset + num_verts * vert_size]
                offset += num_verts * vert_size
                
                frame_verts = []
                for i in range(num_verts):
                    v_packed = verts_data[i*4 : i*4+4]
                    vx = v_packed[0] * self.header['scale'][0] + self.header['translate'][0]
                    vy = v_packed[1] * self.header['scale'][1] + self.header['translate'][1]
                    vz = v_packed[2] * self.header['scale'][2] + self.header['translate'][2]
                    frame_verts.append((vx, vy, vz))
                
                self.frames.append({
                    'type': 'single',
                    'name': name,
                    'bboxmin': bboxmin,
                    'bboxmax': bboxmax,
                    'verts': frame_verts
                })
            else:
                print("Group frame not implemented")
                return offset
        return offset

    def _load_mdl345(self, data):
        """Load Gamestudio MDL3/MDL4/MDL5 format (Conitec/Acknex engine)"""
        offset = 0
        
        # MDL3/4/5 header (84 bytes = 0x54):
        # version[4], unused1, scale[3], offset[3], unused2, unused3[3],
        # numskins, skinwidth, skinheight, numverts, numtris, numframes,
        # numskinverts, flags, unused4
        header_fmt = '<4s I 3f 3f I 3f I I I I I I I I I'
        header_size = struct.calcsize(header_fmt)  # Should be 84 bytes
        unpacked = struct.unpack_from(header_fmt, data, offset)
        offset += header_size

        self.header = {
            'ident': unpacked[0],
            'version': 0,  # Not used in MDL3/4/5
            'scale': unpacked[2:5],
            'translate': unpacked[5:8],  # Called 'offset' in Conitec docs
            'boundingradius': 0.0,
            'eyeposition': (0, 0, 0),
            'num_skins': unpacked[12],
            'skinwidth': unpacked[13],
            'skinheight': unpacked[14],
            'num_verts': unpacked[15],
            'num_tris': unpacked[16],
            'num_frames': unpacked[17],
            'num_skinverts': unpacked[18],  # Separate from num_verts!
            'synctype': 0,
            'flags': unpacked[19],
            'size': 0.0
        }

        print(f"Loaded MDL format: {self.header['ident'].decode('latin-1')}")
        print(f"  Scale: {self.header['scale']}")
        print(f"  Translate: {self.header['translate']}")
        print(f"  Verts: {self.header['num_verts']}, SkinVerts: {self.header['num_skinverts']}, Tris: {self.header['num_tris']}, Frames: {self.header['num_frames']}")

        # Skins - each has a skintype prefix
        for skin_idx in range(self.header['num_skins']):
            skintype = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            # Determine bytes per pixel based on skintype
            # 0 = 8-bit palettized, 2 = 565 RGB (16-bit), 3 = 4444 ARGB (16-bit)
            # +8 = has mipmaps (MDL5 only)
            has_mipmaps = (skintype & 8) != 0
            base_type = skintype & 7
            
            if base_type == 0:
                bpp = 1
                skin_type_str = 'single_8bit'
            elif base_type == 2:
                bpp = 2
                skin_type_str = 'single_16bit_565'
            elif base_type == 3:
                bpp = 2
                skin_type_str = 'single_16bit_4444'
            elif base_type == 4:
                bpp = 3
                skin_type_str = 'single_24bit_888'
            elif base_type == 5:
                bpp = 4
                skin_type_str = 'single_32bit_8888'
            else:
                print(f"Unknown skin type: {skintype}")
                bpp = 1
                skin_type_str = 'unknown'
            
            # For MDL5, width/height always come per-skin; for MDL3/4 they're in header
            if self.header['ident'] == b'MDL5':
                # MDL5: width and height always follow skintype
                skin_width, skin_height = struct.unpack_from('<II', data, offset)
                offset += 8
            else:
                skin_width = self.header['skinwidth']
                skin_height = self.header['skinheight']
            
            skin_size = skin_width * skin_height * bpp
            skin_data = data[offset:offset+skin_size]
            offset += skin_size
            
            # Skip mipmaps if present (MDL5 with skintype >= 8)
            if has_mipmaps:
                mip1_size = (skin_width // 2) * (skin_height // 2) * bpp
                mip2_size = (skin_width // 4) * (skin_height // 4) * bpp
                mip3_size = (skin_width // 8) * (skin_height // 8) * bpp
                offset += mip1_size + mip2_size + mip3_size
            
            self.skins.append({
                'type': skin_type_str,
                'width': skin_width,
                'height': skin_height,
                'data': skin_data
            })
            print(f"  Skin {skin_idx}: type={skintype}, {skin_width}x{skin_height}, bpp={bpp}")

        # Skin vertices (UV coords) - numskinverts entries
        # Each is: short u, short v
        for _ in range(self.header['num_skinverts']):
            u, v = struct.unpack_from('<hh', data, offset)
            offset += 4
            self.skinverts.append((u, v))

        # Triangles - each has index_xyz[3] and index_uv[3]
        # 6 shorts = 12 bytes per triangle
        tri_fmt = '<3h3h'
        tri_size = struct.calcsize(tri_fmt)
        for _ in range(self.header['num_tris']):
            tri = struct.unpack_from(tri_fmt, data, offset)
            offset += tri_size
            # Store as (index_xyz[0], index_xyz[1], index_xyz[2], index_uv[0], index_uv[1], index_uv[2])
            self.triangles.append(tri)

        # Frames
        offset = self._load_frames_mdl345(data, offset)

        print(f"Loaded MDL: skins={len(self.skins)}, frames={len(self.frames)}")

    def _load_frames_mdl345(self, data, offset):
        """Load frames for MDL3/MDL4/MDL5 format"""
        for frame_idx in range(self.header['num_frames']):
            # Frame type: 0 = byte-packed positions, 2 = word-packed positions
            frame_type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            
            # bboxmin, bboxmax (trivertx_t = 4 bytes each), name (16 bytes)
            if frame_type == 0:
                # Byte-packed: 3 bytes pos + 1 byte normal
                bbox_fmt = '<4B4B'
            else:
                # Word-packed: 3 shorts pos + 1 byte normal + 1 byte unused = 8 bytes
                bbox_fmt = '<8B8B'
            
            bbox_size = struct.calcsize(bbox_fmt)
            bbox_data = struct.unpack_from(bbox_fmt, data, offset)
            offset += bbox_size
            
            name = struct.unpack_from('<16s', data, offset)[0]
            offset += 16
            name = name.strip(b'\x00').decode('utf-8', errors='ignore')
            
            num_verts = self.header['num_verts']
            frame_verts = []
            
            if frame_type == 0:
                # Byte-packed vertices: 3 bytes + 1 byte normal = 4 bytes each
                vert_size = 4
                verts_data = data[offset : offset + num_verts * vert_size]
                offset += num_verts * vert_size
                
                for i in range(num_verts):
                    v_packed = verts_data[i*4 : i*4+4]
                    vx = v_packed[0] * self.header['scale'][0] + self.header['translate'][0]
                    vy = v_packed[1] * self.header['scale'][1] + self.header['translate'][1]
                    vz = v_packed[2] * self.header['scale'][2] + self.header['translate'][2]
                    frame_verts.append((vx, vy, vz))
            else:
                # Word-packed vertices: 3 shorts + 1 byte normal + 1 byte unused = 8 bytes each
                vert_size = 8
                verts_data = data[offset : offset + num_verts * vert_size]
                offset += num_verts * vert_size
                
                for i in range(num_verts):
                    v_packed = struct.unpack_from('<3HBB', verts_data, i*8)
                    vx = v_packed[0] * self.header['scale'][0] + self.header['translate'][0]
                    vy = v_packed[1] * self.header['scale'][1] + self.header['translate'][1]
                    vz = v_packed[2] * self.header['scale'][2] + self.header['translate'][2]
                    frame_verts.append((vx, vy, vz))
            
            self.frames.append({
                'type': 'single',
                'name': name,
                'verts': frame_verts
            })
        
        return offset

    def _load_mdl7(self, data):
        """Load Gamestudio MDL7 format (Conitec/Acknex engine)"""
        offset = 0
        
        # MD7_HEADER structure (from MDL7.h):
        # char ident[4], long version, long bones_num, long groups_num,
        # long mdl7data_size, long entlump_size, long medlump_size,
        # followed by structure sizes (10 unsigned shorts)
        header_fmt = '<4s i i i i i i 10H'
        header_size = struct.calcsize(header_fmt)
        unpacked = struct.unpack_from(header_fmt, data, offset)
        offset += header_size
        
        self.header = {
            'ident': unpacked[0],
            'version': unpacked[1],
            'bones_num': unpacked[2],
            'groups_num': unpacked[3],
            'mdl7data_size': unpacked[4],
            'entlump_size': unpacked[5],
            'medlump_size': unpacked[6],
            # Structure sizes
            'bone_stc_size': unpacked[7],
            'skin_stc_size': unpacked[8],
            'colorvalue_stc_size': unpacked[9],
            'material_stc_size': unpacked[10],
            'skinpoint_stc_size': unpacked[11],
            'triangle_stc_size': unpacked[12],
            'mainvertex_stc_size': unpacked[13],
            'framevertex_stc_size': unpacked[14],
            'bonetrans_stc_size': unpacked[15],
            'frame_stc_size': unpacked[16],
            # Compatibility fields
            'num_skins': 0,
            'skinwidth': 0,
            'skinheight': 0,
            'num_verts': 0,
            'num_tris': 0,
            'num_frames': 0,
            'scale': (1.0, 1.0, 1.0),
            'translate': (0.0, 0.0, 0.0),
        }
        
        print(f"Loaded MDL format: MDL7 (version {self.header['version']})")
        print(f"  Bones: {self.header['bones_num']}, Groups: {self.header['groups_num']}")
        
        # Read bones (skip for now, but need to advance offset)
        # MD7_BONE: unsigned short parent_index, BYTE[2], float x,y,z, char name[20]
        bone_size = self.header['bone_stc_size']
        offset += self.header['bones_num'] * bone_size
        
        # Read groups - we'll combine all groups into one mesh for simplicity
        all_verts = []
        all_skinverts = []
        all_triangles = []
        vertex_offset = 0
        skinvert_offset = 0
        
        for group_idx in range(self.header['groups_num']):
            # MD7_GROUP structure:
            # unsigned char typ, BYTE deformers, BYTE max_weights, BYTE unused,
            # long groupdata_size, char name[16],
            # long numskins, long num_stpts, long numtris, long numverts, long numframes
            group_fmt = '<B B B B i 16s i i i i i'
            group_size = struct.calcsize(group_fmt)
            group_data = struct.unpack_from(group_fmt, data, offset)
            offset += group_size
            
            group_typ = group_data[0]
            group_datasize = group_data[4]
            group_name = group_data[5].strip(b'\x00').decode('utf-8', errors='ignore')
            num_skins = group_data[6]
            num_stpts = group_data[7]
            num_tris = group_data[8]
            num_verts = group_data[9]
            num_frames = group_data[10]
            
            print(f"  Group {group_idx}: '{group_name}' - skins={num_skins}, stpts={num_stpts}, tris={num_tris}, verts={num_verts}, frames={num_frames}")
            
            # Read skins for this group
            for skin_idx in range(num_skins):
                # MD7_SKIN: unsigned char typ, BYTE[3], long width, long height, char texture_name[16]
                skin_stc_size = self.header['skin_stc_size']
                skin_fmt = '<B 3B i i 16s'
                skin_base_size = struct.calcsize(skin_fmt)
                skin_data = struct.unpack_from(skin_fmt, data, offset)
                offset += skin_stc_size
                
                skin_typ = skin_data[0]
                skin_width = skin_data[4]
                skin_height = skin_data[5]
                skin_name = skin_data[6].strip(b'\x00').decode('utf-8', errors='ignore')
                
                # Determine skin type and size
                has_mipmaps = (skin_typ & 0x08) != 0
                has_material = (skin_typ & 0x10) != 0
                base_type = skin_typ & 0x07
                
                if has_material or base_type == 1:
                    # Material reference, no pixel data
                    print(f"    Skin {skin_idx}: material reference '{skin_name}'")
                    continue
                elif base_type == 6:
                    # DDS file - skip for now
                    print(f"    Skin {skin_idx}: DDS file (not supported)")
                    continue
                elif base_type == 7:
                    # External file reference
                    print(f"    Skin {skin_idx}: external file '{skin_name}'")
                    continue
                
                if base_type == 0:
                    bpp = 1
                    skin_type_str = 'single_8bit'
                elif base_type == 2:
                    bpp = 2
                    skin_type_str = 'single_16bit_565'
                elif base_type == 3:
                    bpp = 2
                    skin_type_str = 'single_16bit_4444'
                elif base_type == 4:
                    bpp = 3
                    skin_type_str = 'single_24bit_888'
                elif base_type == 5:
                    bpp = 4
                    skin_type_str = 'single_32bit_8888'
                else:
                    print(f"    Skin {skin_idx}: unknown type {skin_typ}")
                    bpp = 0
                    skin_type_str = 'unknown'
                
                if bpp > 0 and skin_width > 0 and skin_height > 0:
                    pixel_size = skin_width * skin_height * bpp
                    pixel_data = data[offset:offset+pixel_size]
                    offset += pixel_size
                    
                    # Skip mipmaps if present
                    if has_mipmaps:
                        mip1 = (skin_width // 2) * (skin_height // 2) * bpp
                        mip2 = (skin_width // 4) * (skin_height // 4) * bpp
                        mip3 = (skin_width // 8) * (skin_height // 8) * bpp
                        offset += mip1 + mip2 + mip3
                    
                    # Only store first skin for now
                    if len(self.skins) == 0:
                        self.skins.append({
                            'type': skin_type_str,
                            'width': skin_width,
                            'height': skin_height,
                            'data': pixel_data
                        })
                        self.header['skinwidth'] = skin_width
                        self.header['skinheight'] = skin_height
                        print(f"    Skin {skin_idx}: {skin_type_str} {skin_width}x{skin_height}")
            
            # Read skin points (UV coordinates) - floats 0.0-1.0
            # MD7_SKINPOINT: float s, float t
            skinpoint_stc_size = self.header['skinpoint_stc_size']
            for _ in range(num_stpts):
                s, t = struct.unpack_from('<ff', data, offset)
                offset += skinpoint_stc_size
                # Store as-is (floats 0-1), will handle in viewer
                all_skinverts.append((s, t))
            
            # Read triangles
            # MD7_TRIANGLE: unsigned short v_index[3], then MD7_SKINSET[2]
            # MD7_SKINSET: unsigned short st_index[3], int material
            triangle_stc_size = self.header['triangle_stc_size']
            for _ in range(num_tris):
                # Read vertex indices
                v0, v1, v2 = struct.unpack_from('<3H', data, offset)
                # Read first skinset (UV indices)
                uv0, uv1, uv2 = struct.unpack_from('<3H', data, offset + 6)
                offset += triangle_stc_size
                
                # Adjust indices by vertex/skinvert offset for this group
                all_triangles.append((
                    v0 + vertex_offset,
                    v1 + vertex_offset,
                    v2 + vertex_offset,
                    uv0 + skinvert_offset,
                    uv1 + skinvert_offset,
                    uv2 + skinvert_offset
                ))
            
            # Read main vertices
            # MD7_MAINVERTEX: float x,y,z, unsigned short bone_index, then normal
            mainvertex_stc_size = self.header['mainvertex_stc_size']
            group_verts = []
            for _ in range(num_verts):
                vx, vy, vz = struct.unpack_from('<fff', data, offset)
                offset += mainvertex_stc_size
                group_verts.append((vx, vy, vz))
            
            # Read frames
            # MD7_FRAME: char frame_name[16], long vertices_count, long transmatrix_count
            frame_stc_size = self.header['frame_stc_size']
            framevertex_stc_size = self.header['framevertex_stc_size']
            bonetrans_stc_size = self.header['bonetrans_stc_size']
            
            for frame_idx in range(num_frames):
                frame_name = struct.unpack_from('<16s', data, offset)[0]
                frame_name = frame_name.strip(b'\x00').decode('utf-8', errors='ignore')
                verts_count, trans_count = struct.unpack_from('<ii', data, offset + 16)
                offset += frame_stc_size
                
                # Read frame vertices - these override main vertices
                # MD7_FRAMEVERTEX: float x,y,z, unsigned short vertindex, then normal
                frame_verts = list(group_verts)  # Start with main vertices
                for _ in range(verts_count):
                    fx, fy, fz = struct.unpack_from('<fff', data, offset)
                    vert_idx = struct.unpack_from('<H', data, offset + 12)[0]
                    offset += framevertex_stc_size
                    if vert_idx < len(frame_verts):
                        frame_verts[vert_idx] = (fx, fy, fz)
                
                # Skip bone transforms
                offset += trans_count * bonetrans_stc_size
                
                # For first group, create frame; for others, extend existing frame
                if group_idx == 0:
                    self.frames.append({
                        'type': 'single',
                        'name': frame_name,
                        'verts': frame_verts
                    })
                else:
                    # Extend existing frame's verts
                    if frame_idx < len(self.frames):
                        self.frames[frame_idx]['verts'].extend(frame_verts)
            
            # Update offsets for next group
            vertex_offset += num_verts
            skinvert_offset += num_stpts
        
        # Store combined data
        self.skinverts = all_skinverts
        self.triangles = all_triangles
        self.header['num_verts'] = vertex_offset
        self.header['num_tris'] = len(all_triangles)
        self.header['num_frames'] = len(self.frames)
        self.header['num_skinverts'] = skinvert_offset
        
        print(f"Loaded MDL7: total verts={vertex_offset}, tris={len(all_triangles)}, frames={len(self.frames)}")


