import struct
import numpy as np

class MDL:
    def __init__(self):
        self.header = {}
        self.skins = []
        self.texcoords = []
        self.triangles = []
        self.frames = []

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = f.read()

        offset = 0

        # Header
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

        if self.header['ident'] != b'IDPO':
            print("Not a valid MDL file")
            return

        # Skins
        for _ in range(self.header['num_skins']):
            # Try to peek at the group/type
            group = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            print(f"Skin group/type: {group}")
            
            if group == 0:
                width = self.header['skinwidth']
                height = self.header['skinheight']
                size = width * height
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_8bit', 'data': skin_data})
            elif group == 2:
                # Assuming type 2 is 16-bit skin
                width = self.header['skinwidth']
                height = self.header['skinheight']
                size = width * height * 2
                skin_data = data[offset:offset+size]
                offset += size
                self.skins.append({'type': 'single_16bit', 'data': skin_data})
            else:
                # Group skin
                print(f"Group skin detected. Type: {group}")
                nb = struct.unpack_from('<I', data, offset)[0]
                offset += 4
                print(f"Number of skins in group: {nb}")
                
                # Read times
                times = struct.unpack_from(f'<{nb}f', data, offset)
                offset += nb * 4
                
                width = self.header['skinwidth']
                height = self.header['skinheight']
                size = width * height
                
                group_skins = []
                for i in range(nb):
                    skin_data = data[offset:offset+size]
                    offset += size
                    group_skins.append(skin_data)
                
                self.skins.append({'type': 'group', 'times': times, 'data': group_skins})

        # Texture Coords
        # onseam (I), s (I), t (I)
        st_fmt = '<III'
        st_size = struct.calcsize(st_fmt)
        for _ in range(self.header['num_verts']):
            st = struct.unpack_from(st_fmt, data, offset)
            offset += st_size
            self.texcoords.append(st)

        # Triangles
        # facesfront (I), vertindex (3I)
        tri_fmt = '<I3I'
        tri_size = struct.calcsize(tri_fmt)
        for _ in range(self.header['num_tris']):
            tri = struct.unpack_from(tri_fmt, data, offset)
            offset += tri_size
            self.triangles.append(tri)

        # Frames
        for _ in range(self.header['num_frames']):
            type = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            if type == 0:
                # Simple frame
                # bboxmin (trivertx_t), bboxmax (trivertx_t), name (16s)
                # trivertx_t is 3B (v) + 1B (lightnormalindex)
                frame_header_fmt = '<4B 4B 16s'
                frame_header_size = struct.calcsize(frame_header_fmt)
                fh = struct.unpack_from(frame_header_fmt, data, offset)
                offset += frame_header_size
                
                bboxmin = fh[0:4]
                bboxmax = fh[4:8]
                name = fh[8].strip(b'\x00').decode('utf-8', errors='ignore')
                
                verts = []
                vert_fmt = '<4B' # v[3], lightnormalindex
                vert_size = struct.calcsize(vert_fmt)
                
                # Read all verts for this frame
                # It's faster to read them all at once
                num_verts = self.header['num_verts']
                verts_data = data[offset : offset + num_verts * vert_size]
                offset += num_verts * vert_size
                
                # Parse verts
                # We can use numpy or struct iter
                # Let's just store the raw bytes for now or parse them
                # Parsing them is better for the viewer
                
                # To convert compressed vert to real coord:
                # v = scale * packed + translate
                
                frame_verts = []
                for i in range(num_verts):
                    v_packed = verts_data[i*4 : i*4+4]
                    # v_packed is 4 bytes: x, y, z, light
                    # x, y, z are unsigned bytes
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
                return

        print(f"Loaded MDL: {filename}")
        print(f"Skins: {len(self.skins)}")
        print(f"Frames: {len(self.frames)}")


