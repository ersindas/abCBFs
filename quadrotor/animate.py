"""Animate the saved paper figure."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile

import fitz
import imageio_ffmpeg
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream, FloatObject

from plotting.safety_plot_geometry import saved_barrier_kappa


def serialize(operations, reader):
    stream = ContentStream(None, reader)
    stream.operations = operations
    return stream.get_data()


def marked_parts(operations, reader):
    """Separate exact static drawing operators from tagged dynamic groups."""
    parts, static = [], []
    position = 0
    while position < len(operations):
        args, operator = operations[position]
        if operator == b'BMC' and str(args[0]).startswith('/anim_'):
            if static:
                parts.append((None, serialize(static, reader), None))
                static = []
            name = str(args[0])[1:]
            depth, end = 1, position + 1
            while depth and end < len(operations):
                if operations[end][1] in (b'BMC', b'BDC'):
                    depth += 1
                elif operations[end][1] == b'EMC':
                    depth -= 1
                end += 1
            if depth:
                raise ValueError('Unbalanced paper animation tags.')
            body = operations[position + 1:end - 1]
            parts.append((name, serialize(body, reader), body))
            position = end
        else:
            static.append(operations[position])
            position += 1
    if static:
        parts.append((None, serialize(static, reader), None))
    return parts


def progressive_path(operations, count):
    """Keep the first recorded vertices, retaining original paint operators."""
    result, seen = [], 0
    for arguments, operator in operations:
        if operator in (b'm', b'l'):
            if seen < count:
                result.append((arguments, operator))
            seen += 1
        else:
            result.append((arguments, operator))
    return result


class PaperRenderer:
    def __init__(self, figure, case):
        self.figure = Path(figure)
        self.original = self.figure.read_bytes()
        self.reader = PdfReader(io.BytesIO(self.original))
        text = self.reader.metadata.get('/AnimationSpec') if self.reader.metadata else None
        if text is None:
            raise ValueError('Regenerate the paper PDF with the current plotter before animating it: AnimationSpec is missing.')
        self.spec = json.loads(text)
        if self.spec.get('version') != 1:
            raise ValueError('Unsupported paper animation specification.')
        self.paths = {'adaptive': Path(case) / 'adaptive.npz',
                      'nominal': Path(case) / 'nominal.npz', 'primary': Path(case) / 'primary.npz',
                      'flows': Path(case) / 'backup_flows.npz'}
        self.saved = {}
        for name, path in self.paths.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != self.spec['source_sha256'][name]:
                raise ValueError(f'The paper PDF and animation use different {name} data.')
            with np.load(path, allow_pickle=False) as data:
                self.saved[name] = {key: data[key] for key in data.files}
        self.adaptive = self.saved['adaptive']
        self.time = self.adaptive['t']
        self.kappa = saved_barrier_kappa(*(self.saved[name] for name in ['adaptive', 'nominal', 'primary']))
        if self.kappa != float(self.spec['kappa']):
            raise ValueError('Paper figure kappa does not match the selected simulation.')
        for name in ['nominal', 'primary', 'flows']:
            if not np.array_equal(self.saved[name]['t'], self.time):
                raise ValueError('The selected data do not share controller sample times.')
        if 'kappa' in self.saved['flows']:
            saved_barrier_kappa(self.adaptive, self.saved['flows'])
        resources = self.reader.pages[0]['/Resources']['/XObject']
        self.forms = [resources[outer]['/Resources']['/XObject']['/fullpage']
                      for outer in ['/fzFrm0', '/fzFrm1']]
        self.xrefs = [form.indirect_reference.idnum for form in self.forms]
        self.parts = [marked_parts(ContentStream(form, self.reader).operations, self.reader)
                      for form in self.forms]
        self.names = {name for parts in self.parts for name, _, _ in parts if name is not None}
        required = {'anim_trace_adaptive', 'anim_trace_nominal', 'anim_trace_primary',
                    'anim_input_0', 'anim_input_1', 'anim_drone_0_0'}
        required.update(f'anim_parameter_{i}' for i in range(6))
        if not required <= self.names:
            raise ValueError('Required dynamic paths are missing from the paper PDF.')
        self.pose_times = {index: float(time) for index, time in enumerate(self.spec['glyph_times'])
                           if time is not None}
        pose_parts = {}
        for name in self.names:
            if name.startswith('anim_drone_'):
                _, _, pose, part = name.split('_')
                pose_parts.setdefault(int(pose), set()).add(int(part))
        if (set(pose_parts) != set(self.pose_times) or
                any(parts != set(range(5)) for parts in pose_parts.values())):
            raise ValueError('The final paper PDF is missing a complete drone pose.')
        self.document = fitz.open(stream=self.original, filetype='pdf')
        self.page = self.document[0]
        self.reference = self.pixels()
        self.transform = self.spec['top_transform']
        self.initial_state = self.adaptive['x'][0]

    def pixels(self):
        pixmap = self.page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3).copy()

    def xy(self, points):
        points = np.asarray(points, dtype=float)
        origin, size = np.asarray(self.transform['origin']), np.asarray(self.transform['size'])
        limits = np.asarray([self.transform['xlim'], self.transform['ylim']])
        return origin + (points - limits[:, 0]) * size / (limits[:, 1] - limits[:, 0])

    def drone_part(self, operations, part, index):
        def endpoints(state):
            heading = np.array([np.cos(state[2]), -np.sin(state[2])])
            up = np.array([np.sin(state[2]), np.cos(state[2])])
            ends = np.array([state[:2] - .22 * heading, state[:2] + .22 * heading])
            return ends, ends + .11 * up
        ends, tips = endpoints(self.adaptive['x'][index])
        old_ends, old_tips = endpoints(self.initial_state)
        if part in [0, 1, 3]:
            if part == 0:
                points = self.xy(ends)
            else:
                side = 0 if part == 1 else 1
                points = self.xy([ends[side], tips[side]])
            geometry = [([FloatObject(float(v)) for v in point], b'm' if j == 0 else b'l')
                        for j, point in enumerate(points)]
            return geometry + [(args, op) for args, op in operations if op not in (b'm', b'l')]
        side = 0 if part == 2 else 1
        shift = self.xy(tips[side]) - self.xy(old_tips[side])
        moved = []
        for args, op in operations:
            if op in (b'm', b'l', b'c'):
                points = np.asarray(args, dtype=float).reshape(-1, 2) + shift
                moved.append(([FloatObject(float(v)) for v in points.ravel()], op))
            else:
                moved.append((args, op))
        return moved

    def dynamic_bytes(self, name, original, operations, index):
        current = float(self.time[index])
        if name.startswith('anim_trace_') or name.startswith('anim_input_'):
            return serialize(progressive_path(operations, index + 1), self.reader)
        if name.startswith('anim_parameter_'):
            end = np.searchsorted(self.time, min(current, self.spec['parameter_end_time']) + 1e-12, side='right')
            return serialize(progressive_path(operations, int(end)), self.reader)
        if name.startswith('anim_snapshot_'):
            start = int(name.split('_')[2]) / 1000.
            return original if current + 1e-12 >= start else b''
        if name.startswith('anim_end_'):
            return b''
        if name.startswith('anim_drone_'):
            _, _, glyph, part = name.split('_')
            glyph, part = int(glyph), int(part)
            if glyph == 0:
                return serialize(self.drone_part(operations, part, index), self.reader)
            return b''
        raise ValueError(f'Unknown animation path: {name}')

    def frame(self, index):
        if index == len(self.time) - 1:
            # Restore all drone poses in the final paper frame.
            return self.reference.copy()
        for xref, parts in zip(self.xrefs, self.parts):
            content = [raw if name is None else self.dynamic_bytes(name, raw, body, index)
                       for name, raw, body in parts]
            self.document.update_stream(xref, b'\n'.join(content))
        self.page = self.document.reload_page(self.page)
        return self.pixels()

    def close(self):
        self.document.close()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=here / 'data')
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--output', type=Path, default=here / 'figures/simulation.gif')
    parser.add_argument('--mp4', type=Path)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--hold-seconds', type=float, default=5.)
    args = parser.parse_args()
    if args.fps < 1 or args.hold_seconds < 0.:
        parser.error('fps must be positive and hold-seconds must be nonnegative.')
    if args.output.suffix.lower() != '.gif':
        parser.error('Output must end in .gif.')
    figure = args.figure or args.output.parent / 'figure_paper.pdf'
    if not figure.exists():
        parser.error('Run make_fig.py first.')

    renderer = PaperRenderer(figure, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(renderer.time)))
    indices += [len(renderer.time) - 1] * int(round(args.hold_seconds * args.fps))
    height, width = renderer.reference.shape[:2]
    mp4 = args.mp4 or args.output.with_suffix('.mp4')
    mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(mp4), (width, height), fps=args.fps,
        codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
        output_params=['-crf', '16', '-preset', 'medium', '-movflags', '+faststart',
                       '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=white'])
    writer.send(None)
    try:
        for index in indices:
            writer.send(renderer.frame(index))
    finally:
        writer.close()
        renderer.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    crop = f'crop={width}:{height}:0:0:exact=1'
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        palette = Path(temporary) / 'palette.png'
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(mp4),
                        '-vf', crop + ',palettegen=max_colors=256:stats_mode=full',
                        '-frames:v', '1', str(palette)], check=True)
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(mp4), '-i', str(palette),
                        '-lavfi', '[0:v]' + crop + '[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3',
                        '-loop', '0', str(args.output)], check=True)
    print(f'Saved {args.output} and {mp4}')


if __name__ == '__main__':
    main()
