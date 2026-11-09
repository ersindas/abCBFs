"""Animate the paper PDF. Preserve its fonts, layout, and complete final frame.

The final PNG is exact; GIF and H.264 previews use lossy encoding.
"""
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
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream, FloatObject

from safety_plot_geometry import saved_barrier_kappa


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
        self.paths = {'adaptive': Path(case) / 'adaptive/continuous_candidate.npz',
                      'nominal': Path(case) / 'nominal.npz', 'primary': Path(case) / 'primary.npz',
                      'flows': Path(case) / 'verification/independent_backup_flows.npz'}
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
        self.original_fonts = sorted(self.page.get_fonts(full=True))
        self.reference = self.pixels()
        self.marker_check = self.verify_marker_invisibility()
        reference_png = self.figure.with_suffix('.png')
        self.reference_png_match = None
        if reference_png.exists():
            with Image.open(reference_png) as image:
                self.reference_png_match = np.array_equal(self.reference, np.asarray(image.convert('RGB')))
            if not self.reference_png_match:
                raise ValueError('The paper PNG does not match its PDF at the required render scale.')
        self.transform = self.spec['top_transform']
        self.initial_state = self.adaptive['x'][0]

    def pixels(self):
        pixmap = self.page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3).copy()

    def verify_marker_invisibility(self):
        # Strip only our invisible tags, preserving every enclosed operator.
        stripped = fitz.open(stream=self.original, filetype='pdf')
        for xref, parts in zip(self.xrefs, self.parts):
            stripped.update_stream(xref, b'\n'.join(raw for _, raw, _ in parts))
        page = stripped[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        pixels = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3)
        identical = np.array_equal(self.reference, pixels)
        stripped.close()
        if not identical:
            raise ValueError('Removing invisible animation tags changed the rendered paper figure.')
        return identical

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
            # Match the reference video's single moving drone. The complete
            # paper's historical poses reappear in the untouched final frame.
            return b''
        raise ValueError(f'Unknown animation path: {name}')

    def frame(self, index):
        if index == len(self.time) - 1:
            # The last frame is the unmodified supplied PDF, not a reconstruction.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True, help='Final paper PDF, including its final tick edits.')
    parser.add_argument('--output', type=Path, required=True, help='Output GIF path.')
    parser.add_argument('--mp4', type=Path)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--hold-seconds', type=float, default=5.)
    parser.add_argument('--check-only', action='store_true', help='Validate and render selected frames without encoding an animation.')
    args = parser.parse_args()
    if args.stride < 1 or args.fps < 1 or args.hold_seconds < 0.:
        parser.error('stride and fps must be positive; hold-seconds must be nonnegative.')
    if args.output.suffix.lower() != '.gif':
        parser.error('Output must end in .gif.')
    renderer = PaperRenderer(args.figure, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(0, len(renderer.time), args.stride))
    if indices[-1] != len(renderer.time) - 1:
        indices.append(len(renderer.time) - 1)
    indices += [len(renderer.time) - 1] * int(round(args.hold_seconds * args.fps))
    height, width = renderer.reference.shape[:2]
    mp4 = args.mp4 or args.output.with_suffix('.mp4')
    if args.check_only:
        backup = np.flatnonzero(renderer.adaptive['used_backup'])
        samples = sorted(set([0, int(backup[0]) if len(backup) else 1,
                              len(renderer.time) // 2, len(renderer.time) - 1]))
        for index in samples:
            Image.fromarray(renderer.frame(index)).save(args.output.with_name(f'{args.output.stem}_check_{index}.png'))
    else:
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
    final = renderer.frame(len(renderer.time) - 1)
    if not np.array_equal(final, renderer.reference):
        raise AssertionError('Final animation frame differs from the original paper PDF.')
    if sorted(renderer.page.get_fonts(full=True)) != renderer.original_fonts:
        raise AssertionError('Animation changed the paper font resources.')
    Image.fromarray(final).save(args.output.with_suffix('.png'))
    report = {'sources': {key: str(path.resolve()) for key, path in renderer.paths.items()},
              'source_sha256': renderer.spec['source_sha256'], 'kappa': renderer.kappa,
              'figure': str(args.figure.resolve()), 'figure_sha256': hashlib.sha256(renderer.original).hexdigest(),
              'simulation_time_range': [float(renderer.time[0]), float(renderer.time[-1])],
              'frames': len(indices), 'fps': args.fps, 'controller_sample_indices': indices,
              'requested_end_hold_seconds': args.hold_seconds,
              'final_pose_hold_seconds': (len(indices) - indices.index(len(renderer.time) - 1)) / args.fps,
              'final_paper_drone_pose_count': len(renderer.pose_times),
              'final_paper_drone_pose_times': list(renderer.pose_times.values()),
              'all_final_paper_drone_poses_complete': True,
              'used_backup_times': renderer.time[renderer.adaptive['used_backup']].tolist(),
              'snapshot_times': renderer.spec['snapshot_times'],
              'raw_frame_size': [width, height], 'mp4_size': [width + width % 2, height + height % 2],
              'mp4_padding': 'At most one white pixel at the right/bottom for H.264 4:2:0; original content is not rescaled.',
              'mp4': str(mp4), 'check_only': args.check_only,
              'invisible_tags_render_identically': renderer.marker_check,
              'paper_png_matches_pdf': renderer.reference_png_match,
              'final_raw_frame_pixel_identical_to_paper_pdf': True,
              'original_font_resources_unchanged': True,
              'raw_final_frame_sha256': hashlib.sha256(final.tobytes()).hexdigest(),
              'fonts_and_layout': 'Exact supplied PDF fonts, axes, labels, legend, colors and panel geometry; no Matplotlib recreation.',
              'animation_edits': 'Progressive recorded paths and backup snapshots, with one original moving drone glyph using saved pitch. The final untouched paper frame includes every static paper pose.',
              'encoding_note': 'GIF quantization and H.264 compression are lossy; pixel identity is asserted for the uncompressed final frame and PNG, not encoded previews.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    renderer.close()
    print(json.dumps({'animation': str(args.output), 'frames': len(indices),
                      'fps': args.fps, 'raw_frame_size': [width, height], 'check_only': args.check_only}, indent=2))


if __name__ == '__main__':
    main()
