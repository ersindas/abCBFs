"""Render the paper figure using its original fonts and layout."""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json

import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, FloatObject, NameObject
from safety_plot_geometry import (projected_safe_boundary, projected_safe_polygon,
                                  projected_terminal_boundary, saved_barrier_kappa)


PAINT = {b'S', b's', b'f', b'f*', b'F', b'B', b'B*', b'b', b'b*'}
PATH = {b'm', b'l', b'c', b'h', b're', b'v', b'y'}


def parsed(raw, reader):
    d = DecodedStreamObject()
    d.set_data(raw.encode('ascii') if isinstance(raw, str) else raw)
    return ContentStream(d, reader).operations


def floats(values):
    return [FloatObject(float(v)) for v in values]


def polyline(points, close=False):
    ops = [(floats(p), b'm' if i == 0 else b'l') for i, p in enumerate(points)]
    if close:
        ops.append(([], b'h'))
    return ops


def animation_group(name, operations):
    """Invisible PDF marked content; retains every existing drawing operator."""
    return [([NameObject('/' + name)], b'BMC')] + operations + [([], b'EMC')]


def path_index(ops):
    paints, pending = [], []
    for i, (args, op) in enumerate(ops):
        if op in PATH:
            pending.append(i)
        elif op == b'n':
            pending = []
        elif op in PAINT:
            paints.append((pending, i))
            pending = []
    return paints


def text_blocks(ops):
    blocks, start = [], None
    for i, (_, op) in enumerate(ops):
        if op == b'BT':
            start = i
        elif op == b'ET':
            blocks.append((start, i))
    return blocks


def transform_path(ops, old_center, new_center, scale=(1., 1.)):
    old_center, new_center, scale = map(np.asarray, (old_center, new_center, scale))
    result = []
    for args, op in ops:
        if op in (b'm', b'l', b'c'):
            arr = np.asarray(args, dtype=float).reshape(-1, 2)
            result.append((floats(((arr-old_center)*scale+new_center).ravel()), op))
        else:
            result.append((args, op))
    return result


class Editor:
    def __init__(self, form, reader):
        self.form, self.reader = form, reader
        self.ops = copy.deepcopy(ContentStream(form, reader).operations)
        self.paths = path_index(self.ops)
        self.replacements, self.deletions, self.insertions = {}, set(), {}

    def replace_path(self, number, path):
        indices, paint = self.paths[number]
        assert indices
        self.replacements[indices[0]] = path
        self.deletions.update(indices[1:])
        if not path:
            self.replacements[paint] = [([], b'n')]

    def original_path(self, number):
        return [self.ops[i] for i in self.paths[number][0]]

    def tag_path(self, number, name):
        indices, paint = self.paths[number]
        first = indices[0]
        self.replacements[first] = ([([NameObject('/' + name)], b'BMC')] +
                                    self.replacements.get(first, [self.ops[first]]))
        self.replacements[paint] = (self.replacements.get(paint, [self.ops[paint]]) +
                                    [([], b'EMC')])

    def shift_text(self, first_td_index, dx=0., dy=0.):
        args, op = self.ops[first_td_index]
        assert op == b'Td'
        self.replacements[first_td_index] = [(floats([float(args[0])+dx, float(args[1])+dy]), op)]

    def finish(self):
        out = []
        for i, op in enumerate(self.ops):
            out.extend(self.insertions.get(i, []))
            if i in self.deletions:
                continue
            out.extend(self.replacements.get(i, [op]))
        stream = ContentStream(None, self.reader)
        stream.operations = out
        self.form.set_data(stream.get_data())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--case', type=Path, required=True,
                    help='Directory containing the simulation data.')
    ap.add_argument('--snapshot-times', type=float, nargs='+', default=None,
                    help='Retained saved snapshot times; defaults to 0.08, 0.3, 0.5, 1.1, 1.93 seconds.')
    args = ap.parse_args()
    assets = Path(__file__).resolve().parent / 'assets'
    case = args.case.resolve()
    paths = dict(
        template=assets / 'paper_template.pdf',
        adaptive=case / 'adaptive/continuous_candidate.npz',
        nominal=case / 'nominal.npz',
        primary=case / 'primary.npz',
        flows=case / 'verification/independent_backup_flows.npz',
        tubes=args.output / 'tube_visualization/tube_preview_data.npz',
        original_data=assets / 'glyph_template.npz')
    d = {k: np.load(p) for k, p in paths.items() if k != 'template'}
    a, n, p0, f, tubes, old = [d[k] for k in ['adaptive', 'nominal', 'primary', 'flows', 'tubes', 'original_data']]
    kappa = saved_barrier_kappa(a, n, p0)
    if 'kappa' in f:
        saved_barrier_kappa(a, f)
    assert np.array_equal(a['x'][0], n['x'][0]) and np.array_equal(a['x'][0], p0['x'][0])
    source_snapshot_times = np.asarray(tubes['times'], dtype=float)
    template_snapshot_times = np.array([.08,.3,.5,.7,.9,1.1])
    times = np.asarray([.08,.3,.5,1.1,1.93] if args.snapshot_times is None else args.snapshot_times, dtype=float)
    assert times.shape == (5,) and np.all(np.diff(times) > 0.)
    assert abs(times[0]-.08) < 1e-12
    ids = np.array([int(np.argmin(abs(f['t']-t))) for t in times])
    tid = np.array([int(np.argmin(abs(tubes['times']-t))) for t in times])
    assert np.allclose(f['t'][ids], times, atol=1e-12, rtol=0.)
    assert np.allclose(tubes['times'][tid], times, atol=1e-12, rtol=0.)
    assert np.max(abs(tubes['centers'][tid]-f['estimated_flows'][ids, :, :2])) < 1e-12
    from tube_geometry import swept_polygons

    # These are the original local PDF coordinates of the top Form stream.
    x0, y0 = 58.9485586035, 222.1478987833
    width, height = 544.5889678956, 190.8916212167
    coordinate_bounds = np.concatenate([a['x'][:, :2], n['x'][:, :2], p0['x'][:, :2],
        f['estimated_flows'][ids, :, :2].reshape(-1, 2), f['true_flows'][ids, :, :2].reshape(-1, 2),
        (tubes['centers'][tid]-tubes['radii'][tid]).reshape(-1, 2),
        (tubes['centers'][tid]+tubes['radii'][tid]).reshape(-1, 2)])
    xlim = (.5, 7.4)
    ylim = (-.15, 3.5)  # User-requested display limits; full saved prediction data remain unchanged.

    def xy(points, old_scale=False):
        pp = np.asarray(points, dtype=float)
        zz0, zz1 = (.5, 3.5) if old_scale else ylim
        return np.stack([x0+(pp[..., 0]-.5)*width/6.9,
                         y0+(pp[..., 1]-zz0)*height/(zz1-zz0)], axis=-1)

    physical_fail = np.flatnonzero(np.any(n['h_components'] < 0., axis=1))
    nominal_end = len(n['t'])
    nominal_xy = n['x'][:nominal_end, :2]
    nominal_minimum_altitude_index = int(np.argmin(n['dense_x'][:, 1]))
    template_glyph_times = [0., .30, .60, .90, 1.15, 1.35, 4.]
    glyph_slot_times = [0., .30, .60, None, 1.15, 1.93, 4.]
    glyph_times = [t for t in glyph_slot_times if t is not None]
    lower_x = [58.9485586035, 260.4782145687, 462.0078705339]
    lower_width, lower_bottom, lower_height = 141.5296559652, 40.04893, 65.10904
    large_parameter_minimum = float(min(a['theta_hat'][:, [0, 2, 3]].min(),
                                        a['theta_true'][[0, 2, 3]].min()))
    large_parameter_lower = (-.36 if large_parameter_minimum >= -.345 else
                             float(np.floor((large_parameter_minimum-.02)*10)/10))
    lower_ranges = [((-23., 23.), 4.), ((large_parameter_lower, .26), 1.2), ((-.008, .092), 1.2)]
    # Compact three-column, two-row legend in the unused unsafe region.
    # Preserve exact embedded source glyphs and font sizes.
    original_legend_bbox = [435.0039367675781, 91.06597900390625,
                            597.1875, 185.98162841796875]
    legend_left_data, legend_data_right = 2.75, 7.30
    legend_data_bottom = -.10
    legend_top_data = legend_data_bottom + (.78+.10)*(3.65/4.85)
    legend_pdf_lo = xy([legend_left_data, legend_data_bottom])
    legend_pdf_hi = xy([legend_data_right, legend_top_data])
    annotation_positions = dict(backup_set=[3.25, 3.08], safe_set=[1.62, 2.12],
                                floor_label=[6.78, 1.22], wall_label=[1.06, 3.02], goal=[.74, .98],
                                unsafe=[1.12, .48], flow_error_label=[4.25,3.24],
                                final_time=[float(a['x'][-1,0]),1.35])

    def shifted_text_group(ops, first_td, anchor):
        result = copy.deepcopy(ops)
        for i, (values, op) in enumerate(result):
            if i == first_td:
                assert op == b'Td'
                result[i] = (floats(anchor), op)
        return result

    def refined_legend(edit, reader):
        left, bottom = legend_pdf_lo
        right, top_y = legend_pdf_hi
        result = parsed(f'q /A2 gs 1 g .8 G .7 w {left:.10f} {bottom:.10f} {right-left:.10f} {top_y-bottom:.10f} re B', reader)
        # Matplotlib's original glyph runs, with shorter handle-to-text gaps.
        text_ranges = [(3745,3771), (3780,3800), (3813,3818), (3831,3836), (3849,3854)]
        old_anchor_x = [472.139575,470.563941,470.563941,470.563941,470.563941]
        # First column: flow descriptions. Second: primary and nominal.
        # Third: robust adaptive controller.
        column_x = [left+4., left+4.+81., left+4.+81.+126.]
        row_y = [float(legend_pdf_lo[1]+(.48+.10)*height/4.85),
                 float(legend_pdf_lo[1]+(.04+.10)*height/4.85)]
        entries = [(0,0), (0,1), (1,0), (1,1), (2,0)]
        colors = ['.1215686275 .6156862745 .3333333333', '0 0 0',
                  '.2039215686 .2862745098 .3686274510',
                  '.0549019608 .4549019608 .5647058824',
                  '.4235294118 .2274509804 .5568627451']
        line_widths = [1.3,1.,1.8,2.3*.75,2.3]
        dashes = ['[]','[4 2]','[9 3.6]','[]','[]']
        for i, ((col,row), (begin,end)) in enumerate(zip(entries,text_ranges)):
            xx, yy = column_x[col], row_y[row]
            result += parsed(f'q {colors[i]} RG {line_widths[i]} w {dashes[i]} 0 d 0 J {xx:.10f} {yy+4.445:.10f} m {xx+16.32:.10f} {yy+4.445:.10f} l S 0 g', reader)
            block = copy.deepcopy(edit.ops[begin:end])
            first = next(j for j, (_, op) in enumerate(block) if op == b'Td')
            # Estimated-flow hat has its own original horizontal offset.
            offset = old_anchor_x[i]-470.563941
            block[first] = (floats([xx+20.32+offset,yy]), b'Td')
            result += block + parsed('Q', reader)
        result += parsed('Q', reader)
        return result

    def lower_xy(panel, tt, vv):
        (lo, hi), tend = lower_ranges[panel]
        return np.column_stack([lower_x[panel]+np.asarray(tt)/tend*lower_width,
                                lower_bottom+(np.asarray(vv)-lo)/(hi-lo)*lower_height])

    for with_tubes in [False, True]:
        reader = PdfReader(paths['template'])
        top = reader.pages[0]['/Resources']['/XObject']['/fzFrm0']['/Resources']['/XObject']['/fullpage']
        lower = reader.pages[0]['/Resources']['/XObject']['/fzFrm1']['/Resources']['/XObject']['/fullpage']
        edit = Editor(top, reader)
        assert len(edit.paths) == 80
        # Project the implemented smooth safe set over pitch, rather than
        # shading the larger intersection of the two position half spaces.
        safe_curve = projected_safe_boundary(xlim[1], ylim[1], kappa=kappa)
        edit.replace_path(2, polyline(xy(projected_safe_polygon(xlim[1], ylim[1], kappa=kappa)), True))
        curve = projected_terminal_boundary(xlim[1], ylim[1], kappa=kappa)
        edit.replace_path(3, polyline(xy(np.vstack([curve, [7.4, ylim[1]]])), True))
        edit.replace_path(4, polyline(xy(curve)))
        edit.replace_path(15, polyline(xy(safe_curve)))
        edit.replace_path(16, [])

        # Replace only vertical numeric ticks; retain the source font object,
        # exact size, tick width, label x position and baseline offset.
        for number in [12, 13, 14]:
            edit.replace_path(number, [])
        for start, stop in [(3359, 3363), (3369, 3373), (3379, 3383)]:
            assert edit.ops[start][1] == b'BT' and edit.ops[stop][1] == b'ET'
            edit.deletions.update(range(start, stop+1))
        ticks = ['q /A2 gs 0 g 0 G']
        for z in range(int(np.ceil(ylim[0])), int(np.floor(ylim[1]))+1):
            yy = float(xy([.5, float(z)])[1])
            ticks += [f'.8 w {x0:.10f} {yy:.10f} m {x0+3.5:.10f} {yy:.10f} l B',
                      f'BT /F1 15.9402241594 Tf 47.6445716 {yy-5.534793626:.10f} Td ({z}) Tj ET']
        ticks.append('Q')
        edit.insertions[edit.paths[12][0][0]] = parsed('\n'.join(ticks), reader)
        edit.replace_path(21, polyline(xy(p0['x'][:, :2])))
        edit.replace_path(22, polyline(xy(nominal_xy)))
        edit.replace_path(23, polyline(xy(a['x'][:, :2])))
        edit.deletions.update(range(3713, 3855))
        edit.insertions[3713] = refined_legend(edit, reader)
        edit.replacements[3511] = [(floats([2.3*.75]), b'w')]
        # Original glyph geometry and colors at the original display times.
        for j, t in enumerate(glyph_slot_times):
            if t is None:
                for path_number in range(30+5*j,35+5*j):
                    edit.replace_path(path_number, [])
                continue
            state = a['x'][int(np.argmin(abs(a['t']-t)))]
            old_template_time = template_glyph_times[j]
            old_state = old['xa'][:, min(int(old_template_time/float(old['dt'])), old['xa'].shape[1]-1)]
            heading = np.array([np.cos(state[2]), -np.sin(state[2])])
            up = np.array([np.sin(state[2]), np.cos(state[2])])
            p1, p2 = state[:2]-.22*heading, state[:2]+.22*heading
            old_heading = np.array([np.cos(old_state[2]), -np.sin(old_state[2])])
            old_up = np.array([np.sin(old_state[2]), np.cos(old_state[2])])
            old1, old2 = old_state[:2]-.22*old_heading, old_state[:2]+.22*old_heading
            start = 30+5*j
            edit.replace_path(start, polyline(xy([p1, p2])))
            for k, (point, oldpoint) in enumerate([(p1, old1), (p2, old2)]):
                tip, oldtip = point+.11*up, oldpoint+.11*old_up
                edit.replace_path(start+1+2*k, polyline(xy([point, tip])))
                edit.replace_path(start+2+2*k, transform_path(edit.original_path(start+2+2*k),
                    xy(oldtip, True), xy(tip), (1., 3./(ylim[1]-ylim[0]))))
        for j, source_time in enumerate(template_snapshot_times[1:]):
            if np.any(np.isclose(times,source_time,atol=1e-12,rtol=0.)):
                i = int(np.argmin(abs(f['t']-source_time)))
                edit.replace_path(65+2*j, polyline(xy(f['estimated_flows'][i, :, :2])))
                edit.replace_path(66+2*j, polyline(xy(f['true_flows'][i, :, :2])))
            else:
                edit.replace_path(65+2*j, [])
                edit.replace_path(66+2*j, [])
        # The first and newly requested late snapshot have no template slots.
        extra_flows = []
        for extra_time in [.08,1.93]:
            extra_i = int(np.argmin(abs(f['t']-extra_time)))
            assert abs(float(f['t'][extra_i])-extra_time) < 1e-12
            group = parsed(f'q /A2 gs {x0} {y0} {width} {height} re W n .1215686275 .6156862745 .3333333333 RG 1.3 w [] 0 d', reader)
            group += polyline(xy(f['estimated_flows'][extra_i, :, :2])) + [([], b'S')]
            group += parsed('0 G 1 w [4 2] 0 d', reader)
            group += polyline(xy(f['true_flows'][extra_i, :, :2])) + [([], b'S')]
            group += parsed('Q', reader)
            extra_flows += animation_group(f'anim_snapshot_{round(extra_time * 1000)}_extra', group)
        edit.insertions[edit.paths[65][0][0]] = extra_flows
        # The requested annotation removes the cyan X, not trajectory samples.
        edit.replace_path(75, [])
        edit.replace_path(76, polyline(xy([[1., 1.]])))
        for num, oldp, newp in [(77, [1., 1.], [1., 1.]),
                (78, old['xa'][:2, 0], a['x'][0, :2]),
                (79, old['xa'][:2, -1], a['x'][-1, :2])]:
            # Goal and endpoint markers are sized in points, not state units.
            edit.replace_path(num, transform_path(edit.original_path(num), xy(oldp, True), xy(newp)))
        floor_delta = float(xy([1., 1.])[1]-xy([1., 1.], True)[1])
        wall_anchor = xy(annotation_positions['wall_label'])
        edit.shift_text(4868, *(wall_anchor-np.asarray(edit.ops[4868][0], dtype=float)))
        floor_anchor = xy(annotation_positions['floor_label'])
        edit.shift_text(4879, *(floor_anchor-np.asarray(edit.ops[4879][0], dtype=float)))
        goal_anchor = xy(annotation_positions['goal'])-np.array([12.120552, 0.])
        edit.shift_text(4947, *(goal_anchor-np.asarray(edit.ops[4947][0], dtype=float)))
        unsafe_anchor = xy(annotation_positions['unsafe'])-np.array([20.136963, 0.])
        edit.shift_text(4890, *(unsafe_anchor-np.asarray(edit.ops[4890][0], dtype=float)))
        safe_anchor = xy(annotation_positions['safe_set'])-np.array([23.4349975, 0.])
        safe_shift = safe_anchor-np.asarray(edit.ops[4895][0], dtype=float)
        edit.shift_text(4895, *safe_shift)
        edit.shift_text(4900, *safe_shift)
        start_delta = xy(a['x'][0, :2])-xy(old['xa'][:2, 0], True)
        end_delta = xy(a['x'][-1, :2])-xy(old['xa'][:2, -1], True)
        edit.shift_text(4982, *start_delta)
        final_time_anchor = xy(annotation_positions['final_time']) - np.array([15.05748, 0.])
        edit.shift_text(5020, *(final_time_anchor-np.asarray(edit.ops[5020][0], dtype=float)))
        # Shift the existing backup-set glyphs into the open upper region,
        # and draw them after geometry so tubes cannot cover letters.
        backup_label_ops = copy.deepcopy(edit.ops[3249:3262])
        backup_anchor = xy(annotation_positions['backup_set'])-np.array([34.344086, 0.])
        backup_shift = backup_anchor-np.asarray(edit.ops[3251][0], dtype=float)
        for local_index in [2,7]:
            old_values, op = backup_label_ops[local_index]
            assert op == b'Td'
            backup_label_ops[local_index] = (floats(np.asarray(old_values,dtype=float)+backup_shift),op)
        edit.deletions.update(range(3249, 3262))
        edit.insertions[len(edit.ops)-1] = (parsed('q .3333333333 g', reader)
            + backup_label_ops + parsed('Q', reader))

        # Semantic tags are non-rendering PDF operators. The animation edits
        # these exact paths while keeping all fonts, axes and labels intact.
        for number, name in [(21, 'primary'), (22, 'nominal'), (23, 'adaptive')]:
            edit.tag_path(number, 'anim_trace_' + name)
        for j, time in enumerate(glyph_slot_times):
            if time is not None:
                for part in range(5):
                    edit.tag_path(30 + 5 * j + part, f'anim_drone_{j}_{part}')
        for j, time in enumerate(template_snapshot_times[1:]):
            if np.any(np.isclose(times, time, atol=1e-12, rtol=0.)):
                for kind in range(2):
                    edit.tag_path(65 + 2 * j + kind, f'anim_snapshot_{round(time * 1000)}_{kind}')
        edit.tag_path(79, 'anim_end_marker')
        edit.replacements[5018] = [([NameObject('/anim_end_label')], b'BMC'), edit.ops[5018]]
        edit.replacements[5025] = [edit.ops[5025], ([], b'EMC')]

        if with_tubes:
            tube_ops = parsed(f'q /A2 gs {x0} {y0} {width} {height} re W n .86 .93 .83 rg', reader)
            for i in tid:
                # Identical opaque fills form the geometric union without
                # opacity accumulation where adjacent swept cells overlap.
                group = []
                for polygon in swept_polygons(tubes['centers'][i], tubes['radii'][i]):
                    if len(polygon) >= 3:
                        group.extend(polyline(xy(polygon), True))
                        group.append(([], b'f'))
                tube_ops.extend(animation_group(f'anim_snapshot_{round(float(tubes["times"][i]) * 1000)}_tube', group))
            tube_ops.extend(parsed('Q', reader))
            # Below every trajectory and legend; above background sets.
            edit.insertions[edit.paths[21][0][0]] = tube_ops
        # Draw quadrotor bodies/rotors after all flow curves and tube fills.
        # Move the complete original graphics-state groups with revised paths.
        drone_ops = []
        for op_index in range(3855,4443):
            if op_index not in edit.deletions:
                drone_ops.extend(edit.replacements.get(op_index,[edit.ops[op_index]]))
        edit.deletions.update(range(3855,4443))
        tail = edit.insertions.get(len(edit.ops)-1,[])
        if with_tubes:
            flow_label_anchor = xy(annotation_positions['flow_error_label'])
            leader = xy([[5.415,3.235],[5.66,3.16]])
            tail += animation_group('anim_snapshot_80_label', parsed(f'q /A2 gs .3333333333 g .3333333333 G .5 w [] 0 d {leader[0,0]:.10f} {leader[0,1]:.10f} m {leader[1,0]:.10f} {leader[1,1]:.10f} l S BT /F1 12.6525499 Tf {flow_label_anchor[0]:.10f} {flow_label_anchor[1]:.10f} Td [(flow) -326.385498 (error) -326.385498 (bound)] TJ ET Q', reader))
        edit.insertions[len(edit.ops)-1] = tail + drone_ops
        edit.finish()

        edit_lower = Editor(lower, reader)
        assert len(edit_lower.paths) == 63
        # Preserve the existing tick text and font objects while mapping
        # their values to the widened parameter range. The original three
        # tick values remain legible without crowding the time labels.
        for number, value, first_td in [(27, -.2, 505), (28, 0., 524), (29, .2, 540)]:
            yy = float(lower_xy(1, [0.], [value])[0, 1])
            old_yy = lower_bottom+(value+.36)/.62*lower_height
            edit_lower.replace_path(number, polyline([[lower_x[1], yy], [lower_x[1]+3.5, yy]]))
            edit_lower.shift_text(first_td, dy=yy-old_yy)
        for j in range(2):
            edit_lower.replace_path(18+j, polyline(lower_xy(0, a['t'], a['u'][:, j])))
            edit_lower.tag_path(18+j, f'anim_input_{j}')
        mask = a['t'] <= 1.2+1e-12
        for panel, groups, path_start in [(1, [0, 2, 3], 37), (2, [1, 4, 5], 57)]:
            for j, parameter in enumerate(groups):
                edit_lower.replace_path(path_start+j, polyline(lower_xy(panel, a['t'][mask], a['theta_hat'][mask, parameter])))
                edit_lower.tag_path(path_start+j, f'anim_parameter_{parameter}')
                truth_start = 30 if panel == 1 else 50
                edit_lower.replace_path(truth_start+j, polyline(lower_xy(panel, [0., 1.2], [a['theta_true'][parameter]]*2)))
        edit_lower.finish()
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.add_metadata({'/AnimationSpec': json.dumps(dict(version=1, kappa=kappa,
            source_sha256={key: hashlib.sha256(paths[key].read_bytes()).hexdigest()
                           for key in ['adaptive', 'nominal', 'primary', 'flows']},
            top_transform=dict(origin=[x0, y0], size=[width, height], xlim=xlim, ylim=ylim),
            glyph_times=glyph_slot_times, snapshot_times=times.tolist(),
            parameter_end_time=1.2))})
        # Remove 19 pt of unused left page margin, leaving >3 pt around the
        # leftmost lower-panel label. Panel and text coordinates stay unchanged.
        writer.pages[0].cropbox.lower_left = (19., 0.)
        writer.pages[0].mediabox.lower_left = (19., 0.)
        args.output.mkdir(parents=True, exist_ok=True)
        filename = 'figure_paper_tubes' if with_tubes else 'figure_paper_updated'
        out = args.output / (filename+'.pdf')
        with out.open('wb') as stream:
            writer.write(stream)
        doc = fitz.open(out)
        doc[0].get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(out.with_suffix('.png'))
        doc.close()

    bounds = np.concatenate([a['x'][:, :2], n['x'][:, :2], p0['x'][:, :2],
        f['estimated_flows'][ids, :, :2].reshape(-1, 2), f['true_flows'][ids, :, :2].reshape(-1, 2),
        (tubes['centers'][tid]-tubes['radii'][tid]).reshape(-1, 2),
        (tubes['centers'][tid]+tubes['radii'][tid]).reshape(-1, 2)])
    assert bounds[:, 0].min() >= xlim[0] and bounds[:, 0].max() <= xlim[1]
    assert bounds[:, 1].min() >= ylim[0]
    clipped_predictions = []
    for time, flow_i, tube_i in zip(times,ids,tid):
        entry = dict(time=float(time), estimated_flow_max_z=float(f['estimated_flows'][flow_i,:,1].max()),
            true_flow_max_z=float(f['true_flows'][flow_i,:,1].max()),
            tube_max_z=float((tubes['centers'][tube_i,:,1]+tubes['radii'][tube_i,:,1]).max()))
        if max(entry['estimated_flow_max_z'],entry['true_flow_max_z'],entry['tube_max_z']) > ylim[1]:
            clipped_predictions.append(entry)
    center_points = np.concatenate([a['x'][:, :2], n['x'][:, :2], p0['x'][:, :2],
        f['estimated_flows'][ids, :, :2].reshape(-1, 2), f['true_flows'][ids, :, :2].reshape(-1, 2)])
    inside_legend = ((center_points[:, 0] >= legend_left_data) & (center_points[:, 0] <= legend_data_right)
                     & (center_points[:, 1] >= legend_data_bottom) & (center_points[:, 1] <= legend_top_data))
    tube_lower = tubes['centers'][tid]-tubes['radii'][tid]
    tube_upper = tubes['centers'][tid]+tubes['radii'][tid]
    tube_hits_legend = ((tube_upper[:, :, 0] >= legend_left_data) & (tube_lower[:, :, 0] <= legend_data_right)
                        & (tube_upper[:, :, 1] >= legend_data_bottom) & (tube_lower[:, :, 1] <= legend_top_data))
    assert not inside_legend.any() and not tube_hits_legend.any()
    from shapely.geometry import LineString, box
    legend_rectangle = box(*legend_pdf_lo, *legend_pdf_hi)
    legend_curve_clearance = {}
    for curve_name, curve_data, stroke_width in [('adaptive',a['x'][:,:2],2.3),
            ('nominal',nominal_xy,2.3*.75), ('primary',p0['x'][:,:2],1.8)]:
        clearance = float(LineString(xy(curve_data)).distance(legend_rectangle)-stroke_width/2.-.7/2.)
        assert clearance > 0., (curve_name,clearance)
        legend_curve_clearance[curve_name] = clearance

    audit = dict(source_sha256={k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in paths.items()},
        kappa=kappa,
        safe_set_projection=f'Projection over pitch of h_soft >= 0, with kappa={kappa:g} and theta_max=pi/3; boundary evaluated at theta=0.',
        terminal_position_projection=f'Closure of the smooth terminal surrogate projection, kappa={kappa:g}, position margins 2m, maximum pitch barrier 0.05.',
        true_parameters=a['theta_true'].tolist(), actual_mass_kg=float(1./(1.+a['theta_true'][3])),
        template_stream_edit=True, page_size_pt=[589.8795166, 341.7200012],
        changed_axis_range={'p_z_original': [.5, 3.5], 'p_z_updated': list(ylim),
            'reason': 'User requested p_z upper display limit 3.5 m. Full stored one-second prediction data are drawn with the axes clipping path.'},
        original_panel_geometry_fonts_colors_and_annotation_wording_preserved=True,
        legend_relocation_only=False,
        appearance_changes={
            'legend': 'Three columns and two rows in lower right, exact source glyphs and font sizes, compact spacing and rectangular border.',
            'annotation_anchor_positions_data': annotation_positions,
            'nominal_curve_width_pt_original': 2.3,
            'nominal_curve_width_pt_updated': 2.3*.75,
            'nominal_endpoint_cross_removed': True,
            'outside_left_margin_cropped_pt': 19.,
            'lower_panels': 'Numerical content and complete panel appearance retained.',
            'quadrotor_layer': 'All bodies, stems, and rotors drawn after curves and tube fills.',
            'flow_error_label': 'Added flow error bound text beside earliest displayed shaded envelope.',
            'omitted_flow_tube_snapshots': [.7,.9],
            'drone_omitted_time': .9,
            'drone_relocated_from_time': 1.35,
            'drone_relocated_to_time': 1.93,
            'new_flow_tube_snapshot_time': 1.93},
        legend_original_bbox_pt=original_legend_bbox,
        legend_updated_bbox_data=[legend_left_data,legend_data_bottom,legend_data_right,legend_top_data],
        no_trajectory_or_tube_crosses_legend=True,
        legend_curve_stroke_clearance_pt=legend_curve_clearance,
        parameter_groups=[[0, 2, 3], [1, 4, 5]], parameter_xlim=[0., 1.2],
        middle_parameter_ylim_original=[-.36, .26],
        middle_parameter_ylim_updated=[large_parameter_lower, .26],
        snapshot_times=times.tolist(), source_snapshot_times=source_snapshot_times.tolist(), full_backup_horizon=1., flow_samples=int(f['taus'].size),
        drone_glyph_convention='Original draw_quad: body=[cos(theta),-sin(theta)], stems=[sin(theta),cos(theta)]; rotor centers rotate with stems; rotor disks remain circles.',
        drone_glyphs=[dict(time=float(t), saved_time=float(a['t'][int(np.argmin(abs(a['t']-t)))]),
            position=a['x'][int(np.argmin(abs(a['t']-t))), :2].tolist(),
            pitch_rad=float(a['x'][int(np.argmin(abs(a['t']-t))), 2]),
            pitch_deg=float(np.rad2deg(a['x'][int(np.argmin(abs(a['t']-t))), 2]))) for t in glyph_times],
        initial_state=a['x'][0].tolist(),
        nominal_prediction_dt=float(n['prediction_dt']), adaptive_prediction_dt=float(a['prediction_dt']),
        nominal_first_physical_violation_sample_time=float(n['t'][physical_fail[0]]) if physical_fail.size else None,
        nominal_display_last_time=float(n['t'][nominal_end-1]),
        nominal_display_rule='Complete four-second trajectory; cyan endpoint cross removed at user request.',
        nominal_cross_time=None,
        nominal_cross_state=None,
        nominal_minimum_altitude_time=float(n['dense_t'][nominal_minimum_altitude_index]),
        nominal_minimum_altitude_state=n['dense_x'][nominal_minimum_altitude_index].tolist(),
        primary_display_duration=float(p0['t'][-1]), adaptive_display_duration=float(a['t'][-1]),
        tube_interpretation='Union of computed axis-aligned position error rectangles under piecewise-linear interpolation.',
        tube_color_rgb=[.86, .93, .83], tube_opaque_for_eps=True,
        tube_snapshot_times=times.tolist(), all_complete_flows_and_tubes_within_axes=(len(clipped_predictions)==0),
        predictions_clipped_by_user_requested_display_limits=clipped_predictions,
        prediction_data_truncated=False,
        displayed_snapshot_count=int(len(times)),
        displayed_quadrotor_count=len(glyph_times),
        omitted_display_snapshot_times=[.7,.9],
        quadrotor_graphics_drawn_after_all_flow_curves=True,
        adaptive_backup_steps=int(np.count_nonzero(a['used_backup'][:-1])),
        adaptive_minimum_recorded_tightened_barrier=float(a['hbar'].min()),
        simulations_run_by_plot_script=False)
    (args.output/'figure_verification.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
