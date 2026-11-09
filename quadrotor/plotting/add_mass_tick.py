"""Add the final mass-parameter axis tick."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, TextStringObject


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    assert args.input.resolve() != args.output.resolve()
    reader = PdfReader(args.input)
    page = reader.pages[0]
    xobjects = page['/Resources']['/XObject']
    lower = xobjects['/fzFrm1']['/Resources']['/XObject']['/fullpage']
    top = xobjects['/fzFrm0']['/Resources']['/XObject']['/fullpage']
    original = lower.get_data()
    original_top = top.get_data()
    marker = b'% Added regular parameter tick at -0.4; original operators unchanged.'
    assert b'% Added true delta_m tick' not in original and marker not in original, 'Use the original figure, without an already appended tick.'
    ops = ContentStream(lower, reader).operations

    def block_at(x, y=None):
        matches = []
        for start, (_, op) in enumerate(ops):
            if op != b'BT':
                continue
            end = start + 1
            while ops[end][1] != b'ET':
                end += 1
            block = ops[start:end + 1]
            td = next((i for i, (_, op) in enumerate(block) if op == b'Td'), None)
            if td is not None:
                pos = list(map(float, block[td][0]))
                if abs(pos[0] - x) < 1e-5 and (y is None or abs(pos[1] - y) < 1e-5):
                    matches.append((block, td, pos))
        assert len(matches) == 1, (x, y, len(matches))
        return matches[0]

    # Reuse the exact minus, decimal point and digit glyph runs of the -0.2 tick.
    negative, td, pos = block_at(224.636696, 54.244148)
    zero, zero_td, zero_pos = block_at(237.034692, 73.9741601)
    scale_y = (zero_pos[1] - pos[1]) / .2
    new_baseline = pos[1] + (-.4 - (-.2)) * scale_y
    label = copy.deepcopy(negative)
    label[td] = ([FloatObject(pos[0]), FloatObject(new_baseline)], b'Td')
    last_text = max(i for i, (_, op) in enumerate(label) if op == b'Tj')
    assert label[last_text][0][0] == '2'
    label[last_text] = ([TextStringObject('4')], b'Tj')
    label_stream = ContentStream(None, reader)
    label_stream.operations = label

    axis_x = 260.4782145687
    axis_bottom = 40.04893
    axis_height = 65.10904
    axis_lower, axis_upper = -.4, .26
    tick_y = axis_bottom + (-.4 - axis_lower) / (axis_upper - axis_lower) * axis_height
    assert abs(scale_y - axis_height / (axis_upper - axis_lower)) < 1e-5
    append = (b'\n' + marker + b'\nq 0 g 0 G .8 w [] 0 d 0 J\n' +
              f'{axis_x:.10f} {tick_y:.10f} m {axis_x + 3.5:.10f} {tick_y:.10f} l S\n'.encode('ascii') +
              label_stream.get_data() + b'\nQ\n')
    lower.set_data(original + append)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        writer.write(stream)

    after = PdfReader(args.output)
    aft = after.pages[0]['/Resources']['/XObject']
    new_lower = aft['/fzFrm1']['/Resources']['/XObject']['/fullpage'].get_data()
    new_top = aft['/fzFrm0']['/Resources']['/XObject']['/fullpage'].get_data()
    assert new_lower == original + append
    assert new_top == original_top
    assert tuple(after.pages[0].mediabox) == tuple(page.mediabox)
    assert tuple(after.pages[0].cropbox) == tuple(page.cropbox)

    before_doc, after_doc = fitz.open(args.input), fitz.open(args.output)
    before_pix = before_doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    after_pix = after_doc[0].get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    before = np.frombuffer(before_pix.samples, np.uint8).reshape(before_pix.height, before_pix.width, 3)
    after_image = np.frombuffer(after_pix.samples, np.uint8).reshape(after_pix.height, after_pix.width, 3)
    changes = np.any(before != after_image, axis=2)
    yy, xx = np.nonzero(changes)
    assert yy.size, 'The tick must be visible in the rendered result.'
    bbox_px = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
    bbox_pt = [v / 3 for v in bbox_px]
    # This rectangle contains only the added -0.4 tick and label.
    permitted = (np.indices(changes.shape)[1] >= 196 * 3) & (np.indices(changes.shape)[1] <= 246 * 3) & (np.indices(changes.shape)[0] >= 290 * 3) & (np.indices(changes.shape)[0] <= 310 * 3)
    assert not np.any(changes & ~permitted), bbox_pt
    after_doc[0].get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(args.output.with_suffix('.png'))
    audit = dict(input=str(args.input.resolve()), output=str(args.output.resolve()),
                 input_sha256=sha256(args.input.read_bytes()), output_sha256=sha256(args.output.read_bytes()),
                 tick_value=-.4, true_delta_m=-.37, middle_parameter_ylim=[-.4, .26], parameter='delta_m', source_fonts_preserved=True,
                 source_font_size_pt=15.9402242, original_lower_stream_is_exact_prefix=True,
                 top_stream_unchanged=True, page_and_crop_boxes_unchanged=True,
                 all_rendered_changes_confined_to_added_tick=True,
                 changed_pixel_bbox_in_page_points=bbox_pt, changed_pixel_count=int(changes.sum()),
                 other_panels_pixel_identical=True, simulations_run=False)
    args.output.with_name(args.output.stem + '_tick_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
