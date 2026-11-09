"""Finalize the two parameter-panel time labels."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject


def sha(data):
    return hashlib.sha256(data).hexdigest()


def serialize(ops, reader):
    content = ContentStream(None, reader)
    content.operations = ops
    return content.get_data()


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
    original_lower, original_top = lower.get_data(), top.get_data()
    ops = ContentStream(lower, reader).operations
    replacements = {}
    audit_labels = []
    for old_x, axis_x in [(250.506453, 260.4782145687), (452.036109, 462.0078705339)]:
        hits = []
        for start, (_, op) in enumerate(ops):
            if op != b'BT':
                continue
            end = start + 1
            while ops[end][1] != b'ET':
                end += 1
            block = ops[start:end + 1]
            first_td = next((i for i, (_, op) in enumerate(block) if op == b'Td'), None)
            if first_td is None:
                continue
            xx, yy = map(float, block[first_td][0])
            if abs(xx - old_x) < 1e-6 and abs(yy - 25.4793417) < 1e-6:
                hits.append((start, end, block, first_td))
        assert len(hits) == 1, 'Expected exactly one original 0.0 time label per panel.'
        start, end, block, first_td = hits[0]
        strings = [str(a[0]) for a, op in block if op == b'Tj']
        assert strings == ['0', ':', '0']  # CMMI12 ':' glyph is the math decimal point.
        assert str(block[1][0][0]) == '/F1' and abs(float(block[1][0][1]) - 15.9402242) < 1e-7
        shortened = copy.deepcopy(block[:4]) + [([], b'ET')]
        # Match the exact centered single-digit tick in the neighboring input panel.
        digit_advance = 7.803987
        new_x = axis_x - digit_advance / 2.
        shortened[first_td] = ([FloatObject(new_x), FloatObject(25.4793417)], b'Td')
        replacements[start] = (end, shortened)
        audit_labels.append(dict(old='0.0', new='0', axis_x=axis_x, new_text_x=new_x))
    result = []
    cursor = 0
    while cursor < len(ops):
        if cursor in replacements:
            end, block = replacements[cursor]
            result.extend(block)
            cursor = end + 1
        else:
            result.append(ops[cursor])
            cursor += 1
    lower.set_data(serialize(result, reader))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        writer.write(stream)
    check = PdfReader(args.output)
    xafter = check.pages[0]['/Resources']['/XObject']
    new_lower = xafter['/fzFrm1']['/Resources']['/XObject']['/fullpage']
    new_top = xafter['/fzFrm0']['/Resources']['/XObject']['/fullpage']
    assert new_top.get_data() == original_top
    assert new_lower.get_data() == serialize(result, reader)
    assert tuple(check.pages[0].mediabox) == tuple(page.mediabox)
    assert tuple(check.pages[0].cropbox) == tuple(page.cropbox)
    before_doc, after_doc = fitz.open(args.input), fitz.open(args.output)
    before_pix = before_doc[0].get_pixmap(matrix=fitz.Matrix(3,3), alpha=False)
    after_pix = after_doc[0].get_pixmap(matrix=fitz.Matrix(3,3), alpha=False)
    b = np.frombuffer(before_pix.samples,np.uint8).reshape(before_pix.height,before_pix.width,3)
    a = np.frombuffer(after_pix.samples,np.uint8).reshape(after_pix.height,after_pix.width,3)
    changed = np.any(a != b,axis=2)
    y,x = np.indices(changed.shape)
    rectangles = [[230,304,252,321],[432,304,454,321]]
    permitted = np.zeros(changed.shape,dtype=bool)
    changed_counts = []
    for lo_x,lo_y,hi_x,hi_y in rectangles:
        mask = (x>=lo_x*3)&(x<hi_x*3)&(y>=lo_y*3)&(y<hi_y*3)
        permitted |= mask
        changed_counts.append(int((changed&mask).sum()))
    assert all(v>0 for v in changed_counts)
    assert not np.any(changed & ~permitted), 'Unexpected visual change outside the two time labels.'
    after_doc[0].get_pixmap(matrix=fitz.Matrix(2.5,2.5),alpha=False).save(args.output.with_suffix('.png'))
    audit = dict(input=str(args.input.resolve()),output=str(args.output.resolve()),
                 input_sha256=sha(args.input.read_bytes()),output_sha256=sha(args.output.read_bytes()),
                 changed_labels=audit_labels, label_font='CMR12', label_font_size_pt=15.9402242,
                 all_other_lower_operators_unchanged=True,top_stream_unchanged=True,
                 page_and_crop_boxes_unchanged=True, two_label_regions_page_pt=rectangles,
                 changed_pixels_per_label=changed_counts, all_other_pixels_identical=True,
                 mass_minus_point_four_tick_unchanged=True, simulations_run=False)
    args.output.with_name(args.output.stem+'_zero_tick_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
