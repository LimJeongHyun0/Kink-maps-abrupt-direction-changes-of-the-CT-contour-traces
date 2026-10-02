"""Extract manual red-trace polylines (PowerPoint connector 'line' shapes) from each
[Curvature][TxDy]_raw.pptx, together with the slice image each slide shows.

Output: /home/claude/traces.json
  { set_name: { 'label': str, 'slides': [ { 'idx': 1..10, 'image': path,
        'pic': {'off':[x,y],'ext':[cx,cy]}, 'lines': [[u0,v0,u1,v1], ...] } ] } }
  (u,v) are normalized picture coordinates (0..1 across the picture box).
"""
import os, re, json
from xml.etree import ElementTree as ET

ROOT = '/home/claude/raw/unpacked'
NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}

def slide_order(d):
    pres = ET.parse(os.path.join(d, 'ppt/presentation.xml')).getroot()
    rels = ET.parse(os.path.join(d, 'ppt/_rels/presentation.xml.rels')).getroot()
    rid2target = {r.get('Id'): r.get('Target') for r in rels}
    order = []
    for sid in pres.find('p:sldIdLst', NS):
        rid = sid.get('{%s}id' % NS['r'])
        order.append(os.path.normpath(os.path.join(d, 'ppt', rid2target[rid])))
    return order

def parse_slide(path):
    root = ET.parse(path).getroot()
    relp = os.path.join(os.path.dirname(path), '_rels', os.path.basename(path) + '.rels')
    rid2target = {r.get('Id'): r.get('Target') for r in ET.parse(relp).getroot()}
    pics, lines, texts, other = [], [], [], []
    for t in root.iter('{%s}t' % NS['a']):
        if t.text: texts.append(t.text)
    for pic in root.iter('{%s}pic' % NS['p']):
        blip = pic.find('.//a:blip', NS)
        rid = blip.get('{%s}embed' % NS['r'])
        xfrm = pic.find('.//a:xfrm', NS)
        off = xfrm.find('a:off', NS); ext = xfrm.find('a:ext', NS)
        src = pic.find('.//a:srcRect', NS)
        pics.append({
            'image': os.path.normpath(os.path.join(os.path.dirname(path), rid2target[rid])),
            'off': [int(off.get('x')), int(off.get('y'))],
            'ext': [int(ext.get('cx')), int(ext.get('cy'))],
            'srcRect': None if src is None else {k: int(src.get(k, 0)) for k in ('l', 't', 'r', 'b')},
        })
    for cx in root.iter('{%s}cxnSp' % NS['p']):
        sppr = cx.find('p:spPr', NS)
        geom = sppr.find('a:prstGeom', NS)
        col = sppr.find('a:ln/a:solidFill/a:srgbClr', NS)
        xfrm = sppr.find('a:xfrm', NS)
        off = xfrm.find('a:off', NS); ext = xfrm.find('a:ext', NS)
        x, y = int(off.get('x')), int(off.get('y'))
        w, h = int(ext.get('cx')), int(ext.get('cy'))
        fh = xfrm.get('flipH') == '1'; fv = xfrm.get('flipV') == '1'
        x0, x1 = (x + w, x) if fh else (x, x + w)
        y0, y1 = (y + h, y) if fv else (y, y + h)
        rec = {'geom': geom.get('prst') if geom is not None else None,
               'color': col.get('val') if col is not None else None,
               'p0': [x0, y0], 'p1': [x1, y1]}
        lines.append(rec)
    for sp in root.iter('{%s}sp' % NS['p']):
        g = sp.find('.//a:prstGeom', NS); cg = sp.find('.//a:custGeom', NS)
        other.append(g.get('prst') if g is not None else ('custGeom' if cg is not None else '?'))
    return pics, lines, texts, other

out = {}
for d in sorted(os.listdir(ROOT)):
    dpath = os.path.join(ROOT, d)
    m = re.match(r'\[(\w+)\]\[(\w+)\]_raw', d)
    curv, spec = m.group(1), m.group(2)
    name = '%s_%s' % (curv, spec)
    slides = slide_order(dpath)
    recs = []
    title = None
    for i, s in enumerate(slides):
        pics, lines, texts, other = parse_slide(s)
        if i == 0:
            title = ' '.join(texts)
            if pics or lines:
                print('  NOTE: title slide has pics/lines', len(pics), len(lines))
            continue
        if len(pics) != 1:
            print('  WARN %s %s: %d pictures' % (name, os.path.basename(s), len(pics)))
        pic = pics[0]
        bad = [l for l in lines if l['geom'] != 'line' or l['color'] != 'FF0000']
        if bad:
            print('  WARN %s %s: %d non-red/non-line shapes: %s' % (name, os.path.basename(s), len(bad),
                  set((b['geom'], b['color']) for b in bad)))
        if other:
            print('  NOTE %s %s: other sp shapes: %s' % (name, os.path.basename(s), other))
        ox, oy = pic['off']; cx, cy = pic['ext']
        segs = []
        for l in lines:
            if l['geom'] != 'line' or l['color'] != 'FF0000':
                continue
            u0 = (l['p0'][0] - ox) / cx; v0 = (l['p0'][1] - oy) / cy
            u1 = (l['p1'][0] - ox) / cx; v1 = (l['p1'][1] - oy) / cy
            segs.append([u0, v0, u1, v1])
        recs.append({'idx': i, 'slide': os.path.basename(s), 'image': pic['image'],
                     'pic': {'off': pic['off'], 'ext': pic['ext'], 'srcRect': pic['srcRect']},
                     'lines': segs})
    out[name] = {'curvature': curv, 'specimen': spec, 'title': title, 'slides': recs}
    nl = [len(r['lines']) for r in recs]
    print('%-14s title=%-16s slides=%d  segments/slide=%s' % (name, title, len(recs), nl))

json.dump(out, open('/home/claude/traces.json', 'w'), indent=1)
print('saved traces.json')
