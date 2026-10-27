#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
check_deliverables.py -- one-shot pre-submission check for the EFM package.

Run after every change to manuscript_efm.tex / supplementary_efm.tex /
highlights_efm.tex / cover_letter_efm.tex (i.e. after build_papers.ps1):

    python manuscript/check_deliverables.py

Reports, per document: Overfull hbox / vbox (with sizes), Underfull count,
undefined references, page count. Plus the journal-facing constraints:
abstract word count (judge on the paste-block/whitespace basis; a pdftotext
extraction counts higher because math such as "$\xi_0\ge3$" splits into several
tokens), tex<->md mirror drift, title rendering in all four PDFs, Highlights
line length, SI cross-reference integrity (every hard-written S reference is
matched against the numbers the SI PDF actually renders), leftover internal-meta
phrases, and the graphical abstract (size + whether its text still carries the
current title).

Exit code is non-zero if any hard check fails, so the script can gate a
commit or an upload.
"""
import difflib
import io
import os
import re
import subprocess
import sys
import unicodedata

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, 'submission_EFM')
SI = os.path.join(PKG, 'supplementary')

DOCS = [
    ('main', os.path.join(PKG, 'manuscript_efm')),
    ('SI', os.path.join(SI, 'supplementary_efm')),
    ('highlights', os.path.join(PKG, 'highlights_efm')),
    ('cover letter', os.path.join(PKG, 'cover_letter_efm')),
]
ABSTRACT_LIMIT = 250
HIGHLIGHT_LIMIT = 85

FAILS = []
WARNS = []


def fail(msg):
    FAILS.append(msg)
    print('  [FAIL] ' + msg)


def warn(msg):
    WARNS.append(msg)
    print('  [warn] ' + msg)


def ok(msg):
    print('  [ ok ] ' + msg)


def read(p):
    return io.open(p, encoding='utf-8', errors='replace').read()


def pdftext(p):
    t = subprocess.run(['pdftotext', '-layout', p, '-'], capture_output=True)
    if t.returncode != 0:
        fail('pdftotext failed for %s' % p)
        return ''
    t = t.stdout.decode('utf-8', 'replace')
    # lineno writes line numbers into the text stream; with -layout they sit at the
    # start of each physical line and break exact-phrase matching.
    return re.sub(r'(?m)^\s*\d+\s+', ' ', t)


flat = lambda s: re.sub(r'\s+', ' ', s)

print('=' * 78)
print('EFM deliverable check -- %s' % PKG)
print('=' * 78)

# ---------------------------------------------------------------- 1. logs
print('\n[1] LaTeX logs (Overfull / Underfull / undefined / pages)')
missing_logs = []
for tag, base in DOCS:
    log = base + '.log'
    if not os.path.exists(log):
        missing_logs.append(tag)
        continue
    t = read(log)
    hb = re.findall(r'Overfull \\hbox \(([\d.]+)pt', t)
    vb = re.findall(r'Overfull \\vbox \(([\d.]+)pt', t)
    uf = len(re.findall(r'Underfull \\(?:hbox|vbox)', t))
    und = len([l for l in t.splitlines() if 'undefined' in l.lower()])
    err = len([l for l in t.splitlines() if l.startswith('! ')])
    pm = re.search(r'Output written on .+\((\d+) pages', t)
    pages = pm.group(1) if pm else '?'
    line = ('%-12s pages=%-4s hbox=%d%s vbox=%d%s underfull=%-4d undefined=%d errors=%d'
            % (tag, pages, len(hb), hb[:4], len(vb), vb[:4], uf, und, err))
    if err or und or hb or vb:
        fail(line)
    else:
        print('  [ ok ] ' + line)
if missing_logs:
    warn('no .log for %s -> not built here (the published archive git-ignores *.log); '
         'layout metrics skipped, run manuscript/build_papers.ps1 first' 
         % ', '.join(missing_logs))

# ------------------------------------------------------- 2. abstract words
print('\n[2] Abstract (EFM: one paragraph, <= %d words)' % ABSTRACT_LIMIT)
main_tex = read(os.path.join(PKG, 'manuscript_efm.tex'))
m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', main_tex, re.S)
if not m:
    fail('no \\begin{abstract} found')
    ab_raw = ''
else:
    ab_raw = m.group(1)
    if re.search(r'\n\s*\n', ab_raw.strip()):
        fail('abstract is not a single paragraph (blank line inside)')
    else:
        ok('single paragraph')


def strip_tex(s):
    s = re.sub(r'\\(?:emph|textbf|textit|mathrm|text)\{(.*?)\}', r'\1', s, flags=re.S)
    s = re.sub(r'\$([^$]*)\$', r'\1', s)
    s = re.sub(r'\\[a-zA-Z]+', ' ', s)
    s = s.replace('{', ' ').replace('}', ' ').replace('\\', ' ').replace('%', '')
    s = s.replace('---', '-').replace('~', ' ')
    return flat(s).strip()


tex_plain = strip_tex(ab_raw)
tok_tex = len(tex_plain.split())

md_path = os.path.join(PKG, 'title_abstract_keywords_efm.md')
md_abs = ''
md_tok = None
if os.path.exists(md_path):
    md = read(md_path)
    try:
        md_abs = flat(re.sub(r'\s+', ' ', md.split('# Abstract')[1].split('---')[0])).strip()
    except Exception:
        fail('could not locate "# Abstract" block in title_abstract_keywords_efm.md')
if md_abs:
    md_tok = len(md_abs.split())
    # counting basis that ignores pure-symbol tokens (>=, dashes, tau bars)
    alpha = len([w for w in md_abs.split() if re.search('[A-Za-z]', w)])
    print('  paste-block tokens (editor basis) : %d' % md_tok)
    print('  tokens containing letters         : %d' % alpha)
    print('  tex markup-stripped tokens        : %d' % tok_tex)
else:
    print('  paste-block md not present (it is intentionally not published); '
          'judging on the tex basis')

basis = md_tok if md_tok is not None else tok_tex
label = 'paste block' if md_tok is not None else 'tex markup-stripped'
if basis > ABSTRACT_LIMIT:
    fail('abstract exceeds %d words (%s = %d)' % (ABSTRACT_LIMIT, label, basis))
else:
    ok('within limit (%s = %d, headroom %d)' % (label, basis, ABSTRACT_LIMIT - basis))
if md_tok is not None and abs(md_tok - tok_tex) > 12:
    warn('tex and md token counts differ by %d (math expansion); check mirror'
         % abs(md_tok - tok_tex))

# mirror drift: compare ASCII alphanumeric skeletons (subscript digits folded via NFKD,
# non-ASCII glyphs such as xi/tau/macron dropped on both sides)
if not md_abs:
    warn('mirror check skipped: no paste-block md in this copy')
else:
    ka = re.sub(r'[^A-Za-z0-9]', '', re.sub(r'\\[a-zA-Z]+', '', tex_plain).replace(' ', ''))
    kb = re.sub(r'[^\x00-\x7F]', '', unicodedata.normalize('NFKD', md_abs))
    kb = re.sub(r'[^A-Za-z0-9]', '', kb)
    if ka != kb:
        sm = difflib.SequenceMatcher(None, ka, kb)
        diffs = [(t, ka[i1:i2], kb[j1:j2])
                 for t, i1, i2, j1, j2 in sm.get_opcodes() if t != 'equal']
        fail('tex <-> md abstract mirror drifts (%d place(s)):' % len(diffs))
        for t, x, y in diffs[:6]:
            print('         [%s] tex=%r  md=%r' % (t, x[:70], y[:70]))
    else:
        ok('tex and md abstracts are letter-identical')

# ------------------------------------------------------ 3. title rendering
print('\n[3] Title consistency (tex \\title vs rendered PDFs vs md)')
tex_title = flat(re.search(r'\\title\{(.*?)\}', main_tex, re.S).group(1)).strip()
print('  tex: %s' % tex_title)
for tag, base in DOCS:
    if tag == 'SI':
        continue          # the SI carries its own section heads, not the paper title
    p = base + '.pdf'
    if os.path.exists(p):
        hit = tex_title in flat(pdftext(p))
        (ok if hit else fail)('%-12s title rendered verbatim' % tag)
if os.path.exists(md_path):
    hit = tex_title in flat(read(md_path))
    (ok if hit else fail)('%-12s contains the title' % 'md block')
for tag, base in DOCS:
    p = base + '.pdf'
    if os.path.exists(p) and 'homogenization bias and a mechanism-keyed' in flat(pdftext(p)):
        fail('%s still carries the superseded title fragment' % tag)

# --------------------------------------------------------- 4. highlights
print('\n[4] Highlights (3-5 bullets, each <= %d chars, separate editable file)'
      % HIGHLIGHT_LIMIT)
htxt = os.path.join(PKG, 'highlights_efm.txt')
if os.path.exists(htxt):
    lines = [l.strip() for l in read(htxt).splitlines() if l.strip()]
    for k, l in enumerate(lines, 1):
        n = len(l)
        if n > HIGHLIGHT_LIMIT:
            fail('bullet %d is %d chars: %s' % (k, n, l[:60]))
        else:
            ok('bullet %d: %2d chars' % (k, n))
    if not 3 <= len(lines) <= 5:
        fail('need 3-5 highlights, found %d' % len(lines))
    if os.path.exists(os.path.join(PKG, 'highlights_efm.pdf')):
        hp = flat(pdftext(os.path.join(PKG, 'highlights_efm.pdf')))
        for l in lines:
            if flat(l) not in hp:
                fail('bullet not present in highlights_efm.pdf: %s' % l[:50])
else:
    fail('highlights_efm.txt missing (EFM requires an editable file)')

# ------------------------------------------- 5. SI cross-reference integrity
print('\n[5] SI cross-reference integrity')
sipdf = os.path.join(SI, 'supplementary_efm.pdf')
if os.path.exists(sipdf):
    st = pdftext(sipdf)
    st = st.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    stj = re.sub(r'\s*\n\s*', ' ', st)
    have = set(int(n) for n in re.findall(r'\bTable\s+S(\d+)[:.]', stj))
    havef = set(int(n) for n in re.findall(r'\bFigure\s+S(\d+)[:.]', stj))
    # SI section numbers are hand-written in the SI source (\section*{S7. ...}), not produced
    # by a counter, so the source is the authority; the PDF prose contains too many
    # "S7."-shaped collisions (caption ends, table cells) to mine reliably.
    haves = set(int(n) for n in
                re.findall(r'\\section\*?\{\s*S(\d+)\.', read(os.path.join(SI, 'supplementary_efm.tex'))))
    print('  SI renders Table S%s-S%s (%d numbered), Figure S%s-S%s (%d)'
          % (min(have), max(have), len(have), min(havef), max(havef), len(havef)))
    print('  SI sections in source: %s' % sorted(haves))
    if len(have) != max(have):
        warn('SI table numbers are not contiguous: gaps at %s'
             % sorted(set(range(1, max(have) + 1)) - have))
    main = read(os.path.join(PKG, 'manuscript_efm.tex'))
    miss = []
    cnt = 0
    for i, l in enumerate(main.splitlines(), 1):
        if l.strip().startswith('%'):
            continue
        for kind, pool in (('Table', have), ('Fig', havef), ('Section', haves)):
            for mm in re.finditer(kind + r'(?:ure|ion)?s?[~ ]+S(\d+)', l):
                cnt += 1
                if int(mm.group(1)) not in pool:
                    miss.append('L%d %s S%s' % (i, kind, mm.group(1)))
        # continuation forms actually used in this manuscript: "Tables~S25 and S26"
        for mm in re.finditer(r'\band\s+S(\d+)\b', l):
            n = int(mm.group(1))
            cnt += 1
            if not (n in have or n in havef or n in haves):
                miss.append('L%d continuation S%d' % (i, n))
    print('  hard-written Table/Fig/Section S references in main text: %d' % cnt)
    if miss:
        fail('dangling SI references: ' + ', '.join(miss[:10]))
    else:
        ok('every hard-written Table/Fig S reference resolves')
else:
    fail('supplementary_efm.pdf missing')

# ------------------------------------------- 6. internal-meta leftovers
print('\n[6] Deliverable hygiene (no internal meta in what the editor sees)')
BANNED = ['moved from the main text', 'earlier draft', 'to be added', 'placeholder',
          'TODO', 'for length', 'ported from the IJMS build',
          'Journal of Materials Science\\']
for tag, base in DOCS:
    pdf = base + '.pdf'
    if not os.path.exists(pdf):
        continue
    t = flat(pdftext(pdf))
    bad = [b for b in BANNED if b not in ('Journal of Materials Science\\',) and b in t]
    if bad:
        fail('%s PDF contains internal meta: %s' % (tag, bad))
    else:
        ok('%-12s PDF free of internal-meta phrases' % tag)
    # journal-name accuracy: IJMS means Mechanical Sciences
    t2 = pdftext(pdf) if tag == 'cover letter' else ''
    if 'Journal of Materials Science' in flat(t2):
        fail('cover letter names the wrong source journal (should be Mechanical Sciences)')

# --------------------------------------------------- 7. graphical abstract
print('\n[7] Graphical abstract (>= 531 x 1328 px h x w; text follows the current title)')
GA = os.path.join(PKG, 'graphical_abstract', 'graphical_abstract_efm')
if not os.path.exists(GA + '.png'):
    fail('graphical_abstract_efm.png missing (regenerate with make_graphical_abstract_efm.py)')
else:
    try:
        from PIL import Image
        w, h = Image.open(GA + '.png').size
        if h >= 531 and w >= 1328:
            ok('PNG %d x %d px (h x w), minimum is 531 x 1328' % (h, w))
        else:
            fail('PNG %d x %d px (h x w) is below the 531 x 1328 minimum' % (h, w))
    except ImportError:
        warn('PIL unavailable: PNG size not checked')
    for ext in ('.pdf', '.tif', '.eps'):
        if not os.path.exists(GA + ext):
            warn('graphical abstract %s missing' % ext)
if os.path.exists(GA + '.pdf'):
    gat = flat(pdftext(GA + '.pdf'))
    if tex_title in gat:
        ok('GA text carries the current title verbatim')
    else:
        fail('GA text does not contain the current title (stale figure? rerun the generator)')

print('\n' + '=' * 78)
print('hard failures: %d   warnings: %d' % (len(FAILS), len(WARNS)))
print('=' * 78)
sys.exit(1 if FAILS else 0)
