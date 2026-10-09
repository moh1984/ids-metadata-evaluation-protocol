#!/usr/bin/env python3
"""
Cross-check the manuscript, the response letter and the repository.

Every round of this revision has produced the same class of defect: a number or
a claim fixed in one file and left stale in another. Tracking that by hand has
failed repeatedly, so it is checked mechanically here instead.

    python audit.py <manuscript.docx-dir> <response_letter.md> <repo-dir>

Exits non-zero if any check fails, and prints what to fix.
"""

import os
import re
import sys

PARA = re.compile(r'<w:p(?:\s[^>]*)?>.*?</w:p>', re.S)
TBL = re.compile(r'<w:tbl>.*?</w:tbl>', re.S)

problems = []


def fail(check, detail):
    problems.append((check, detail))


def ptext(p):
    return re.sub(r'\s+', ' ', ''.join(re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p))).strip()


def load(doc_dir):
    return open(os.path.join(doc_dir, 'word', 'document.xml'), encoding='utf-8').read()


def main(doc_dir, letter_path, repo_dir):
    x = load(doc_dir)
    letter = open(letter_path, encoding='utf-8').read()
    paras_m = list(PARA.finditer(x))
    paras = [ptext(m.group(0)) for m in paras_m]
    body = ' '.join(t for t in paras if not re.match(r'^\[\d+\] [A-ZÇ]', t))

    # ---- captions appear in ascending order -------------------------------
    for kind, pat in (('Table', r'^Table (\d+)([a-z]?)\.'), ('Figure', r'^Figure\. (\d+) ')):
        seen = []
        for t in paras:
            m = re.match(pat, t)
            if m and m.group(0) not in [s[0] for s in seen]:
                seen.append((m.group(0), int(m.group(1))))
        nums = [n for _, n in seen]
        if nums != sorted(nums):
            fail(f'{kind} order', f'captions appear as {nums}')

    # ---- every caption is referenced, and no reference dangles ------------
    caps = {m.group(1) + m.group(2) for t in paras
            for m in [re.match(r'^Table (\d+)([a-z]?)\.', t)] if m}
    refs = set(re.findall(r'\bTable (\d+[a-z]?)\b', body))
    if caps - refs:
        fail('Table references', f'never cited: {sorted(caps - refs)}')
    if refs - caps:
        fail('Table references', f'cited but absent: {sorted(refs - caps)}')

    fcaps = {int(m.group(1)) for t in paras for m in [re.match(r'^Figure\. (\d+) ', t)] if m}
    frefs = {int(n) for n in re.findall(r'\bFigs?\. (\d+)', body)}
    frefs |= {int(n) for n in re.findall(r'\bFigs?\. \d+ and (\d+)', body)}
    if fcaps - frefs:
        fail('Figure references', f'never cited: {sorted(fcaps - frefs)}')
    if frefs - fcaps:
        fail('Figure references', f'cited but absent: {sorted(frefs - fcaps)}')

    # ---- section cross-references point at sections that exist ------------
    heads = {m.group(1) for t in paras for m in [re.match(r'^(\d+\.\d+) [A-Z]', t)] if m}
    for ref in set(re.findall(r'Section (\d+\.\d+)', body)):
        if ref not in heads:
            fail('Section reference', f'Section {ref} is cited but no such heading exists')

    # ---- citations ascending, every reference cited ------------------------
    first = {}
    for m in re.finditer(r'\[(\d+(?:\s*[-,]\s*\d+)*)\]', body):
        for part in re.split(r'\s*,\s*', m.group(1)):
            for n in ([int(v) for v in part.split('-')] if '-' in part else [int(part)]):
                first.setdefault(n, m.start())
    listed = {int(m.group(1)) for t in paras for m in [re.match(r'^\[(\d+)\] [A-ZÇ]', t)] if m}
    if sorted(first, key=lambda n: first[n]) != list(range(1, len(listed) + 1)):
        fail('Citations', 'first appearances are not in ascending order')
    if listed != set(first):
        fail('Citations', f'listed but uncited: {sorted(listed - set(first))}; '
                          f'cited but unlisted: {sorted(set(first) - listed)}')

    # ---- the letter must not point at tables the manuscript renamed --------
    cap_titles = {}
    for t in paras:
        m = re.match(r'^Table (\d+[a-z]?)\. (.+)$', t)
        if m:
            cap_titles.setdefault(m.group(1), m.group(2))
    for n in set(re.findall(r'\bTable (\d+[a-z]?)\b', letter)):
        if n not in cap_titles:
            fail('Letter', f'refers to Table {n}, which does not exist in the manuscript')

    # ---- phrases that must not survive anywhere ---------------------------
    banned = {
        'operational base rate': 'the reviewers asked for this term to go',
        'true base rate': 'same',
        'accepts neither class weights nor sample weights': 'MLPClassifier does accept sample_weight',
        'overlap almost entirely': 'overlap is not a test of a paired difference',
        'all results here use the default 0.5 cutoff': 'contradicts the threshold table',
    }
    for phrase, why in banned.items():
        if phrase in x:
            fail('Manuscript wording', f'{phrase!r} still present — {why}')
        # the letter may quote a banned phrase in order to say it was removed
        for m in re.finditer(re.escape(phrase), letter):
            around = letter[max(0, m.start() - 90):m.end() + 90]
            if not re.search(r'(removed|replaced|no longer|withdraw|previously)', around, re.I):
                fail('Letter wording', f'{phrase!r} used as a claim — {why}')
                break

    # ---- the editor requires every letter in a table at 10 pt or more -----
    cap_at = [(m.start(), re.match(r'^Table (\d+[a-z]?)\.', ptext(m.group(0))))
              for m in PARA.finditer(x)]
    cap_at = [(s, m.group(1)) for s, m in cap_at if m]
    for m in TBL.finditer(x):
        before = [n for s, n in cap_at if s < m.start()]
        label = before[-1] if before else '?'
        sizes = {int(v) / 2 for v in re.findall(r'<w:sz w:val="(\d+)"/>', m.group(0))}
        small = sorted(p for p in sizes if p < 10)
        if small:
            fail('Table font size', f'Table {label} contains {small} pt text; the editor requires 10 pt')

    # ---- a heading with nothing under it but a table reads as an orphan ---
    heads = [(i, ptext(m.group(0))) for i, m in enumerate(paras_m)
             if re.match(r'^\d+(\.\d+)? ?[A-Z]', ptext(m.group(0)))
             and len(ptext(m.group(0))) < 62]
    tbl_spans = [(m.start(), m.end()) for m in TBL.finditer(x)]
    for k, (i, h) in enumerate(heads):
        stop = heads[k + 1][0] if k + 1 < len(heads) else len(paras_m)
        words = 0
        for j in range(i + 1, stop):
            t = ptext(paras_m[j].group(0))
            if not t or any(a <= paras_m[j].start() < b for a, b in tbl_spans):
                continue
            if re.match(r'^(Table \d+\.|Figure\. \d|Note:|\[\d+\])', t):
                continue
            words += len(t.split())
        if words < 15:
            fail('Bare section', f'{h[:50]!r} has {words} words of its own prose')

    # ---- table notes are text too, and the editor's rule covers them ------
    for m in PARA.finditer(x):
        t = ptext(m.group(0))
        if not t.startswith('Note:'):
            continue
        sizes = {int(v) / 2 for v in re.findall(r'<w:sz w:val="(\d+)"/>', m.group(0))}
        small = sorted(p for p in sizes if p < 10)
        if small:
            fail('Table note size', f'{small} pt in: {t[:55]}')

    # ---- figures are drawn at final print size, so their point sizes are
    #      the point sizes on the page; anything under 10 breaks the rule ---
    for script in ('make_figures.py', 'prevalence_sensitivity.py'):
        p = os.path.join(repo_dir, 'src', script)
        if not os.path.exists(p):
            continue
        code = open(p, encoding='utf-8').read()
        small = sorted({float(v) for v in re.findall(r'fontsize=([\d.]+)', code)
                        if float(v) < 10})
        if small:
            fail('Figure font size', f'{script} draws text at {small} pt')

    # ---- the repository must not describe a version of the paper that is
    #      no longer the paper -------------------------------------------
    # the 4% figure is inherited from simulated data; no wording may call it
    # real, true or operational, in any file
    stale_repo = ['at an Operational Base Rate', 'operational base rate', 'true base rate',
                  'real base rate', 'REAL base rate', 'Realistic evaluation base rate',
                  'realistic base rate', 'true 24:1', 'real 24:1',
                  'all four stages', 'to all four', 'Table 11, Fig. 10']
    for root, _, names in os.walk(repo_dir):
        if any(skip in root for skip in ('.git', '__pycache__', 'results')):
            continue
        for name in names:
            if name == 'audit.py':          # this file defines the banned phrases
                continue
            if not name.endswith(('.md', '.py', '.sh', '.cff', '.ipynb')):
                continue
            path = os.path.join(root, name)
            text = open(path, encoding='utf-8', errors='replace').read()
            for phrase in stale_repo:
                if phrase in text:
                    fail('Repository wording',
                         f'{os.path.relpath(path, repo_dir)} still says {phrase!r}')

    # ---- figure numbers quoted in the letter must match the manuscript ----
    for n in {int(v) for v in re.findall(r'\bFigs?\. (\d+)', letter)}:
        if n not in fcaps:
            fail('Letter', f'refers to Fig. {n}, which does not exist in the manuscript')

    # ---- claims in the letter that the manuscript has since narrowed ------
    for phrase, why in {
        'every ensemble-versus-baseline PR-AUC difference excludes zero':
            'only the gradient-boosting comparisons were computed',
        'SMOTE applied correctly':
            'the implementation is acknowledged as simplified under Table 11',
    }.items():
        if phrase in letter:
            fail('Letter claim', f'{phrase!r} — {why}')
        if phrase in x:
            fail('Manuscript claim', f'{phrase!r} — {why}')

    # ---- doubled connectives left by successive edits ---------------------
    for m in re.finditer(r'\bwhile\b[^.]{0,120}\bwhile\b', body):
        fail('Prose', f'"while ... while" in: {m.group(0)[:90]}')

    # ---- every table and figure number quoted in the repository must be
    #      one the manuscript actually has, and must name the same thing ----
    titles = {n: t.lower() for n, t in cap_titles.items()}
    fig_titles = {}
    for t in paras:
        m = re.match(r'^Figure\. (\d+) (.+)$', t)
        if m:
            fig_titles.setdefault(int(m.group(1)), m.group(2).lower())
    for root, _, names in os.walk(repo_dir):
        if any(skip in root for skip in ('.git', '__pycache__', 'results')):
            continue
        for name in names:
            if name == 'audit.py' or not name.endswith(('.md', '.sh', '.ipynb', '.py')):
                continue
            path = os.path.join(root, name)
            text = open(path, encoding='utf-8', errors='replace').read()
            rel = os.path.relpath(path, repo_dir)
            for n in set(re.findall(r'\bTable (\d+[a-z]?)\b', text)):
                if n not in titles:
                    fail('Repository numbering',
                         f'{rel} cites Table {n}, which the manuscript does not have')
            for n in {int(v) for v in re.findall(r'\bFig(?:ure)?\. (\d+)\b', text)}:
                if n not in fig_titles:
                    fail('Repository numbering',
                         f'{rel} cites Fig. {n}, which the manuscript does not have')

    # ---- repository files the manuscript promises -------------------------
    tables = os.path.join(repo_dir, 'results', 'tables')
    figures = os.path.join(repo_dir, 'results', 'figures')
    if os.path.isdir(tables):
        for need in ('table12_bootstrap_intervals.csv', 'table12b_paired_differences.csv'):
            p = os.path.join(tables, need)
            if not os.path.exists(p):
                fail('Repository', f'{need} is missing')
            elif 'Gradient Boosting' not in open(p, encoding='utf-8').read():
                fail('Repository', f'{need} has no gradient-boosting row')
    if os.path.isdir(figures):
        n_fig = len([f for f in os.listdir(figures) if f.endswith('.png')])
        if n_fig != len(fcaps):
            fail('Repository', f'{n_fig} figure files against {len(fcaps)} captions')

    # ---- report ------------------------------------------------------------
    if problems:
        print(f'{len(problems)} problem(s):\n')
        for check, detail in problems:
            print(f'  [{check}] {detail}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
