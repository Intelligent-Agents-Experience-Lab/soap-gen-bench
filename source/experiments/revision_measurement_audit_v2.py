"""Judge-free numeric audit, v2.

Flags a note when it states a measurement (value with a unit, or an X/Y pair such as a blood
pressure) that cannot be found in the conversation. v2 normalizes spoken numbers in the
conversation ("one twenty over eighty", "ninety eight point six", "two and a half") and skips
X/Y pairs that read as dates. It is a screening measure: a flag can be a legitimate inference
(e.g. a unit conversion), so flagged notes are reviewed by hand before being called errors.

    python -m source.experiments.revision_measurement_audit_v2 aci data/external/aci-bench
    python -m source.experiments.revision_measurement_audit_v2 jsonl FILE.jsonl
"""
import csv, json, re, sys
csv.field_size_limit(10**9)
UNIT = r'(?:mg/dl|mg|mcg|g/dl|mmhg|bpm|beats per minute|breaths per minute|%|°f|°c|degrees|kg|lbs?|pounds|cm|mm|ml|units|mmol/l|ng/ml|iu)'
PAT = re.compile(r'\b(\d{2,3})\s*/\s*(\d{2,3})\b|\b(\d+(?:\.\d+)?)\s*' + UNIT + r'(?![a-z])', re.I)
ONES = {w: i for i, w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split())}
TENS = {w: 10 * i for i, w in enumerate('_ _ twenty thirty forty fifty sixty seventy eighty ninety'.split()) if w != '_'}
def words_to_digits(text):
    """Convert spoken numbers to digits: 'one twenty over seventy' -> 120/70, 'two and a half' -> 2.5,
    'ninety eight point seven' -> 98.7, '1,000' -> 1000, 'a thousand' -> 1000."""
    s = text.lower()
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)
    toks = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[^a-z\d\s]", s)
    def val(t):
        return ONES.get(t, TENS.get(t))
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if val(t) is None and not (t == 'a' and i + 1 < len(toks) and toks[i+1] in ('hundred', 'thousand')):
            out.append(t); i += 1; continue
        total, cur, j = 0, 0, i
        if t == 'a':
            cur = 1; j += 1
        while j < len(toks):
            w = toks[j]; v = val(w)
            if v is not None:
                # 'one twenty' / 'one twenty eight' hundreds shorthand (vitals)
                if 1 <= cur <= 9 and j > i and val(toks[j-1]) is not None and val(toks[j-1]) < 10 and v >= 10:
                    cur = cur * 100 + v
                elif cur % 10 == 0 and v < 10 and (cur % 100 >= 20 or cur >= 100 and cur % 100 == 0 or cur == 0):
                    cur += v
                elif cur % 100 == 0 and cur >= 100 and v >= 10:
                    cur += v
                elif cur == 0:
                    cur = v
                else:
                    break
            elif w == 'hundred': cur = (cur or 1) * 100
            elif w == 'thousand': total += (cur or 1) * 1000; cur = 0
            elif w == 'and' and j + 1 < len(toks) and val(toks[j+1]) is not None and cur >= 100: pass
            elif w == '-' and j + 1 < len(toks) and val(toks[j+1]) is not None: pass
            else: break
            j += 1
        num = str(total + cur)
        # decimals: 'point seven' / 'point five six'
        if j < len(toks) and toks[j] == 'point':
            k = j + 1; dec = ''
            while k < len(toks) and toks[k] in ONES and ONES[toks[k]] < 10:
                dec += str(ONES[toks[k]]); k += 1
            if dec: num = f'{num}.{dec}'; j = k
        # 'and a half'
        if toks[j:j+3] == ['and', 'a', 'half']:
            num = str(float(num) + 0.5).rstrip('0').rstrip('.'); j += 3
        out.append(num); i = j
    s = ' '.join(out)
    s = re.sub(r'\b(\d{2,3})\s+over\s+(\d{2,3})\b', r'\1/\2 \1 \2', s)
    s = re.sub(r'\b(\d+)\s*\.\s*(\d+)\b', r'\1.\2 \1 \2', s)          # 'ninety eight . seven' -> 98.7
    s = re.sub(r'(?:\bpoint|\ba\s*\.|(?<![\d])\.)\s*(\d+)\b', r'0.\1', s)  # 'point five', 'a . five', '.6' -> 0.5/0.6
    return s
def is_date(a, b):
    return int(a) <= 12 and int(b) <= 31 or int(a) <= 12 and len(b) == 2 and int(b) >= 19
def unsupported(note, conv):
    norm = words_to_digits(conv) + ' ' + conv.lower()
    nums = set(re.findall(r'\d+(?:\.\d+)?', norm))
    bad = []
    for m in PAT.finditer(note):
        if m.group(1):
            a, b = m.group(1), m.group(2)
            if is_date(a, b): continue
            if not (f'{a}/{b}' in norm or (a in nums and b in nums)): bad.append(f'{a}/{b}')
        else:
            v = m.group(3).rstrip('0').rstrip('.') if '.' in m.group(3) else m.group(3)
            if v not in nums and m.group(3) not in nums: bad.append(m.group(0).strip())
    return sorted(set(bad))
if __name__ == '__main__':
    mode, path = sys.argv[1], sys.argv[2]
    if mode == 'aci':
        from pathlib import Path
        files = {'train': 'train.csv', 'valid': 'valid.csv', 'test1': 'clinicalnlp_taskB_test1.csv', 'test2': 'clinicalnlp_taskC_test2.csv', 'test3': 'clef_taskC_test3.csv'}
        allr = []
        for k, f in files.items():
            rows = list(csv.DictReader(open(Path(path) / f, encoding='utf-8')))
            flags = [unsupported(r['note'], r['dialogue']) for r in rows]
            print(k, len(rows), 'flagged', sum(bool(x) for x in flags), f'({sum(bool(x) for x in flags)/len(rows):.0%})')
            allr += [(k, r['encounter_id'], f) for r, f in zip(rows, flags)]
        n = len(allr); b = sum(bool(f) for _, _, f in allr); print('ALL', n, b, f'{b/n:.0%}')
        for x in [x for x in allr if x[2]][:15]: print(x)
    else:  # jsonl with conversation + reference
        rows = [json.loads(l) for l in open(path, encoding='utf-8')]
        key = 'reference' if 'reference' in rows[0] else 'output'
        flags = [unsupported(r[key], r['conversation']) for r in rows]
        b = sum(bool(x) for x in flags); print(path, len(rows), 'flagged', b, f'({b/len(rows):.0%})')
        for r, f in [(r, f) for r, f in zip(rows, flags) if f][:6]: print(r['case_id'], f)


def report(groups, output):
    """Write a v1-compatible JSON report: {group: (path, text_key)} -> per-case flags and McNemar vs base."""
    from scipy.stats import binomtest
    from .revision_core import read_jsonl, write_json
    out = dict(status='POST_HOC', version='v2',
               definition='Post hoc, judge-free audit: measurements in a note that never appear in its conversation, '
                          'after normalizing spoken numbers in the conversation and skipping date-like ratios.',
               groups={})
    flags = {}
    for name, (path, key) in groups.items():
        rows = read_jsonl(path)
        per = {r['case_id']: unsupported(r[key], r['conversation']) for r in rows}
        flags[name] = {k: int(bool(v)) for k, v in per.items()}
        out['groups'][name] = dict(n=len(rows), notes_with_unsupported_measurement=sum(flags[name].values()),
                                   unsupported_measurements=sum(len(v) for v in per.values()),
                                   per_case={k: v for k, v in per.items() if v})
    base = flags.get('base')
    for name, f in flags.items():
        if base and name != 'base' and not name.endswith('references') and set(f) == set(base):
            pos = sum(1 for k in base if base[k] == 0 and f[k] == 1)
            neg = sum(1 for k in base if base[k] == 1 and f[k] == 0)
            out['groups'][name]['vs_base'] = dict(discordant=[pos, neg], test='exact McNemar',
                                                  p=float(binomtest(pos, pos + neg).pvalue) if pos + neg else 1.0)
    write_json(output, out)
    return out
