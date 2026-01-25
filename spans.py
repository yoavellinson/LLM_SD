from difflib import SequenceMatcher

def split_into_spans(text, span_words=10):
    words = text.split()
    spans, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) >= span_words:
            spans.append(" ".join(buf))
            buf = []
    if buf:
        spans.append(" ".join(buf))
    return spans


def span_mask_for_speaker(spans, target_turns, speaker):
    span_tokens = [s.split() for s in spans]
    flat = [w for sp in span_tokens for w in sp]

    spk_text = " ".join(t["text"] for t in target_turns if t["speaker"] == speaker)
    spk_tokens = spk_text.split()

    sm = SequenceMatcher(a=flat, b=spk_tokens, autojunk=False)
    token_mask = [0] * len(flat)
    for m in sm.get_matching_blocks():
        for i in range(m.a, m.a + m.size):
            if i < len(token_mask):
                token_mask[i] = 1

    span_mask = []
    idx = 0
    for toks in span_tokens:
        n = len(toks)
        frac = sum(token_mask[idx:idx+n]) / max(1, n)
        span_mask.append(1 if frac >= 0.5 else 0)
        idx += n

    return span_mask
