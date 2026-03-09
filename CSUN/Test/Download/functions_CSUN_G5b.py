
# Functions_CSUN.py
import yaml

def vertical_overlap(a, b):
    return not (a.b < b.t or b.b < a.t)

def bbox_from_tokens(tokens):
    l = min(t.l for t in tokens)
    r = max(t.r for t in tokens)
    t = min(t.t for t in tokens)
    b = max(t.b for t in tokens)
    return dict(l=l, t=t, r=r, b=b, w=r-l, h=b-t)

def write_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, sort_keys=False)

def g5a_classify_tokens(ctx, cfg):
    tokens = ctx["tokens"]
    tokens_g5a = []
    for tok in tokens:
        if tok["kind"] in ["WD_SPACE", "WIDE_SPACE"]:
            tok["role"] = "Space"
        elif tok["kind"] == "H_RUN":
            tok["role"] = "TextRun"
        else:
            tok["role"] = "Unknown"
        tokens_g5a.append(tok)
    ctx["tokens_g5a"] = tokens_g5a

def g5a_write_token_report(ctx, cfg):
    write_yaml(ctx["tokens_g5a"], "G5a_write_token_report.json")

def g5b_make_streams(ctx, cfg):
    tokens = ctx["tokens_g5a"]
    streams = []
    tokens_sorted = sorted(tokens, key=lambda t: t["t"])
    for tok in tokens_sorted:
        placed = False
        for s in streams:
            if vertical_overlap(SimpleToken(tok), SimpleToken(s["tokens"][0])):
                s["tokens"].append(tok)
                placed = True
                break
        if not placed:
            streams.append(dict(tokens=[tok]))
    for s in streams:
        s["tokens"] = sorted(s["tokens"], key=lambda t: t["l"])
    ctx["streams"] = streams

def debug_render_g5b_stream_boxes(ctx, cfg):
    boxes = []
    for i, s in enumerate(ctx["streams"]):
        bbox = bbox_from_tokens([SimpleToken(t) for t in s["tokens"]])
        boxes.append(dict(src_id=i, bbox=bbox))
    write_yaml(boxes, "G5b_dbg_stream_boxes.json")

def g5b_write_stream_report(ctx, cfg):
    out = dict(g5b_streams=[])
    for i, s in enumerate(ctx["streams"]):
        bbox = bbox_from_tokens([SimpleToken(t) for t in s["tokens"]])
        stream = []
        for tok in s["tokens"]:
            stream.append(dict(
                kind=tok["kind"],
                role=tok["role"],
                bbox=dict(l=tok["l"], t=tok["t"], r=tok["r"], b=tok["b"],
                          w=tok["r"]-tok["l"], h=tok["b"]-tok["t"])
            ))
        out["g5b_streams"].append(dict(
            src_id=i,
            bbox=bbox,
            atom_count=len(stream),
            gap_count=sum(1 for t in stream if t["kind"] in ["WD_SPACE","WIDE_SPACE"]),
            warnings=[],
            stream=stream
        ))
    write_yaml(out, "input_g5b_streams.yaml")

class SimpleToken:
    def __init__(self, d):
        self.l = d["l"]
        self.r = d["r"]
        self.t = d["t"]
        self.b = d["b"]
