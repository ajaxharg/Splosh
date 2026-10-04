#!/usr/bin/env python3
"""Reference decoders for the GGUF quant formats of Unsloth's UD-Q4_K_M, checked against an MLX pack.

    tools/gguf_check.py <file.gguf> [--pack mlx-q4] [--reference mlx-q8] [--rows 48] [--every 8]

With --pack alone: decodes rows of one tensor of each format (and a few named ones) and prints
their correlation with the same tensor of the MLX pack under inputs/. A wrong decoder or a wrong
name mapping shows as a correlation far from 1.

With --reference as well: the error of the GGUF and of --pack against the reference pack (the
8-bit one), as relative RMS over the sampled rows, by format, for every --every'th layer.

What it establishes about a Qwen3.5-family GGUF, beyond the block layouts (which follow
llama.cpp's ggml-quants.c; the tables are its kvalues_iq4nl and iq3s_grid, MIT):
  * tensor names map one to one (NAMES below); blk.<layers> is the MTP block, not used here;
  * a 2-D tensor's dims are [inner, rows], row-major, a whole number of blocks a row;
  * the gated-delta value heads are in a different order: GGUF head i is MLX head
    3 * (i % 16) + i // 16. That permutes the rows of ssm_alpha, ssm_beta and attn_gate, the
    value rows of attn_qkv (from row 4096), the value channels of ssm_conv1d, ssm_a,
    ssm_dt.bias, and the columns of ssm_out, in chunks of 128;
  * ssm_a is -exp(A_log); norm weights are fp32 with the +1 already added, as in the MLX pack.
"""
import argparse, collections, json, mmap, os, re, struct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = {0: "F32", 1: "F16", 8: "Q8_0", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 20: "IQ4_NL", 21: "IQ3_S", 23: "IQ4_XS", 30: "BF16"}
GGUF, DATA, TENSORS = None, 0, {}

def open_gguf(path):
    """Fill TENSORS: name -> (dims, type, offset from DATA)."""
    global GGUF, DATA, TENSORS
    GGUF = path
    with open(path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    o = 0
    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, buf, o); o += struct.calcsize("<" + fmt); return v[0]
    def rs():
        nonlocal o
        n = rd("Q"); s = bytes(buf[o:o + n]); o += n; return s.decode("utf-8", "replace")
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
    def rv(t):
        nonlocal o
        if t == 8: return rs()
        if t == 9:
            et = rd("I"); n = rd("Q")
            if et == 8:
                for _ in range(n): rs()
                return None
            o += struct.calcsize(sizes[et]) * n; return None
        return rd(sizes[t])
    assert buf[:4] == b"GGUF", "not a GGUF file"
    o = 4; rd("I"); count = rd("Q"); pairs = rd("Q"); alignment = 32
    for _ in range(pairs):
        key = rs(); value = rv(rd("I"))
        if key == "general.alignment": alignment = value
    for _ in range(count):
        name = rs(); dims = [rd("Q") for _ in range(rd("I"))]; kind = rd("I"); offset = rd("Q")
        TENSORS[name] = (dims, TYPES.get(kind, str(kind)), offset)
    DATA = (o + alignment - 1) // alignment * alignment

BLOCK = {"Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q3_K": (256, 110), "Q8_0": (32, 34),
         "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "IQ3_S": (256, 110)}
IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)

def iq3s_grid():
    text = open(os.path.join(os.path.dirname(ROOT), "Splish/runtime/metal/abi/QuantTables.h")).read()   # llama.cpp's iq3s_grid
    body = text.split("kIQ3SGrid[512] = {", 1)[1].split("};", 1)[0]
    values = np.array([int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]+", body)], dtype=np.uint32)
    assert len(values) == 512
    return values.view(np.uint8).reshape(512, 4).astype(np.float32)

def f16(b):
    return np.frombuffer(b, dtype="<f2").astype(np.float32)

def k4_scales(scales):   # [blocks, 12] -> sc[blocks, 8], m[blocks, 8]
    s = scales.astype(np.uint32)
    sc = np.zeros((len(s), 8), np.float32); mn = np.zeros((len(s), 8), np.float32)
    for j in range(4):
        sc[:, j] = s[:, j] & 63; mn[:, j] = s[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
        mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn

def decode(kind, raw):
    """raw: uint8 [blocks, block_bytes] -> float32 [blocks, block_elements]."""
    n = len(raw)
    if kind == "Q4_K" or kind == "Q5_K":
        d = f16(raw[:, 0:2].tobytes()); dmin = f16(raw[:, 2:4].tobytes())
        sc, mn = k4_scales(raw[:, 4:16])
        base = 16 if kind == "Q4_K" else 48
        qs = raw[:, base:base + 128].reshape(n, 4, 32)          # 64-element halves
        lo = np.stack([qs & 15, qs >> 4], axis=2).reshape(n, 8, 32).astype(np.float32)   # group j = 2 * half + nibble
        if kind == "Q5_K":
            qh = raw[:, 16:48]                                    # bit j of qh[e] is the fifth bit of element e of group j
            hi = np.stack([(qh >> j) & 1 for j in range(8)], axis=1).astype(np.float32)
            lo = lo + 16 * hi
        return (lo * (d[:, None] * sc)[:, :, None] - (dmin[:, None] * mn)[:, :, None]).reshape(n, 256)
    if kind == "Q6_K":
        ql = raw[:, 0:128].reshape(n, 2, 2, 32); qh = raw[:, 128:192].reshape(n, 2, 32)
        scales = raw[:, 192:208].view(np.int8).astype(np.float32); d = f16(raw[:, 208:210].tobytes())
        out = np.zeros((n, 8, 32), np.float32)
        for j in range(8):
            hb, quarter = j // 4, j % 4
            lo = (ql[:, hb, quarter & 1, :] >> (4 * (quarter >> 1))) & 15
            hi = (qh[:, hb, :] >> (2 * quarter)) & 3
            q = (lo | (hi << 4)).astype(np.float32) - 32
            s = np.repeat(scales[:, 2 * j:2 * j + 2], 16, axis=1)
            out[:, j, :] = d[:, None] * s * q
        return out.reshape(n, 256)
    if kind == "Q3_K":
        hmask = raw[:, 0:32]; qs = raw[:, 32:96].reshape(n, 2, 32); d = f16(raw[:, 108:110].tobytes())
        t = raw[:, 96:108].copy().view("<u4")                     # three words
        t0, t1, t2 = t[:, 0], t[:, 1], t[:, 2]
        aux = np.stack([(t0 & 0x0f0f0f0f) | (((t2 >> 0) & 0x03030303) << 4), (t1 & 0x0f0f0f0f) | (((t2 >> 2) & 0x03030303) << 4),
                        ((t0 >> 4) & 0x0f0f0f0f) | (((t2 >> 4) & 0x03030303) << 4), ((t1 >> 4) & 0x0f0f0f0f) | (((t2 >> 6) & 0x03030303) << 4)], axis=1)
        sc = aux.astype("<u4").view(np.uint8).reshape(n, 16).astype(np.float32) - 32      # one per 16 elements
        out = np.zeros((n, 8, 32), np.float32)
        for j in range(8):
            hb, jj = j // 4, j % 4
            lo = (qs[:, hb, :] >> (2 * jj)) & 3
            hi = (hmask >> j) & 1
            q = (lo | (hi << 2)).astype(np.float32) - 4
            s = np.repeat(sc[:, 2 * j:2 * j + 2], 16, axis=1)
            out[:, j, :] = d[:, None] * s * q
        return out.reshape(n, 256)
    if kind == "Q8_0":
        return f16(raw[:, 0:2].tobytes())[:, None] * raw[:, 2:34].view(np.int8).astype(np.float32)
    if kind == "IQ4_NL":
        qs = raw[:, 2:18]
        idx = np.concatenate([qs & 15, qs >> 4], axis=1)
        return f16(raw[:, 0:2].tobytes())[:, None] * IQ4NL[idx]
    if kind == "IQ4_XS":
        d = f16(raw[:, 0:2].tobytes()); sh = raw[:, 2:4].copy().view("<u2")[:, 0].astype(np.uint32); sl = raw[:, 4:8].astype(np.uint32)
        qs = raw[:, 8:136].reshape(n, 8, 16)
        out = np.zeros((n, 8, 32), np.float32)
        for j in range(8):
            ls = ((sl[:, j // 2] >> (4 * (j % 2))) & 0xF) | (((sh >> (2 * j)) & 3) << 4)
            idx = np.concatenate([qs[:, j] & 15, qs[:, j] >> 4], axis=1)
            out[:, j, :] = (d * (ls.astype(np.float32) - 32))[:, None] * IQ4NL[idx]
        return out.reshape(n, 256)
    if kind == "IQ3_S":
        grid = iq3s_grid()
        d = f16(raw[:, 0:2].tobytes()); qs = raw[:, 2:66].reshape(n, 8, 8).astype(np.uint32); qh = raw[:, 66:74].astype(np.uint32)
        signs = raw[:, 74:106].reshape(n, 8, 4); scales = raw[:, 106:110].astype(np.uint32)
        out = np.zeros((n, 8, 32), np.float32)
        for j in range(8):
            scale = (scales[:, j // 2] >> (4 * (j % 2))) & 15
            index = qs[:, j, :] | (((qh[:, j][:, None] >> np.arange(8)) & 1) << 8)          # [n, 8] grid entries of 4
            magnitude = grid[index].reshape(n, 32)
            bits = np.unpackbits(signs[:, j, :], axis=1, bitorder="little").astype(np.float32)   # bit e % 8 of byte e / 8
            out[:, j, :] = (d * (1 + 2 * scale.astype(np.float32)))[:, None] * magnitude * (1 - 2 * bits)
        return out.reshape(n, 256)
    raise ValueError(kind)

def gguf_rows(name, rows):
    dims, kind, offset = TENSORS[name]
    inner, total = dims[0], dims[1]
    elements, size = BLOCK[kind]
    per_row = inner // elements * size
    with open(GGUF, "rb") as f:
        f.seek(DATA + offset); raw = np.frombuffer(f.read(per_row * rows), dtype=np.uint8).reshape(-1, size)
    return decode(kind, raw).reshape(rows, inner), kind, total

def mlx_rows(name, rows, pack):
    index = json.load(open(f"{ROOT}/inputs/{pack}/model.safetensors.index.json"))["weight_map"]
    def read(tensor, count_rows):
        path = f"{ROOT}/inputs/{pack}/{index[tensor]}"
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); e = header[tensor]
            f.seek(8 + n + e["data_offsets"][0])
            width = {"U32": 4, "BF16": 2}[e["dtype"]]
            return np.frombuffer(f.read(count_rows * e["shape"][1] * width), dtype={"U32": "<u4", "BF16": "<u2"}[e["dtype"]]).reshape(count_rows, e["shape"][1])
    words = read(name + ".weight", rows)
    bf = lambda a: (a.astype(np.uint32) << 16).view(np.float32)
    scales, biases = bf(read(name + ".scales", rows)), bf(read(name + ".biases", rows))
    groups = scales.shape[1]
    bits = words.shape[1] * 32 // (groups * 64)
    codes = np.stack([(words >> (bits * i)) & ((1 << bits) - 1) for i in range(32 // bits)], axis=2).reshape(rows, -1).astype(np.float32)
    return codes * np.repeat(scales, 64, axis=1) + np.repeat(biases, 64, axis=1)

NAMES = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_alpha": "linear_attn.in_proj_a",
         "ssm_beta": "linear_attn.in_proj_b", "ssm_out": "linear_attn.out_proj", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
         "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}

def mlx_name(gguf):
    if gguf == "token_embd.weight": return "language_model.model.embed_tokens"
    if gguf == "output.weight": return "language_model.lm_head"
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight", gguf)
    return f"language_model.model.layers.{m.group(1)}.{NAMES[m.group(2)]}"

def relative(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("gguf"); p.add_argument("--pack", default="mlx-q4"); p.add_argument("--reference")
    p.add_argument("--rows", type=int, default=48); p.add_argument("--every", type=int, default=8)
    args = p.parse_args()
    open_gguf(args.gguf)
    layers = 1 + max(int(m.group(1)) for m in (re.match(r"blk\.(\d+)\.", n) for n in TENSORS) if m)
    mtp = f"blk.{layers - 1}." if any(".nextn." in n for n in TENSORS) else "blk.none."
    packed = [(n, k) for n, (d, k, _) in TENSORS.items() if k in BLOCK and len(d) == 2 and not n.startswith(mtp)]
    if not args.reference:
        picked = {}
        for name, kind in packed: picked.setdefault(kind, name)
        extra = ["token_embd.weight", "output.weight", "blk.0.attn_gate.weight", "blk.3.attn_q.weight"]
        print(f"{'tensor':34} {'type':7} {'shape':16} corr     rms(gguf - pack) / rms(pack)   [{args.pack}]")
        for name in list(picked.values()) + [e for e in extra if e in TENSORS and e not in picked.values()]:
            permuted = ".ssm_alpha." in name or ".ssm_beta." in name
            g, kind, total = gguf_rows(name, 48 if permuted else args.rows)
            m = mlx_rows(mlx_name(name), 48 if permuted else args.rows, args.pack)
            if permuted: m = m[[3 * (i % 16) + i // 16 for i in range(48)]]     # GGUF head i is MLX head 3 (i % 16) + i // 16
            print(f"{name:34} {kind:7} {str([total, g.shape[1]]):16} {np.corrcoef(g.ravel(), m.ravel())[0, 1]:.5f}  {relative(g, m):.4f}")
        return
    index = json.load(open(f"{ROOT}/inputs/{args.reference}/model.safetensors.index.json"))["weight_map"]
    present = {s for s in set(index.values()) if os.path.exists(f"{ROOT}/inputs/{args.reference}/{s}")}
    results, weights, missing = collections.defaultdict(list), collections.Counter(), 0
    for name, kind in packed:
        dims = TENSORS[name][0]; weights[kind] += dims[0] * dims[1]
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight", name)
        # The first rows of ssm_alpha, ssm_beta and ssm_out are permuted between the two; left out.
        if m and (m.group(2) in ("ssm_out", "ssm_alpha", "ssm_beta") or int(m.group(1)) % args.every): continue
        if index.get(mlx_name(name) + ".weight") not in present: missing += 1; continue
        g, _, _ = gguf_rows(name, args.rows)
        reference = mlx_rows(mlx_name(name), args.rows, args.reference)
        results[kind].append((relative(g, reference), relative(mlx_rows(mlx_name(name), args.rows, args.pack), reference)))
    if missing: print(f"{missing} sampled tensors are in shards of {args.reference} that are not on disk")
    total = sum(weights.values())
    print(f"{'format':8} {'share':>6} {'n':>3}  gguf vs {args.reference:8}  {args.pack} vs {args.reference}")
    mean_g = mean_p = covered = 0.0
    for kind, items in sorted(results.items(), key=lambda kv: -weights[kv[0]]):
        ge, pe = np.mean([x[0] for x in items]), np.mean([x[1] for x in items])
        share = weights[kind] / total; mean_g += share * ge; mean_p += share * pe; covered += share
        print(f"{kind:8} {share:6.1%} {len(items):3d}  {ge:.4f}            {pe:.4f}")
    if covered: print(f"weighted by share ({covered:.0%} of weights covered): gguf {mean_g / covered:.4f}, {args.pack} {mean_p / covered:.4f}")

if __name__ == "__main__":
    main()
