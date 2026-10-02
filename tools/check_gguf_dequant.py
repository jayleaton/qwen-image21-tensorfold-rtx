"""Check tfimage's GPU GGUF decoders against gguf-py's numpy ones on real tensors."""
import sys, numpy as np, torch
from gguf import GGUFReader, quants
from tfimage.gguf_load import dequantize
path = sys.argv[1]
r = GGUFReader(path); seen = set()
for t in r.tensors:
    k = t.tensor_type.name
    if k in seen: continue
    seen.add(k)
    shape = tuple(int(v) for v in reversed(t.shape.tolist()))
    ref = torch.from_numpy(np.asarray(quants.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape))
    got = dequantize(k, t.data, shape, "cuda", torch.float32).cpu()
    print(k, t.name, shape, "max abs diff", (ref - got).abs().max().item(), "ref absmax", ref.abs().max().item())
