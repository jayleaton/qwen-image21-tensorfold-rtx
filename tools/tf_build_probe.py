"""Build every TensorFold 0.6.1 CUDA extension on this Windows box and record pass/fail + build seconds."""
import importlib, json, sys, time, traceback
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())
TARGETS = [
    ("tensorfold.cuda.nvfp4.checkpoint", "_ext"), ("tensorfold.cuda.nvfp4.linear", "_ext"), ("tensorfold.cuda.nvfp4.linear", "_prompt_ext"),
    ("tensorfold.cuda.kernels.qmm", "_ext"), ("tensorfold.cuda.exl3.linear", "_ext"), ("tensorfold.cuda.kernels.prefill_attention", "_ext"),
    ("tensorfold.cuda.kernels.gdn", "_ext"), ("tensorfold.cuda.experts", "_ext"), ("tensorfold.cuda.exl3.experts", "_ext"),
]
only = sys.argv[1:]
res = {}
for mod, fn in TARGETS:
    key = f"{mod}.{fn}"
    if only and not any(o in key for o in only): continue
    t = time.perf_counter()
    try:
        getattr(importlib.import_module(mod), fn)()
        res[key] = {"ok": True, "s": round(time.perf_counter() - t, 1)}
    except BaseException as e:  # noqa: BLE001
        msg = traceback.format_exc()
        res[key] = {"ok": False, "s": round(time.perf_counter() - t, 1), "err": msg[-3000:]}
    print(key, "OK" if res[key]["ok"] else "FAIL", res[key]["s"], "s", flush=True)
import os; os.makedirs("runs", exist_ok=True)
json.dump(res, open("runs/tf_build_probe.json", "w"), indent=1)
sys.exit(0 if all(r["ok"] for r in res.values()) else 1)
