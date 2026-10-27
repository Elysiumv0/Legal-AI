"""Verify imports trong venv Python 3.12.

Chay: .venv\\Scripts\\python.exe scripts\\verify_imports.py
"""
import importlib
import sys

MODS = [
    ("underthesea", "tokenize tieng Viet (BM25)"),
    ("rank_bm25", "BM25Okapi"),
    ("yaml", "doc config yaml"),
    ("numpy", "vector math"),
    ("torch", "model backbone"),
    ("transformers", "bge-m3 / Qwen2.5"),
    ("qdrant_client", "dense index"),
    ("pydantic", "config schemas"),
    ("pdfplumber", "doc text layer"),
    ("fastapi", "API"),
    ("gradio", "UI"),
]

print(f"python: {sys.version.split()[0]}  exe: {sys.executable}")
print("-" * 62)
bad = 0
for m, why in MODS:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK   {m:14} {str(ver):12} ({why})")
    except Exception as e:
        bad += 1
        print(f"  FAIL {m:14} {type(e).__name__}: {e}")
print("-" * 62)
print(f"failures: {bad}/{len(MODS)}")
