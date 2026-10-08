"""Audits all models registered in ModelRegistry for availability and precision support."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from person1_kv_engine.adapters.registry import ModelRegistry

def main():
    hub = Path.home() / ".cache/huggingface/hub"
    models = ModelRegistry.list_models()
    print(f"Total registered models: {len(models)}")
    for k, v in models.items():
        m_id = v.get("model_id", "")
        h_dir = hub / ("models--" + m_id.replace("/", "--"))
        cached = False
        if h_dir.exists():
            snapshots = h_dir / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                cached = True
        comp = v.get("compatibility_level")
        comp_str = comp.value if hasattr(comp, "value") else str(comp)
        precs = v.get("supported_precisions", [])
        contexts = v.get("supported_contexts", [])
        print(f"Model: {k:<15} | HF: {m_id:<32} | Cached: {str(cached):<5} | Comp: {comp_str:<12} | Prec: {precs} | Ctx: {contexts}")

if __name__ == "__main__":
    main()
