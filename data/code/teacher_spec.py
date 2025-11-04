# teacher_spec.py
# Utilities for loading SPEC and building rendered prompts
from pathlib import Path
from typing import List, Tuple
import json

DEFAULT_SPEC_ROOT = Path("/home/elicer/workspace/1mo/SPEC")

def _task_variants(task: str) -> List[str]:
    t = (task or "").strip()
    if not t:
        return []
    bases = {t, t.replace("_", " "), t.replace(" ", "_"), t.replace("-", " ")}
    also = set()
    for b in list(bases):
        also.add(b.capitalize())
        also.add(b.title())
    return list(bases | also)

def load_spec_text(domain: str, task: str, spec_root: Path = DEFAULT_SPEC_ROOT) -> str:
    """Return SPEC as a bulleted string. Supports:
    - list[str]
    - {"specifications": [{"text": "..."}]}
    - {"rules": ["..."]}
    """
    dom_dir = spec_root / (domain or "General")
    if not dom_dir.exists():
        raise FileNotFoundError(f"[SPEC] Domain directory not found: {dom_dir}")

    tried = []
    variants = _task_variants(task or "General") or ["General", "Communication"]

    for t in variants:
        spec_path = dom_dir / f"{t}.json"
        tried.append(str(spec_path))
        if spec_path.exists():
            with open(spec_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            rules = []
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                rules = data
            elif isinstance(data, dict):
                if "specifications" in data and isinstance(data["specifications"], list):
                    rules = [it["text"] for it in data["specifications"] if isinstance(it, dict) and "text" in it]
                elif "rules" in data and isinstance(data["rules"], list):
                    rules = [str(x) for x in data["rules"]]

            if not rules:
                raise ValueError(f"[SPEC] Unsupported or empty JSON structure: {spec_path}")
            return "\n".join(f"- {r}" for r in rules)

    raise FileNotFoundError(
        "[SPEC] Spec file not found for "
        f"domain='{domain}', task='{task}'. Tried:\n  " + "\n  ".join(tried)
    )

def render_prompts(system_tpl: str, user_tpl: str, *, domain: str, task: str, spec_bullets: str, query: str) -> Tuple[str, str]:
    system_prompt = system_tpl.format(domain=domain, task=task, spec_bullets=spec_bullets)
    user_prompt   = user_tpl.format(query=query)
    return system_prompt, user_prompt
