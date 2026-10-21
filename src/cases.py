"""1,000 handoff cases built from live openFDA labels. See data/manifest.json."""

import json
from pathlib import Path

CASES = json.loads((Path(__file__).resolve().parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))
