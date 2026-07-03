"""Config resolution/hashing helpers shared by the harness and the run driver.

`config_id` is what the run driver (scripts/run_matrix.py) uses to decide
whether a (experiment, model, config, seed, fraction) tuple has already been
run -- it hashes the fully-resolved config dict, not a config file path, so
two differently-named YAML files that resolve to the same dict collide (by
design) and two edits to the same file produce different ids.
"""

from __future__ import annotations

import hashlib
import json


def compute_config_id(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
