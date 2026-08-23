from __future__ import annotations

import pytest

from scripts.materialize_protocol_manifests import materialize_ifeval


def test_ifeval_materialization_rejects_fixed_subset_with_uncovered_ids() -> None:
    """Removing the post-fill coverage guard would silently freeze an incomplete set."""
    rows = [
        {"key": 1, "prompt": "first", "instruction_id_list": ["first-id"]},
        {"key": 2, "prompt": "second", "instruction_id_list": ["second-id"]},
    ]

    with pytest.raises(ValueError, match="uncovered instruction IDs"):
        materialize_ifeval(
            rows,
            {
                "prompt_hash_prefix": "seed\n",
                "subset_size": 1,
                "repository": "example/ifeval",
                "revision": "a" * 40,
            },
            "source-digest",
        )
