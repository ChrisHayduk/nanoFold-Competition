from __future__ import annotations

from nanofold.chain_paths import chain_id_from_stem, chain_id_to_stem


def test_chain_file_stems_preserve_case_sensitive_chain_ids() -> None:
    upper = chain_id_to_stem("6tmg_G")
    lower = chain_id_to_stem("6tmg_g")

    assert upper != lower
    assert upper.casefold() != lower.casefold()
    assert chain_id_from_stem(upper) == "6tmg_G"
    assert chain_id_from_stem(lower) == "6tmg_g"

