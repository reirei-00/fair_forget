"""Prepare auditable local English counterparts from existing benchmark sources.

This command performs no inference, translation, downloads, or publication. Local
bundles supplement the public registry; they never weaken its checksum checks.
"""

import argparse
import copy
import csv
import json
import re
from pathlib import Path

from lm_eval.benchmark.data import PROTOCOLS, REGISTRY, digest, family, load, validate


PREPARATION_VERSION = "source_matched_bilingual_v1"
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "himself", "herself"}
ALIGNMENT_FIELDS = (
    "id",
    "cluster",
    "stratum",
    "panel",
    "group",
    "group_kind",
    "condition",
    "polarity",
    "gold",
    "stereotype",
    "unknown",
    "eligible",
    "protocol",
    "score_group",
    "category",
)


def _unique(rows, key, description):
    indexed = {}
    for row in rows:
        identifier = key(row)
        if identifier in indexed:
            raise ValueError(f"Duplicate {description}: {identifier}")
        indexed[identifier] = row
    return indexed


def _english(uk, task, source):
    out = copy.deepcopy(uk)
    out.update(task=task, language="en", source=source, paired_eligible=True)
    return out


def bbq_counterparts(uk_rows, english_rows):
    indexed = _unique(
        english_rows, lambda r: (r["category"], int(r["example_id"])), "BBQ source ID"
    )
    result = []
    for uk in uk_rows:
        source = uk["source"]
        en = indexed[(source["category"], int(source["example_id"]))]
        for field in (
            "label",
            "question_index",
            "question_polarity",
            "context_condition",
        ):
            if str(source[field]) != str(en[field]):
                raise ValueError(f"{uk['id']}: BBQ {field} differs across languages")
        unknown = [i for i in range(3) if en["answer_info"][f"ans{i}"][1] == "unknown"]
        if unknown != [uk["unknown"]]:
            raise ValueError(f"{uk['id']}: BBQ unknown answer differs")
        row = _english(
            uk,
            "bbq_en",
            {**en, "context_en": en["context"], "question_en": en["question"]},
        )
        row["choices"] = [en[f"ans{i}"] for i in range(3)]
        result.append(row)
    return result


def stereoset_counterparts(uk_rows, english_rows):
    indexed = _unique(english_rows, lambda r: r["id"], "StereoSet source ID")
    result = []
    for uk in uk_rows:
        en = indexed[uk["id"]]
        sentences = _unique(
            en["sentences"], lambda r: r["gold_label"], "StereoSet gold label"
        )
        if set(sentences) != {"stereotype", "anti-stereotype", "unrelated"}:
            raise ValueError(
                f"{uk['id']}: StereoSet requires three semantic candidates"
            )
        # Common target IDs are the pinned UK labels. English source labels remain
        # in provenance; this avoids changing target-macro averaging by language.
        row = _english(uk, "stereoset_en", en)
        row["sentences"] = [
            sentences[label]["sentence"]
            for label in ("stereotype", "anti-stereotype", "unrelated")
        ]
        result.append(row)
    return result


def _wino_sentence(pair, condition, controlled):
    source = pair[f"{condition}_source"]
    spans = re.findall(r"\[([^\]]+)\]", source)
    antecedents = [s for s in spans if s.strip().lower() not in PRONOUNS]
    pronouns = [s for s in spans if s.strip().lower() in PRONOUNS]
    if len(antecedents) != 1 or not pronouns:
        raise ValueError(f"{pair['pair_id']}: unsupported WinoBias coreference chain")
    if pair["referent_en"].lower() not in antecedents[0].lower():
        raise ValueError(f"{pair['pair_id']}: WinoBias antecedent identity mismatch")
    pronoun = pronouns[0]
    # Remove gold antecedent brackets. For the controlled task mark only the first
    # pronoun in the source chain, as a query marker, never the gold candidate.
    marked = False

    def replace(match):
        nonlocal marked
        value = match.group(1)
        if controlled and not marked and value.strip().lower() in PRONOUNS:
            marked = True
            return f"[{value}]"
        return value

    sentence = re.sub(r"\[([^\]]+)\]", replace, source)
    distractor = pair["distractor_en"]
    if not re.search(r"\b" + re.escape(distractor) + r"\b", sentence, re.IGNORECASE):
        raise ValueError(f"{pair['pair_id']}: English distractor missing from sentence")
    # Both candidates use the same occupation-label convention. Reusing the
    # annotated full antecedent only for the correct option would make its
    # determiner/capitalization a systematic gold-answer cue.
    return sentence, pair["referent_en"], distractor, pronoun


def winobias_counterparts(uk_rows, english_pairs, *, controlled):
    indexed = _unique(
        english_pairs,
        lambda p: (p["source_split"], p["sentence_type"], str(p["line_number"])),
        "WinoBias source pair",
    )
    task = "winobias_en_controlled" if controlled else "winobias_en_natural"
    result = []
    for uk in uk_rows:
        if not controlled and uk["score_group"] != "primary_balanced":
            uk.update(
                paired_eligible=False,
                pairing_exclusion_reason="UK agreement/cross control has no original English counterpart",
            )
            continue
        raw = uk["source"]
        split = "dev" if raw["split"] == "validation" else raw["split"]
        pair = indexed[(split, raw["type"], raw["item_number"])]
        if pair[f"{uk['condition']}_gender"] != uk["group"]:
            raise ValueError(f"{uk['id']}: WinoBias pronoun gender differs")
        sentence, referent, distractor, pronoun = _wino_sentence(
            pair, uk["condition"], controlled
        )
        row = _english(
            uk,
            task,
            {
                "sentence_en": sentence,
                "source_pair_id": pair["pair_id"],
                "source_split": split,
                "source_condition": uk["condition"],
                "referent_en": pair["referent_en"],
                "distractor_en": pair["distractor_en"],
                "original_annotated_source": pair[f"{uk['condition']}_source"],
            },
        )
        row["choices"] = [None, None]
        row["choices"][uk["gold"]] = referent
        row["choices"][1 - uk["gold"]] = distractor
        row["pronoun"] = pronoun
        row["pairing_limitation"] = (
            "UK uses neutral occupation paraphrases; EN uses original occupation wording. Difference is language plus adaptation."
            if controlled
            else "UK primary mm/ff variants include grammatical gender agreement; EN uses original occupation wording."
        )
        uk["pairing_limitation"] = row["pairing_limitation"]
        result.append(row)
    return result


def validate_wino_lineage(
    natural, controlled, pairs, natural_construction, controlled_construction
):
    """Verify source occupations/answer order against existing construction records."""
    pairs_by_key = _unique(
        pairs,
        lambda p: (p["source_split"], p["sentence_type"], str(p["line_number"])),
        "WinoBias pair",
    )
    natural_by_id = _unique(
        natural_construction, lambda r: r["pair_id"], "natural construction pair"
    )
    controlled_by_key = _unique(
        controlled_construction,
        lambda r: (
            r["split"],
            r["type"],
            frozenset((r["source_1_en"], r["source_2_en"])),
            r["gender"],
        ),
        "controlled construction row",
    )
    for row in natural:
        raw = row["source"]
        split = "dev" if raw["split"] == "validation" else raw["split"]
        pair = pairs_by_key[(split, raw["type"], raw["item_number"])]
        construction = natural_by_id[pair["pair_id"]]
        text = construction["variants"][raw["variant"]]
        if re.sub(r"</?[TDP]>", "", text) != raw["sentence_uk"]:
            raise ValueError(f"{row['id']}: natural construction sentence differs")
        for tag, field in (
            ("T", "target_span_uk"),
            ("D", "other_span_uk"),
            ("P", "pronoun_span_uk"),
        ):
            spans = re.findall(f"<{tag}>(.*?)</{tag}>", text)
            if spans != [raw[field]]:
                raise ValueError(f"{row['id']}: natural construction span differs")
        if (
            raw["score_group"] == "primary_balanced"
            and raw["coreference_role"] != "target"
        ):
            raise ValueError(
                f"{row['id']}: primary variant must refer to original target"
            )
    for row in controlled:
        raw = row["source"]
        pair = pairs_by_key[(raw["split"], raw["type"], raw["item_number"])]
        original = controlled_by_key[
            (
                raw["split"],
                raw["type"],
                frozenset((pair["pro_source"], pair["anti_source"])),
                raw["pronoun_gender"],
            )
        ]
        for source_field, export_field in (
            ("sentence_uk", "sentence_uk"),
            ("choice_a_uk", "candidate_a_uk"),
            ("choice_b_uk", "candidate_b_uk"),
            ("gold", "gold_candidate"),
        ):
            if original[source_field] != raw[export_field]:
                raise ValueError(
                    f"{row['id']}: controlled construction {source_field} differs"
                )
        for field in ("referent_en", "distractor_en"):
            if original[field] != pair[field]:
                raise ValueError(f"{row['id']}: controlled occupation identity differs")
        if {original["source_1_en"], original["source_2_en"]} != {
            pair["pro_source"],
            pair["anti_source"],
        }:
            raise ValueError(f"{row['id']}: controlled English source differs")


def validate_alignment(uk, en):
    left = _unique(
        [r for r in uk if r["paired_eligible"]], lambda r: r["id"], "eligible UK ID"
    )
    right = _unique(en, lambda r: r["id"], "English ID")
    if left.keys() != right.keys():
        raise ValueError("Declared eligible UK and English ID sets differ")
    for identifier, row in left.items():
        for field in ALIGNMENT_FIELDS:
            if row.get(field) != right[identifier].get(field):
                raise ValueError(f"{identifier}: bilingual {field} differs")
    validate(uk)
    validate(en)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preparation output must be empty; preserve existing bundles")
    uk_paths = {
        "bbq_uk": root / "outputs/benchmark_preparation_20260922/bbq_uk/data/test.csv",
        "stereoset_uk": root
        / "dataset_sources/hf_dataset/stereoset-uk-eval/data/validation.csv",
        "winobias_uk_natural": root
        / "dataset_sources/hf_dataset/winobias-uk-natural/data/validation.csv",
        "winobias_uk_controlled": root
        / "dataset_sources/hf_dataset/winobias-uk-controlled/data/test.csv",
    }
    sources = []

    def source(path, role):
        sources.append(
            {"path": str(path.relative_to(root)), "sha256": digest(path), "role": role}
        )
        return path

    uk_data = {}
    for task, path in uk_paths.items():
        uk_data[task] = load(task, source(path, task))
        for row in uk_data[task]:
            row["paired_eligible"] = True
    bbq = []
    for path in sorted((root / "data/bbq/data").glob("*.jsonl")):
        source(path, "original_english_bbq")
        bbq.extend(json.loads(line) for line in path.read_text().splitlines())
    ss_path = source(root / "data/stereoset/dev.json", "original_english_stereoset")
    ss = json.loads(ss_path.read_text())["data"]["intrasentence"]
    wino_path = source(
        root
        / "dataset_sources/outputs/winobias_uk_build/lapa_pair_translations_corrected.jsonl",
        "embedded_original_english_winobias",
    )
    wino = [json.loads(line)["pair"] for line in wino_path.read_text().splitlines()]
    natural_path = source(
        root
        / "dataset_sources/outputs/winobias_uk_natural_full/natural_construction.jsonl",
        "natural_export_lineage",
    )
    controlled_path = source(
        root / "dataset_sources/outputs/winobias_uk_controlled_full/controlled.csv",
        "controlled_export_lineage",
    )
    with controlled_path.open(encoding="utf-8-sig", newline="") as stream:
        controlled_construction = list(csv.DictReader(stream))
    validate_wino_lineage(
        uk_data["winobias_uk_natural"],
        uk_data["winobias_uk_controlled"],
        wino,
        [json.loads(line) for line in natural_path.read_text().splitlines()],
        controlled_construction,
    )
    english = {
        "bbq_uk": bbq_counterparts(uk_data["bbq_uk"], bbq),
        "stereoset_uk": stereoset_counterparts(uk_data["stereoset_uk"], ss),
        "winobias_uk_natural": winobias_counterparts(
            uk_data["winobias_uk_natural"], wino, controlled=False
        ),
        "winobias_uk_controlled": winobias_counterparts(
            uk_data["winobias_uk_controlled"], wino, controlled=True
        ),
    }
    for task, uk in uk_data.items():
        validate_alignment(uk, english[task])
    # All semantic checks precede writes, so a broken source cannot yield a
    # plausible partial bundle manifest.
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir()
    (output / "pairs").mkdir()
    manifest = {
        "schema_version": 1,
        "preparation_version": PREPARATION_VERSION,
        "preparation_code_sha256": digest(Path(__file__)),
        "human_validated": False,
        "sources": sources,
        "source_revisions": {
            name: (root / f"data/{name}/SOURCE_COMMIT").read_text().strip()
            for name in ("bbq", "stereoset")
        },
        "datasets": {},
        "pairs": [],
    }
    for task, uk in uk_data.items():
        en = english[task]
        en_task = en[0]["task"]
        pair_path = output / "pairs" / f"{task}.jsonl"
        mapping = [
            {"id": r["id"], "uk_id": r["id"], "en_id": r["id"]}
            for r in uk
            if r["paired_eligible"]
        ]
        _write_jsonl(pair_path, mapping)
        pairing = {
            "uk_task": task,
            "en_task": en_task,
            "matched_rows": len(mapping),
            "excluded_uk_rows": len(uk) - len(mapping),
            "filename": str(pair_path.relative_to(output)),
            "sha256": digest(pair_path),
            "alignment": "source_identity_and_scoring_metadata",
            "interpretation": "Language plus adaptation; mechanical matching does not establish translation validity.",
        }
        manifest["pairs"].append(pairing)
        for rows, name, counterpart in ((uk, task, en_task), (en, en_task, task)):
            path = output / "data" / f"{name}.jsonl"
            _write_jsonl(path, rows)
            checksum = digest(path)
            manifest["datasets"][name] = {
                "repo": "local-bundle",
                "revision": f"local-sha256:{checksum}",
                "filename": str(path.relative_to(output)),
                "sha256": checksum,
                "rows": len(rows),
                "language": rows[0]["language"],
                "human_validated": False,
                "format": "adapted_jsonl_v1",
                "protocol": PROTOCOLS[family(name)],
                "counterpart": counterpart,
                "pairing": pairing,
                "original_registry": REGISTRY[task],
            }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.source_root, args.output))


if __name__ == "__main__":
    main()
