import argparse
import hashlib
import json
import re
from pathlib import Path


DEFAULT_TAGS = ("#training_e", "#training_d")


_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _stable_hash_to_int(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)


def build_tag_regex(tags: list[str]) -> re.Pattern:
    # Match tags as standalone tokens, allowing start-of-file and whitespace boundaries.
    tags_escaped = [re.escape(t) for t in tags]
    return re.compile(rf"(^|\s)({'|'.join(tags_escaped)})\b")


def remove_tags(text: str, tags: list[str]) -> str:
    for t in tags:
        # Remove any occurrences, even if followed by punctuation.
        text = re.sub(rf"(?:(?<=\s)|\A){re.escape(t)}\b", "", text)
        text = re.sub(rf"{re.escape(t)}\b", "", text)
    # Clean up double spaces produced by removals.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare tagged markdown corpus for GPT-2 fine-tuning.")
    p.add_argument(
        "--input",
        required=True,
        type=str,
        help="Root folder to scan for *.md (recurses).",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        default=list(DEFAULT_TAGS),
        help="Tags to filter on (include file if any tag exists).",
    )
    p.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory (will create train.txt, valid.txt, and manifest files).",
    )
    p.add_argument(
        "--valid-fraction",
        type=float,
        default=0.05,
        help="Fraction of included files assigned to validation split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Split seed (combined with per-file hash for stability).",
    )
    p.add_argument(
        "--strip-frontmatter",
        action="store_true",
        help="If present, remove YAML frontmatter blocks starting with ---.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_root = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted([p for p in in_root.rglob("*.md") if p.is_file()])
    tag_re = build_tag_regex(list(args.tags))

    included: list[Path] = []
    excluded: list[Path] = []

    for p in md_files:
        txt = _read_text(p)
        if tag_re.search(txt):
            included.append(p)
        else:
            excluded.append(p)

    def split_of(p: Path) -> str:
        # Deterministic split per file path, stable across runs.
        n = (_stable_hash_to_int(str(p)) ^ int(args.seed)) % 10_000
        return "valid" if (n / 10_000.0) < float(args.valid_fraction) else "train"

    train_texts: list[str] = []
    valid_texts: list[str] = []
    kept_files: list[dict] = []

    for p in included:
        raw = _read_text(p)
        text = _strip_frontmatter(raw) if args.strip_frontmatter else raw
        text = remove_tags(text, list(args.tags))
        text = _normalize_whitespace(text)

        split = split_of(p)
        if split == "valid":
            valid_texts.append(text)
        else:
            train_texts.append(text)

        kept_files.append(
            {
                "path": str(p),
                "split": split,
                "chars": len(text),
            }
        )

    train_path = out_dir / "train.txt"
    valid_path = out_dir / "valid.txt"
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"

    train_path.write_text("\n".join(train_texts).strip() + "\n", encoding="utf-8")
    valid_path.write_text("\n".join(valid_texts).strip() + "\n", encoding="utf-8")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in kept_files:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_root": str(in_root),
        "scanned_md_files": len(md_files),
        "included_tagged_files": len(included),
        "excluded_files": len(excluded),
        "kept_files": len(kept_files),
        "train_files": sum(1 for x in kept_files if x["split"] == "train"),
        "valid_files": sum(1 for x in kept_files if x["split"] == "valid"),
        "tags": list(args.tags),
        "valid_fraction": float(args.valid_fraction),
        "strip_frontmatter": bool(args.strip_frontmatter),
        "train_txt": str(train_path),
        "valid_txt": str(valid_path),
        "manifest_jsonl": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
