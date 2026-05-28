#!/usr/bin/env python3
"""Clean training/data/train.txt: strip [[...]] tags, #hashtags, and optional plain words."""

import argparse
import re
from pathlib import Path

# [[Alex]], [[Unassigned]], etc.
BRACKET_TAG_RE = re.compile(r"\[\[[^\]]*\]\]", re.IGNORECASE)
# #training_e, #Sam, ...
HASHTAG_RE = re.compile(r"#\S+")
# trailing " #" at end of line (or before whitespace)
LONE_HASH_RE = re.compile(r"\s#(?=\s|$)")


def load_word_list(path: Path | None, inline: list[str]) -> list[str]:
    words = [w.strip() for w in inline if w.strip()]
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            # "# comment" only — not "#Sam" (hashtag token to remove)
            if re.match(r"^#\s", line):
                continue
            words.append(line)
    seen: set[str] = set()
    out: list[str] = []
    for w in sorted(words, key=len, reverse=True):
        key = w.lower()
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


def plain_word_pattern(word: str, case_sensitive: bool) -> re.Pattern:
    flags = 0 if case_sensitive else re.IGNORECASE
    if word.startswith("[[") or word.startswith("#"):
        return re.compile(re.escape(word), flags)
    return re.compile(rf"\b{re.escape(word)}\b", flags)


def clean_line(line: str, extra_patterns: list[re.Pattern]) -> str:
    line = BRACKET_TAG_RE.sub("", line)
    line = HASHTAG_RE.sub("", line)
    line = LONE_HASH_RE.sub("", line)
    for pat in extra_patterns:
        line = pat.sub("", line)
    line = re.sub(r"[ \t]{2,}", " ", line)
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    return line.rstrip()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Strip [[...]], #hashtags, and optional plain words from a text file."
    )
    p.add_argument("--input", type=Path, default=Path("training/data/train.txt"))
    p.add_argument("--output", type=Path, default=None, help="Default: overwrite --input")
    p.add_argument(
        "--words-file",
        type=Path,
        default=None,
        help="Extra tokens/phrases, one per line (# at line start + space = comment).",
    )
    p.add_argument("--word", action="append", default=[], help="Repeatable extra token.")
    p.add_argument(
        "--drop-lines",
        action="store_true",
        help="Drop lines that still contain a --words-file token after regex (not default).",
    )
    p.add_argument("--case-sensitive", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print stats only; do not write.")
    args = p.parse_args()

    extra_words = load_word_list(args.words_file, args.word)
    extra_patterns = [plain_word_pattern(w, args.case_sensitive) for w in extra_words]

    text = args.input.read_text(encoding="utf-8")
    lines_in = text.splitlines()

    kept: list[str] = []
    removed_lines = 0
    changed_lines = 0

    for line in lines_in:
        new = clean_line(line, extra_patterns)
        if new != line.rstrip():
            changed_lines += 1

        if args.drop_lines and extra_patterns and any(p.search(new) for p in extra_patterns):
            removed_lines += 1
            continue

        if new.strip():
            kept.append(new)
        else:
            removed_lines += 1

    out_text = "\n".join(kept)
    if kept:
        out_text += "\n"

    print(f"Input:  {args.input} ({len(lines_in)} lines)")
    print("Regex:  [[...]], #hashtags, lone trailing #")
    if extra_words:
        print(f"Extra:  {len(extra_words)} ({', '.join(extra_words[:6])}{'...' if len(extra_words) > 6 else ''})")
    print(f"Output: {len(kept)} lines ({changed_lines} changed, {removed_lines} dropped)")

    if args.dry_run:
        return

    out_path = args.output or args.input
    out_path.write_text(out_text, encoding="utf-8")
    print(f"Wrote:  {out_path}")


if __name__ == "__main__":
    main()
