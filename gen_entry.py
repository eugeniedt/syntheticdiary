import os
import time
import datetime as dt
import argparse
from pathlib import Path
from slugify import slugify
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from feedgen.feed import FeedGenerator
import yaml
import re
from typing import Optional
import markdown as md

# --- Config ---
SITE_DIR = Path(__file__).parent
POSTS_DIR = SITE_DIR / "posts"
FEED_FILE = SITE_DIR / "feed.xml"
ASSETS_DIR = SITE_DIR / "assets"
SITE_TITLE = os.getenv("SITE_TITLE", "My Synthetic Diary")
SITE_LINK = os.getenv("SITE_LINK", "https://eugeniedt.github.io/syntheticdiary")
SITE_DESC = os.getenv("SITE_DESC", "My Synthetic Diary")
SITE_THEME = os.getenv("SITE_THEME", "default")
SITE_THEMES = {
    "default": "assets/style.css",
}
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Us")
MODEL_NAME = os.getenv("MODEL_NAME", "eugenieee/synthdiary")
TEMP_MIN = 0.1
TEMP_MAX = 1.7
TOKENS_MIN = 50
TOKENS_MAX = 180
TOP_P = 0.95
SEED = int(os.getenv("SEED", str(int(time.time()))[-6:]))   # daily-ish variability
MIN_CHARS = 100
MAX_CHARS = 1500
MIN_DAYS_BETWEEN_POSTS = float(os.getenv("MIN_DAYS_BETWEEN_POSTS", "2"))
ENTRY_PROMPT = "Today, I"

# --- Prepare model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def _parse_front_matter(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].lstrip("\n")
            return meta, body
    return {}, raw

def _is_diary_post(meta: dict, path: Path) -> bool:
    slug = str(meta.get("slug") or "")
    title = str(meta.get("title") or "")
    if slug.startswith("diary-") or title.startswith("Diary"):
        return True
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}-diary-", path.name))

def _html_escape_title(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;").replace("'", "&#39;"))

def _render_markdown_to_html(markdown_text: str) -> str:
    return md.markdown(
        markdown_text,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html5",
    )

def _is_special_page(meta: dict) -> bool:
    permalink = str(meta.get("permalink", "")).strip()
    return permalink in ("/", "/404.html")

def _stylesheet_href() -> str:
    href = SITE_THEMES.get(SITE_THEME, SITE_THEMES["default"])
    if SITE_THEME not in SITE_THEMES:
        print(f"Warning: unknown SITE_THEME {SITE_THEME!r}; using default stylesheet.")
    return href

def _favicon_head_html() -> str:
    tags: list[str] = []
    ico = ASSETS_DIR / "favicon.ico"
    png = ASSETS_DIR / "favicon.png"
    svg = ASSETS_DIR / "favicon.svg"
    apple = ASSETS_DIR / "apple-touch-icon.png"
    if ico.exists():
        tags.append('<link rel="icon" href="assets/favicon.ico" sizes="any" />')
    elif png.exists():
        tags.append('<link rel="icon" href="assets/favicon.png" type="image/png" />')
    elif svg.exists():
        tags.append('<link rel="icon" href="assets/favicon.svg" type="image/svg+xml" />')
    if apple.exists():
        tags.append('<link rel="apple-touch-icon" href="assets/apple-touch-icon.png" />')
    elif png.exists():
        tags.append('<link rel="apple-touch-icon" href="assets/favicon.png" />')
    return "\n    ".join(tags)

def _page_shell(*, title: str, body_html: str, canonical_path: str) -> str:
    full_title = SITE_TITLE if (not title or title == SITE_TITLE) else f"{SITE_TITLE} — {title}"
    canonical_url = f"{SITE_LINK}{canonical_path}"
    stylesheet = _stylesheet_href()
    favicon_html = _favicon_head_html()
    favicon_block = f"\n    {favicon_html}" if favicon_html else ""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{_html_escape_title(full_title)}</title>
    <meta name="description" content="{_html_escape_title(SITE_DESC)}" />
    <link rel="canonical" href="{canonical_url}" />{favicon_block}
    <link rel="stylesheet" href="{stylesheet}" />
    <link rel="alternate" type="application/rss+xml" title="{_html_escape_title(SITE_TITLE)}" href="feed.xml" />
  </head>
  <body>
    <header class="site-header">
      <nav class="nav">
        <a href="feed.xml">RSS</a>
      </nav>
    </header>
    <main class="container">
      {body_html}
    </main>
    <footer class="container site-footer">
      <p>Generated content. Static site hosted on GitHub Pages.</p>
    </footer>
  </body>
</html>
"""

def ensure_assets():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def _fmt_slug_num(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}".replace(".", "-")

def _diary_post_count(posts: list[tuple[Path, dict]]) -> int:
    return sum(1 for p, meta in posts if _is_diary_post(meta, p))

def next_generation_params(posts: list[tuple[Path, dict]]) -> tuple[float, int]:
    """Alternate low/high sampling settings per diary entry."""
    if _diary_post_count(posts) % 2 == 0:
        return TEMP_MIN, TOKENS_MIN
    return TEMP_MAX, TOKENS_MAX

def _diary_slug(pub_dt: dt.datetime, *, temperature: float, max_tokens: int) -> str:
    k = _fmt_slug_num(TOP_P)
    return slugify(
        f"diary-{pub_dt.strftime('%Y-%m-%d')}-t{_fmt_slug_num(temperature)}-k{k}-n{max_tokens}"
    )

def _generation_suffix_from_slug(slug: str) -> Optional[str]:
    """Return sampling params encoded in newer slugs, e.g. t0-9-k0-95-n220."""
    m = re.match(r"^diary-\d{4}-\d{2}-\d{2}-(.+)$", slug)
    if not m:
        return None
    suffix = m.group(1)
    return suffix if suffix.startswith("t") else None

def _slug_num_to_display(value: str) -> str:
    """Undo slug encoding: 0-9 -> 0.9, 0-95 -> 0.95."""
    if "-" in value:
        whole, frac = value.split("-", 1)
        return f"{whole}.{frac}"
    return value

def _generation_params_from_suffix(suffix: str) -> Optional[dict[str, str]]:
    m = re.match(r"^t(.+?)-k(.+?)-n(\d+)$", suffix)
    if not m:
        return None
    temp, top_p, max_tokens = m.groups()
    return {
        "temperature": _slug_num_to_display(temp),
        "top_p": _slug_num_to_display(top_p),
        "max_tokens": max_tokens,
    }

def _friendly_date_short(d: dt.datetime) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"

def _wrap_text_lines(text: str, width: int = 80) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > width:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines

def _meta_line_parts(date_s: str, slug: str) -> list[str]:
    parts = ["Posted automatically"]
    try:
        d = dt.datetime.fromisoformat(date_s)
        parts.append(_friendly_date_short(d))
    except Exception:
        if date_s:
            parts.append(date_s)

    gen_suffix = _generation_suffix_from_slug(slug)
    if gen_suffix:
        params = _generation_params_from_suffix(gen_suffix)
        if params:
            parts.extend([
                f'temp {params["temperature"]}',
                f'top-p {params["top_p"]}',
                f'{params["max_tokens"]} tokens',
            ])
    return parts

def _format_post_entry_html(display_title: str, date_s: str, slug: str, body: str) -> str:
    lines_html = [
        f'<div class="post-line post-line-title">{_html_escape_title(display_title)}</div>',
    ]
    for part in _meta_line_parts(date_s, slug):
        lines_html.append(
            f'<div class="post-line post-line-meta">{_html_escape_title(part)}</div>'
        )
    for line in _wrap_text_lines(body.strip(), 80):
        lines_html.append(
            f'<div class="post-line post-line-body">{_html_escape_title(line)}</div>'
        )
    return f"<article class='card post-entry'>{''.join(lines_html)}</article>"

def _format_post_meta_html(date_s: str, slug: str) -> str:
    """Human-readable meta strip for index entries."""
    iso_attr = _html_escape_title(date_s)
    date_label = date_s
    try:
        d = dt.datetime.fromisoformat(date_s)
        date_label = _friendly_date_short(d)
    except Exception:
        pass

    parts = [
        '<span class="meta-badge">Posted automatically</span>',
        f'<time datetime="{iso_attr}">{_html_escape_title(date_label)}</time>',
    ]
    gen_suffix = _generation_suffix_from_slug(slug)
    if gen_suffix:
        params = _generation_params_from_suffix(gen_suffix)
        if params:
            parts.extend([
                f'<span class="meta-param">temp {_html_escape_title(params["temperature"])}</span>',
                f'<span class="meta-param">top-p {_html_escape_title(params["top_p"])}</span>',
                f'<span class="meta-param">{_html_escape_title(params["max_tokens"])} tokens</span>',
            ])

    inner = '<span class="meta-sep" aria-hidden="true">·</span>'.join(parts)
    return f'<div class="post-meta">{inner}</div>'

def generate_text(*, temperature: float, max_tokens: int) -> str:
    set_seed(SEED)
    inputs = tokenizer(ENTRY_PROMPT, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=TOP_P,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def basic_clean(text: str) -> str:
    # Trim runaway repeats and hard-wrap long whitespace
    text = " ".join(text.split())
    # Truncate if too long
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "..."
    # Simple profanity/NSFW check placeholder (replace with better filter as needed)
    banned = ["hate", "kill", "violent", "nsfw"]  # customize
    if any(b in text.lower() for b in banned):
        raise ValueError("Content filter triggered")
    return text

def save_post(title: str, content: str, pub_dt: dt.datetime, *, slug: str):
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{pub_dt.strftime('%Y-%m-%d')}-{slug}.md"
    path = POSTS_DIR / filename
    front_matter = {
        "title": title,
        "date": pub_dt.isoformat(),
        "slug": slug,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("---\n")
        yaml.dump(front_matter, f, sort_keys=False)
        f.write("---\n\n")
        f.write(content.strip() + "\n")
    return path

def load_posts_meta():
    posts = []
    for p in sorted(POSTS_DIR.glob("*.md")):
        meta, body = _parse_front_matter(p)
        if meta:
            posts.append((p, meta))
    return posts

def latest_diary_date(posts: list[tuple[Path, dict]]) -> Optional[dt.datetime]:
    diary = []
    for p, meta in posts:
        if _is_diary_post(meta, p):
            diary.append((p, meta))
    if not diary:
        return None
    diary.sort(key=lambda x: x[1].get("date", ""), reverse=True)
    ds = diary[0][1].get("date")
    try:
        return dt.datetime.fromisoformat(str(ds))
    except Exception:
        return None

def build_static_pages():
    ensure_assets()

    # Render each markdown file in posts/ to a corresponding HTML file in posts/
    posts = load_posts_meta()
    posts_sorted = sorted(posts, key=lambda x: x[1].get("date", ""), reverse=True)

    # Build index from the page whose permalink is "/" if present (your about.md)
    home_md = None
    home_meta = None
    home_body = ""
    for p, meta in posts:
        if str(meta.get("permalink", "")).strip() == "/":
            home_md = p
            home_meta, home_body = _parse_front_matter(p)
            break

    diary_items = [(p, m) for (p, m) in posts_sorted if _is_diary_post(m, p)]

    post_feed_html = "<div class='post-feed'>"
    for p, meta in diary_items[:30]:
        title = str(meta.get("title") or p.stem)
        date_s = str(meta.get("date") or "")
        slug = str(meta.get("slug") or "")
        # On the index page, show a human-friendly date title like "May, 17th, 2026".
        display_title = title
        try:
            d = dt.datetime.fromisoformat(date_s)
            day = d.day
            if 11 <= (day % 100) <= 13:
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            display_title = f"{d.strftime('%B')}, {day}{suffix}, {d.year}"
        except Exception:
            pass
        _, body = _parse_front_matter(p)
        post_feed_html += _format_post_entry_html(display_title, date_s, slug, body)
    post_feed_html += "</div>"

    if home_md:
        home_html = _render_markdown_to_html(home_body)
        index_body = f"{home_html}<h2>Latest Entries:</h2>{post_feed_html}"
    else:
        index_body = f"<h1>{_html_escape_title(SITE_TITLE)}</h1><p>{_html_escape_title(SITE_DESC)}</p><h2>Latest Entries:</h2>{post_feed_html}"

    (SITE_DIR / "index.html").write_text(
        _page_shell(title=SITE_TITLE, body_html=index_body, canonical_path="/"),
        encoding="utf-8",
    )

    # 404.html (if markdown exists)
    for p, meta in posts:
        if str(meta.get("permalink", "")).strip() == "/404.html":
            _, body = _parse_front_matter(p)
            body_html = f"<article class='card'>{_render_markdown_to_html(body)}</article>"
            (SITE_DIR / "404.html").write_text(
                _page_shell(title=str(meta.get("title") or "404"), body_html=body_html, canonical_path="/404.html"),
                encoding="utf-8",
            )
            break

    # Individual post pages in posts/
    # Intentionally do not generate per-post HTML pages.
    # The site is meant to be browsable via the homepage feed and via RSS.

def update_rss():
    fg = FeedGenerator()
    fg.title(SITE_TITLE)
    fg.link(href=f"{SITE_LINK}/", rel="alternate")
    fg.link(href=f"{SITE_LINK}/feed.xml", rel="self")
    fg.description(SITE_DESC)
    fg.language('en')
    fg.author({"name": AUTHOR_NAME})

    posts = load_posts_meta()
    # Sort by date desc
    posts.sort(key=lambda x: x[1].get("date", ""), reverse=True)

    for p, meta in posts:
        if not _is_diary_post(meta, p):
            continue
        fe = fg.add_entry()
        title = meta.get("title", p.stem)
        gen_suffix = _generation_suffix_from_slug(str(meta.get("slug") or ""))
        rss_title = f"{title} · {gen_suffix}" if gen_suffix else title
        link = f"{SITE_LINK}/posts/{p.name}"
        fe.title(rss_title)
        fe.link(href=link)
        fe.id(link)
        try:
            fe.pubDate(dt.datetime.fromisoformat(str(meta.get("date"))))
        except Exception:
            pass
        # Include excerpt as description
        _, body = _parse_front_matter(p)
        content = " ".join(body.split())
        fe.description(content[:280])

    fg.rss_str(pretty=True)
    fg.rss_file(str(FEED_FILE))

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate diary posts and rebuild static pages/RSS.")
    p.add_argument(
        "--force",
        "--manual",
        action="store_true",
        help="Force-generate a new diary entry even if the last entry is too recent.",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional UTC date for the entry in YYYY-MM-DD (default: now).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed override (default: SEED env/config).",
    )
    return p.parse_args()

def _utc_dt_for_args(args: argparse.Namespace) -> dt.datetime:
    if not args.date:
        return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    try:
        d = dt.date.fromisoformat(args.date)
    except Exception as e:
        raise SystemExit(f"Invalid --date {args.date!r}. Expected YYYY-MM-DD.") from e
    return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).replace(microsecond=0)

def main():
    args = _parse_args()

    if args.seed is not None:
        global SEED
        SEED = int(args.seed)

    posts = load_posts_meta()
    last_dt = latest_diary_date(posts)
    now = _utc_dt_for_args(args)
    if (not args.force) and (last_dt is not None):
        delta_days = (now - last_dt).total_seconds() / (24 * 3600)
        if delta_days < MIN_DAYS_BETWEEN_POSTS:
            # Still rebuild static pages/feed to keep consistent, but skip generation.
            build_static_pages()
            update_rss()
            print(f"Skipped generation (last entry {delta_days:.2f} days ago).")
            return

    title = f"Diary — {now.strftime('%Y-%m-%d')}"
    temperature, max_tokens = next_generation_params(posts)
    text = generate_text(temperature=temperature, max_tokens=max_tokens)
    text = basic_clean(text)
    if len(text) < MIN_CHARS:
        # Regenerate once if too short
        time.sleep(1)
        text2 = generate_text(temperature=temperature, max_tokens=max_tokens)
        text2 = basic_clean(text2)
        if len(text2) > len(text):
            text = text2

    save_post(
        title,
        text,
        now,
        slug=_diary_slug(now, temperature=temperature, max_tokens=max_tokens),
    )
    build_static_pages()
    update_rss()
    print("Generated and updated feed.")

if __name__ == "__main__":
    main()