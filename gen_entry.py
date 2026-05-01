import os
import time
import random
import datetime as dt
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
CSS_FILE = ASSETS_DIR / "style.css"
SITE_TITLE = os.getenv("SITE_TITLE", "AI Diary")
SITE_LINK = os.getenv("SITE_LINK", "https://eugeniedt.github.io/syntheticdiary")
SITE_DESC = os.getenv("SITE_DESC", "My Synthetic Diary")
AUTHOR_NAME = os.getenv("AUTHOR_NAME", "Us")
TIMEZONE = "UTC"
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
MAX_TOKENS = 220
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = int(os.getenv("SEED", str(int(time.time()))[-6:]))   # daily-ish variability
MIN_CHARS = 400
MAX_CHARS = 1500
MIN_DAYS_BETWEEN_POSTS = float(os.getenv("MIN_DAYS_BETWEEN_POSTS", "2"))

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

def _page_shell(*, title: str, body_html: str, canonical_path: str) -> str:
    full_title = SITE_TITLE if (not title or title == SITE_TITLE) else f"{SITE_TITLE} — {title}"
    canonical_url = f"{SITE_LINK}{canonical_path}"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{_html_escape_title(full_title)}</title>
    <meta name="description" content="{_html_escape_title(SITE_DESC)}" />
    <link rel="canonical" href="{canonical_url}" />
    <link rel="stylesheet" href="/assets/style.css" />
    <link rel="alternate" type="application/rss+xml" title="{_html_escape_title(SITE_TITLE)}" href="/feed.xml" />
    <script>
      (function() {{
        try {{
          var t = localStorage.getItem('theme');
          if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t;
        }} catch (e) {{}}
      }})();
    </script>
  </head>
  <body>
    <header class="site-header">
      <div class="container header-row">
        <a class="brand" href="/">{_html_escape_title(SITE_TITLE)}</a>
        <nav class="nav">
          <a href="/feed.xml">RSS</a>
          <button class="theme-toggle" type="button" id="themeToggle" aria-label="Toggle theme">Theme</button>
        </nav>
      </div>
    </header>
    <main class="container">
      {body_html}
    </main>
    <footer class="container site-footer">
      <p>Generated content. Static site hosted on GitHub Pages.</p>
    </footer>
    <script>
      (function() {{
        var btn = document.getElementById('themeToggle');
        if (!btn) return;
        btn.addEventListener('click', function() {{
          var cur = document.documentElement.dataset.theme;
          var next = (cur === 'dark') ? 'light' : 'dark';
          document.documentElement.dataset.theme = next;
          try {{ localStorage.setItem('theme', next); }} catch (e) {{}}
        }});
      }})();
    </script>
  </body>
</html>
"""

def ensure_assets():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if CSS_FILE.exists():
        return
    CSS_FILE.write_text(
        """/* Minimal, fast static styling with light/dark theme */
:root{
  --bg:#ffffff;
  --fg:#111827;
  --muted:#6b7280;
  --card:#f3f4f6;
  --link:#2563eb;
  --border:#e5e7eb;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
  --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
:root[data-theme="dark"]{
  --bg:#0b1020;
  --fg:#e5e7eb;
  --muted:#9ca3af;
  --card:#0f172a;
  --link:#60a5fa;
  --border:#1f2937;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme]){--bg:#0b1020;--fg:#e5e7eb;--muted:#9ca3af;--card:#0f172a;--link:#60a5fa;--border:#1f2937;}
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  font-family:var(--sans);
  background:var(--bg);
  color:var(--fg);
  line-height:1.6;
}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
.container{max-width:860px;margin:0 auto;padding:24px}
.site-header{border-bottom:1px solid var(--border);background:rgba(0,0,0,0)}
.header-row{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-top:16px;padding-bottom:16px}
.brand{font-weight:700;letter-spacing:.2px}
.nav{display:flex;gap:14px;align-items:center}
.theme-toggle{
  border:1px solid var(--border);
  background:var(--card);
  color:var(--fg);
  padding:6px 10px;
  border-radius:10px;
  cursor:pointer;
}
.card{border:1px solid var(--border);background:var(--card);border-radius:14px;padding:16px}
.post-meta{color:var(--muted);font-family:var(--mono);font-size:.9rem}
.post-list{display:flex;flex-direction:column;gap:12px;padding:0;list-style:none}
.post-list li{padding:14px;border:1px solid var(--border);border-radius:14px;background:var(--card)}
.site-footer{border-top:1px solid var(--border);color:var(--muted);padding-top:16px}
code,pre{font-family:var(--mono)}
pre{overflow:auto;padding:12px;border-radius:12px;border:1px solid var(--border);background:var(--card)}
""",
        encoding="utf-8",
    )

def prompt_for_today():
    today = dt.datetime.now(dt.timezone.utc).strftime("%A, %B %d, %Y")
    # Keep prompt stable but date-specific for variety
    return (
        f"Daily diary entry\n"
        f"Date: {today}\n\n"
        f"Write a reflective, personal, calm diary entry (first-person), ~6-10 sentences, grounded and specific, no harmful or offensive content. "
        f"Keep it PG and avoid named real people. End cleanly without trailing punctuation repetition.\n\n"
        f"Entry:\n"
    )

def generate_text():
    set_seed(SEED)
    prompt = prompt_for_today()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the portion after "Entry:" to avoid prompt bleed
    if "Entry:" in text:
        text = text.split("Entry:", 1)[1].strip()
    return text

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

def save_post(title: str, content: str, pub_dt: dt.datetime):
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(title)
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

    post_list_html = "<ul class='post-list'>"
    for p, meta in diary_items[:30]:
        title = str(meta.get("title") or p.stem)
        date_s = str(meta.get("date") or "")
        html_name = p.name.replace(".md", ".html")
        post_list_html += (
            f"<li><div class='post-meta'>{_html_escape_title(date_s)}</div>"
            f"<a href='/posts/{html_name}'>{_html_escape_title(title)}</a></li>"
        )
    post_list_html += "</ul>"

    if home_md:
        home_title = str(home_meta.get("title") or SITE_TITLE)
        home_html = _render_markdown_to_html(home_body)
        index_body = (
            f"<article class='card'>{home_html}</article>"
            f"<h2>Recent entries</h2>{post_list_html}"
        )
    else:
        index_body = f"<h1>{_html_escape_title(SITE_TITLE)}</h1><p>{_html_escape_title(SITE_DESC)}</p><h2>Recent entries</h2>{post_list_html}"

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
    for p, meta in posts:
        _, body = _parse_front_matter(p)
        title = str(meta.get("title") or p.stem)
        date_s = str(meta.get("date") or "")
        content_html = _render_markdown_to_html(body)
        post_body = (
            f"<article class='card'>"
            f"<h1>{_html_escape_title(title)}</h1>"
            f"<div class='post-meta'>{_html_escape_title(date_s)}</div>"
            f"{content_html}"
            f"</article>"
        )
        out = POSTS_DIR / p.name.replace(".md", ".html")
        out.write_text(
            _page_shell(title=title, body_html=post_body, canonical_path=f"/posts/{out.name}"),
            encoding="utf-8",
        )

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
        link = f"{SITE_LINK}/posts/{p.name.replace('.md', '.html')}"
        fe.title(title)
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

def main():
    posts = load_posts_meta()
    last_dt = latest_diary_date(posts)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    if last_dt is not None:
        delta_days = (now - last_dt).total_seconds() / (24 * 3600)
        if delta_days < MIN_DAYS_BETWEEN_POSTS:
            # Still rebuild static pages/feed to keep consistent, but skip generation.
            build_static_pages()
            update_rss()
            print(f"Skipped generation (last entry {delta_days:.2f} days ago).")
            return

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    title = f"Diary — {now.strftime('%Y-%m-%d')}"
    text = generate_text()
    text = basic_clean(text)
    if len(text) < MIN_CHARS:
        # Regenerate once if too short
        time.sleep(1)
        text2 = generate_text()
        text2 = basic_clean(text2)
        if len(text2) > len(text):
            text = text2

    save_post(title, text, now)
    build_static_pages()
    update_rss()
    print("Generated and updated feed.")

if __name__ == "__main__":
    main()