import os
import time
import random
import datetime as dt
from pathlib import Path
from slugify import slugify
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from feedgen.feed import FeedGenerator
import yaml

# --- Config ---
SITE_DIR = Path(__file__).parent
POSTS_DIR = SITE_DIR / "posts"
FEED_FILE = SITE_DIR / "feed.xml"
SITE_TITLE = "AI Diary"
SITE_LINK = "https://eugeniedt.github.io/sytheticdiary"
SITE_DESC = "My Sythetic Diary"
AUTHOR_NAME = "Us"
TIMEZONE = "UTC"
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
MAX_TOKENS = 220
TEMPERATURE = 0.9
TOP_P = 0.95
SEED = int(os.getenv("SEED", str(int(time.time()))[-6:]))   # daily-ish variability
MIN_CHARS = 400
MAX_CHARS = 1500

# --- Prepare model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
        with open(p, "r", encoding="utf-8") as f:
            head = []
            if f.readline().strip() == "---":
                for line in f:
                    if line.strip() == "---":
                        break
                    head.append(line)
            meta = yaml.safe_load("".join(head)) if head else {}
            if meta:
                posts.append((p, meta))
    return posts

def update_rss():
    fg = FeedGenerator()
    fg.title(SITE_TITLE)
    fg.link(href=SITE_LINK, rel='alternate')
    fg.link(href=f"{SITE_LINK}/feed.xml", rel='self')
    fg.description(SITE_DESC)
    fg.language('en')
    fg.author({"name": AUTHOR_NAME})

    posts = load_posts_meta()
    # Sort by date desc
    posts.sort(key=lambda x: x[1].get("date", ""), reverse=True)

    for p, meta in posts:
        fe = fg.add_entry()
        title = meta.get("title", p.stem)
        link = f"{SITE_LINK}/posts/{p.name.replace('.md','.html')}"
        fe.title(title)
        fe.link(href=link)
        fe.id(link)
        fe.pubDate(meta.get("date"))
        # Include excerpt as description
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().split("---", 2)[-1].strip()
        fe.description(content[:280])

    fg.rss_str(pretty=True)
    fg.rss_file(str(FEED_FILE))

def main():
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
    update_rss()
    print("Generated and updated feed.")

if __name__ == "__main__":
    main()