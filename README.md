## Synthetic Diary (static GitHub Pages)

This repo hosts a fully static website on GitHub Pages.

- **`gen_entry.py`** generates a new diary entry (Markdown source in `posts/`) and produces:
  - `index.html`
  - `posts/*.html`
  - `feed.xml` (RSS)
  - `assets/style.css`
- A scheduled GitHub Action runs the generator daily, but the script only creates a new entry every couple of days (configurable).

### Manual (backend-only) generation

- Normal (auto) behavior, respects `MIN_DAYS_BETWEEN_POSTS`:
  - `python gen_entry.py`
- Manual / forced behavior (always generates a new entry, even if it's too soon):
  - `python gen_entry.py --force`
  - (alias) `python gen_entry.py --manual`
- Optional helpers:
  - `python gen_entry.py --force --date 2026-05-27`
  - `python gen_entry.py --force --seed 12345`

### TO DO 
- rss without html? 
- retrain the model and include into gen_entry 


