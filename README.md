## Synthetic Diary (static GitHub Pages)

This repo hosts a fully static website on GitHub Pages.

- **`gen_entry.py`** generates a new diary entry (Markdown source in `posts/`) and produces:
  - `index.html`
  - `posts/*.html`
  - `feed.xml` (RSS)
  - `assets/style.css`
- A scheduled GitHub Action runs the generator daily, but the script only creates a new entry every couple of days (configurable).

### TO DO 
- rss without html? 

