## Synthetic Diary (static GitHub Pages)

This repo hosts a fully static website on GitHub Pages.

- **`gen_entry.py`** generates a new diary entry (Markdown source in `posts/`) and produces:
  - `index.html`
  - `posts/*.html`
  - `feed.xml` (RSS)
  - `assets/style.css`
- A scheduled GitHub Action runs the generator daily, but the script only creates a new entry every couple of days (configurable).

### Local usage

```bash
python -m pip install -r requirements.txt
python gen_entry.py
```

### GitHub Pages setup

- In your repo settings, enable GitHub Pages and serve from the default branch (root).
- The workflow commits generated static files back into the repo; Pages serves them as-is.

