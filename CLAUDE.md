# Claude Context — danieltabach.github.io

## Project Overview

Personal portfolio site for Danny Tabach. Jekyll + GitHub Pages using Minimal Mistakes remote theme (`mmistakes/minimal-mistakes`), default skin, light theme.

**Owner:** Danny Tabach — Data Scientist at Chase
**Repo:** `danieltabach/danieltabach.github.io`
**Branch strategy:** `main` is production (deploys via GitHub Pages). Feature branches like `draft/staggered-did-post` for in-progress work.

---

## Site Architecture

### Key Directories
- `_posts/` — Blog posts (Jekyll markdown with YAML front matter)
- `_pages/` — Static pages (works.md, about, etc.)
- `_includes/head/custom.html` — MathJax config, global custom CSS
- `_data/ui-text.yml` — Locale overrides (e.g., "Other Works" heading)
- `assets/images/posts/<post-slug>/` — Per-post image directories
- `assets/data/` — Scripts, datasets referenced by posts
- `assets/css/main.scss` — Global SCSS overrides (imports Minimal Mistakes)
- `_config.yml` — Site-wide config, author info, plugins, defaults

### Theme & Styling
- **Remote theme**: `mmistakes/minimal-mistakes` — do NOT install locally
- **Skin**: `default` (light)
- **All custom CSS must target light theme**. Previous sessions found hardcoded dark theme colors (#333, #444, #1a1a1a) in `_pages/works.md` that had to be fixed. Always use light-appropriate colors: borders `#ddd`, backgrounds `#f8f9fa`/`#f5f5f5`, muted text `#999`, tag chips `#e9ecef`/`#495057`.
- Image captions use `*italic text*` immediately after `![alt](path)`. The CSS in `custom.html` styles `img + em` as centered, gray, 0.9em.

### MathJax Setup (Tricky — Read This)
MathJax v2 is loaded in `_includes/head/custom.html`. Key lessons:

1. **`skipTags` must NOT include `'details'`**. It previously did, which blocked all LaTeX rendering inside collapsible sections. Current correct list: `['script', 'noscript', 'style', 'textarea', 'pre', 'code']`.
2. **`<details>` blocks need `markdown="1"`**. Without this attribute, Kramdown treats content inside `<details>` as raw HTML and does NOT process fenced code blocks (```). This means code stays as raw text, and MathJax then interprets Python underscores as subscripts, `*` as math operators, `>=` as `≥`, etc. The fix: `<details markdown="1">`.
3. **Re-typeset on toggle**. MathJax v2 only typesets on page load. A toggle event listener in `custom.html` re-typesets when a `<details>` block is opened.
4. **`ignoreClass`** includes `language-python|language-bash|highlight` to protect code blocks from MathJax processing.

### Posts Page / Works Page (`_pages/works.md`)
- Three sections: "Models From Scratch" (from-scratch category), "Applied Projects" (applied category), "Data Visualization" (data-viz category)
- Uses Liquid `where_exp` filters on post categories
- Standalone project card for Drift Detection (links to GitHub)
- Tag chips are inline `<span>` elements, NOT functional filters — they're display-only labels
- The "Other Works" / related posts heading is controlled by `_data/ui-text.yml` (`related_label`)

---

## Writing Style & Tone Preferences

### Do
- Write in first person plural ("we") when walking through analysis, or second person ("you") when addressing the reader
- Use contractions: don't, can't, isn't, doesn't, won't, it's — **always check for missing apostrophes**
- Conversational but precise. Sound like a practitioner explaining to a peer, not a textbook
- Use concrete examples and analogies (sports analogies work well — "mid-season roster overhaul")
- Be upfront about tradeoffs and limitations. Don't oversell
- Bold key terms on first use: **composition bias**, **parallel trends assumption**, **LATE**
- Image captions in italics, descriptive but concise

### Don't
- Use em-dashes (—). Use commas, periods, or parentheses instead
- Use cooking/kitchen/recipe metaphors. Sports and mechanical analogies are fine
- Use "we require" or overly formal academic language
- Use "straightforward" (overused, prefer "clean" or just say what it does)
- Add emojis to blog posts
- Create README or documentation files unless explicitly asked
- Exaggerate visual scales to make results look more dramatic than they are

### Tone Calibration
The writing should read like a senior IC explaining their approach to a smart colleague who hasn't worked on this specific problem. Technical depth is welcome but jargon should be explained on first use. Every section should answer "why does this matter?" not just "what is this?"

---

## Blog Post Structure (Learned Patterns)

### Front Matter Template
```yaml
---
layout: single
title: "Title Here"
date: YYYY-MM-DD
categories: [applied]  # or [from-scratch], [data-viz]
tags: [python, relevant-tags]
author_profile: true
classes: wide
header:
  teaser: /assets/images/posts/<slug>/teaser-image.png
toc: false
---
```

### Section Flow (Staggered DiD post as reference)
1. **Hook** — one-line italic premise
2. **Introduction** — set the practical scenario, state the problem
3. **Why it matters** — show what goes wrong with simpler approaches
4. **The solution** — build intuition before math
5. **Implementation details** — filtering, sample construction, validation
6. **Results** — numbers in a table, visual confirmation, interpretation
7. **What goes wrong without this** — benchmark comparison (show your method earns its keep)
8. **Limitations** — be honest, builds credibility
9. **Summary** — concise, actionable takeaways
10. **References** — academic citations with DOI links
11. **Appendix** — formal math (in `<details>`) then full code (in `<details>`)

### Appendix Best Practices
- Wrap each section in `<details markdown="1"><summary><strong>Title</strong></summary>`
- Keep code line width under ~80 chars to prevent horizontal overflow
- Use clear variable names and section comments in code
- Separate appendix from main body with `---` dividers
- Plotting code can be omitted from inline appendix (note that full script with plots is available as download)

---

## Simulation Scripts (Python)

### General Approach
- Scripts live in `assets/data/`
- Output images go to `assets/images/posts/<post-slug>/`
- Use `np.random.seed(42)` for reproducibility
- Functions should be well-documented with docstrings
- The script's `main()` function should: generate data, run analysis, print key numbers, save all plots, then verify all expected image files exist

### Matplotlib Preferences
- **Multi-panel charts**: Stack vertically (`plt.subplots(N, 1)`) not horizontally — gives each panel more room and reads better on mobile
- **Figure size**: `(12, 5*N)` for N vertical panels works well
- **Y-axis**: Never hardcode ylim to exaggerate effects. Let matplotlib auto-scale, or use a generous manual range
- **Annotations**: Use `ax.annotate()` with arrows for callouts. Keep annotation text short
- **Colors**: Use a consistent palette. Treatment = blue-ish, Control = gray/green-ish, highlights = red/orange for problem areas
- **Markers**: Use visible marker sizes (8+ for line plots, 80+ for scatter)
- **Subtitles**: Add descriptive subtitles under panel titles to explain what each panel shows
- `plt.tight_layout()` then `plt.savefig(..., dpi=150, bbox_inches='tight')`

### Chart Design Lessons
- If two panels look identical, they're not communicating different things. Redesign
- Early vs late adopter splits are more informative than aggregate treatment lines for showing TWFE problems
- Control comparison charts should show concrete examples (3 specific locations with arrows), not abstract aggregates
- The parallel trends chart should show full scale — don't zoom the Y-axis to make a small divergence look large

---

## Common Pitfalls (Things That Broke Before)

### Kramdown + HTML Blocks
Kramdown does NOT process markdown inside raw HTML tags. If you write:
```html
<details>
```python
code here
```
</details>
```
The fenced code block stays as raw text. Fix: use `<details markdown="1">`.

### MathJax + Code Conflicts
MathJax will process any text that isn't inside `<pre>` or `<code>` tags. If Kramdown doesn't convert your fenced code to `<pre><code>` (see above), MathJax turns Python into gibberish: `_` becomes subscript, `*` becomes operators, `>=` becomes `≥`.

### Apostrophe Drops
When writing or editing large blocks of text, contractions sometimes lose their apostrophes. Always grep for `\b(dont|cant|isnt|doesnt|wont|wouldnt|shouldnt|couldnt|hasnt|havent|didnt|wasnt|werent|arent)\b` before finalizing.

### Dark Theme Colors in Light Theme
The site uses the default (light) Minimal Mistakes skin. Any inline CSS must use light-appropriate colors. Watch for: `#333`, `#444`, `#1a1a1a`, `#252a34` — these are dark theme artifacts.

### Image References
Every `![alt](/assets/images/posts/slug/name.png)` must have a corresponding file. After regenerating plots, verify the file list matches all references in the post. The simulation script's `main()` should print a verification checklist.

### Git Branch
Active development happens on feature branches. The `draft/staggered-did-post` branch has all DiD post work. Don't push directly to `main` without explicit instruction.

---

## Numbers Reference (Staggered DiD Post)

These are the current simulation outputs with `seed=42`. If the simulation is re-run and these change, update the blog post to match:

| Metric | Value |
|--------|-------|
| Treatment pre-post change | +6.08 |
| Control pre-post change | +2.10 |
| DiD estimate | +3.98 |
| Naive estimate | 3.29 (-27% bias) |
| TWFE estimate | 3.79 (-16% bias) |
| Event-study estimate | 3.94 (-12% bias) |
| True effect (built-in) | 4.5 |

---

## Workflow Preferences

1. **Plan before building**: For multi-step changes, enter plan mode and get approval before writing code
2. **Regenerate plots after script changes**: Always re-run the simulation after modifying plotting or data functions
3. **Verify end-to-end**: After changes, run `bundle exec jekyll serve` (or use preview) to check rendering
4. **Commit with detail**: Commit messages should list what changed and why, organized by category
5. **One thing at a time**: When fixing multiple issues, work through them systematically. Don't try to fix everything in one giant edit
6. **Read before editing**: Always read the current file state before making edits. The file may have changed since last read
7. **Check the rendered output**: Screenshots/previews catch things that code review misses (especially MathJax, image sizing, CSS)
