# PeregrineX

Single-page marketing site, served by GitHub Pages from `docs/` on `main`. One static HTML file plus a small `assets/` folder — no build step, no dependencies.

## Structure

- `index.html` — the entire site: styles, copy, and three canvas visualizations (ATC approach scope in the hero, the interactive failure map, the touchdown map). The full 1,999-run baseline dataset is inlined as JSON so the page also works when opened directly from disk.
- `assets/`
  - `spaghetti.webp` — Exhibit A, the real approach-corridor figure from `sim/results/baseline_report/`
  - `worst_landing.mp4` + `worst_landing_poster.jpg` — Exhibit C, instrumented replay of run `285adf5c` (compressed from `sim/results/demo/`)
  - `baseline.json` — the extracted per-run dataset (also inlined in the page; kept here as the source of truth for regeneration)
  - `og.jpg` — social preview image (1200×630)

## Notes

- **Every number and figure on the page is real** — extracted from `sim/results/baseline.parquet` (2,000-run LHS baseline, seed 42, ArduPilot `2b5cebb9`). If the baseline is re-run, regenerate `assets/baseline.json` and re-inline it into the `<script type="application/json" id="baselineData">` block.
- Fonts are loaded from Google Fonts (B612 / B612 Mono — the Airbus cockpit-display typeface — and Barlow Condensed). Everything else is self-contained.
- All animation respects `prefers-reduced-motion`: the hero scope renders a single static frame and scroll reveals are disabled.
- There is deliberately no contact link yet. When a contact address exists, add a mailto to the hero CTA row and the closing section.
- The `og:image` meta tag uses a relative path; switch it to an absolute URL once the site has a domain.

## Preview locally

```bash
python3 -m http.server 4173 --directory .
```
