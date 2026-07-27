# PeregrineX

Single-page marketing site. One static file, no build step, no dependencies.

- `index.html` is the entire site. Serve it from any static host, or open it directly.
- There is currently no contact link on the page. When a contact address exists, restore the "Become a design partner" mailto button in the hero (the `.cta` styles are still in the stylesheet).
- The hero visualization renders a static frame for users with `prefers-reduced-motion` enabled. Append `?static=1` to the URL to preview that version.
