# Layout visualizer — web UI

Static site with a **Monaco** code editor and an in-browser layout engine (pure JavaScript).

The layout visualizer no longer depends on Rotom’s Python runtime, so it can be deployed directly to **GitHub Pages**.

## Run locally

Serve `website/` as static files (no backend needed):

```bash
cd website
python3 -m http.server 8765
```

- **Landing**: `http://127.0.0.1:8765/index.html`
- **Layout visualizer**: `http://127.0.0.1:8765/visualizer.html`

## Deploy to GitHub Pages

This repo includes a GitHub Actions workflow that publishes `website/` to Pages.

1. In your GitHub repo settings, enable Pages with **Source: GitHub Actions**.
2. Push to the default branch; the workflow will deploy automatically.

## Quick instructions

1. In the editor, start with only:
   ```text
   [0:4:1][1:4:1]
   ```
2. Click **Visualize layout** (or press `Ctrl+Enter`).
3. Read the right pane:
   - **Starting tensor** = generated values `0..N-1`
   - **Ciphertext slot packing** = where each tensor value lands
   - **Slot -> tensor mapping** table = exact source per slot
4. Hover any tensor cell or slot to highlight the same linear index across views.

## Editor contract

The editor accepts either:

- a **raw layout string** (recommended), e.g. `roll(0,1) [0:4:1][1:4:1]`
- or **literal assignments** to:

- `layout_str` (string, **required**)
- `n` (optional positive integer; defaults to product of `tensor_shape` or inferred shape, else `16`)
- `tensor_shape` (optional tuple/list of positive ints; inferred from numeric layout dims when omitted)
- `secret` (optional bool, default `False`)

Example (raw):

```text
[0:4:1][1:4:1]
```

Example (assignment mode):

```python
layout_str = "[0:4:1][1:4:1]"
```

Optional explicit override:

```python
layout_str = "[0:4:1][1:4:1]"
n = 32
tensor_shape = (4, 4)
secret = False
```

**Run demo examples** and **Run full script** call `demo_layout_examples()` and the same comparison block as `python layout_visualizer.py` without using the editor text.

## Notes

- The visualizer’s “engine” lives in `website/rotom_layout.js` (a small JS port of the parsing + slot-mapping logic).
- The legacy FastAPI server (`website/server.py`) is no longer required for the website to function.
