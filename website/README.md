# Layout visualizer — web UI

Template site with a **Monaco** code editor and a small **FastAPI** backend that runs [`layout_visualizer.py`](../layout_visualizer.py) in-process (same interpreter as the Rotom tree).

## Run locally

From the **Rotom repository root** (parent of `website/`). Use the same `.venv` as the rest of the project ([root README](../README.md): `python -m venv .venv`).

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Rotom deps (numpy, etc.) — omit if already installed
pip install -r website/requirements.txt
PYTHONPATH=. python -m uvicorn website.server:app --reload --host 127.0.0.1 --port 8765
```

- **Landing** (project overview + paper): [http://127.0.0.1:8765/](http://127.0.0.1:8765/)
- **Layout visualizer**: [http://127.0.0.1:8765/visualizer](http://127.0.0.1:8765/visualizer)

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

## API

| Method | Path | Body |
|--------|------|------|
| `POST` | `/api/run` | `{ "code": "<editor text>" }` |
| `POST` | `/api/run-demo` | `{ "mode": "demo" \| "full" }` |

Response: `{ "ok": true, "output": "<captured stdout>", "viz": <object|null> }`.

When `tensor_shape` is set, `viz` includes the tensor grid, packed ciphertext vectors, and per-slot **`entries`** (each slot’s `label` such as `T[i,j]`, row-major **`linear`** index, `kind`: `tensor` | `gap` | `oob`, and `value`). Omit `tensor_shape` and `viz` is `null` (text-only).
