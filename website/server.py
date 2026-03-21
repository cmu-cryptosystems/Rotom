"""
Local server: landing page + layout visualizer web UI.

Run from the Rotom repository root with `.venv` activated:

    source .venv/bin/activate
    pip install -r website/requirements.txt
    PYTHONPATH=. python -m uvicorn website.server:app --reload --host 127.0.0.1 --port 8765

Optional env (see ``website/README.md``): ``ROTOM_WEB_CORS_ORIGINS``,
``ROTOM_WEB_DEBUG``, ``ROTOM_WEB_SKIP_BROWSER_CHECKS``, size limits, etc.

Routes:
    /           — project landing (startup-style)
    /visualizer — interactive layout visualizer
"""

from __future__ import annotations

import ast
import io
import logging
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Rotom package root (parent of website/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_WEBSITE_DIR = Path(__file__).resolve().parent

_ALLOWED_ASSIGNMENTS = frozenset({"layout_str", "n", "tensor_shape", "secret"})

# Web UI DoS bounds (tune via env if needed)
_MAX_LAYOUT_STR_LEN = int(os.environ.get("ROTOM_WEB_MAX_LAYOUT_LEN", "16384"))
_MAX_N = int(os.environ.get("ROTOM_WEB_MAX_N", "65536"))
_MAX_TENSOR_ELEMENTS = int(os.environ.get("ROTOM_WEB_MAX_TENSOR_ELEMENTS", "250000"))
_MAX_TENSOR_RANK = int(os.environ.get("ROTOM_WEB_MAX_TENSOR_RANK", "16"))

logger = logging.getLogger("rotom.web")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _cors_allow_origins() -> list[str]:
    raw = os.environ.get("ROTOM_WEB_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


# Expected by the bundled visualizer; HTML forms cannot set this header (CSRF mitigation).
_API_CLIENT_HEADER = "x-rotom-client"
_API_CLIENT_VALUE = "web-v1"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline headers for static pages and API responses."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        return response


class BrowserPostProtectionMiddleware(BaseHTTPMiddleware):
    """
    Reduce drive-by CSRF / cross-site POST abuse:
    - Require a custom header (same-origin fetch can send it; simple HTML forms cannot).
    - Reject Sec-Fetch-Site: cross-site (modern browsers).
    - If Origin is sent, it must match the CORS allowlist.

    curl/scripts: send ``X-Rotom-Client: web-v1`` or set ``ROTOM_WEB_SKIP_BROWSER_CHECKS=1``.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.method != "POST" or not request.url.path.startswith("/api/"):
            return await call_next(request)
        if _env_flag("ROTOM_WEB_SKIP_BROWSER_CHECKS"):
            return await call_next(request)

        if request.headers.get(_API_CLIENT_HEADER) != _API_CLIENT_VALUE:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Missing or invalid X-Rotom-Client header "
                    "(use the web UI or see website/README.md)."
                },
            )

        sec_fetch_site = (request.headers.get("sec-fetch-site") or "").lower()
        if sec_fetch_site == "cross-site":
            return JSONResponse(
                status_code=403,
                content={"detail": "Cross-site POST blocked (Sec-Fetch-Site)."},
            )

        origin = request.headers.get("origin")
        if origin:
            allowed = _cors_allow_origins()
            if origin not in allowed:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin not allowed for API requests."},
                )

        return await call_next(request)


def _internal_server_error(exc: BaseException) -> HTTPException:
    logger.exception("Request failed: %s", exc)
    if _env_flag("ROTOM_WEB_DEBUG"):
        return HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error. Set ROTOM_WEB_DEBUG=1 for details.",
    )


def _enforce_web_limits(params: dict) -> None:
    ls = params["layout_str"]
    if not isinstance(ls, str) or len(ls) > _MAX_LAYOUT_STR_LEN:
        raise ValueError(
            f"layout_str must be at most {_MAX_LAYOUT_STR_LEN} characters."
        )

    n = int(params["n"])
    if n < 1 or n > _MAX_N:
        raise ValueError(f"n must be between 1 and {_MAX_N}.")

    ts = params.get("tensor_shape")
    if ts is not None:
        if len(ts) > _MAX_TENSOR_RANK:
            raise ValueError(
                f"tensor_shape may have at most {_MAX_TENSOR_RANK} dimensions."
            )
        total = _product(ts)
        if total > _MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"Tensor size (product of shape) must be at most {_MAX_TENSOR_ELEMENTS}."
            )


def _node_to_literal(node: ast.AST):
    """Convert an AST expression node to a Python value (literals only)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_node_to_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_node_to_literal(elt) for elt in node.elts)
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def _product(values: tuple[int, ...]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def infer_shape_from_layout_str(layout_str: str) -> tuple[int, ...] | None:
    """
    Infer tensor shape from layout dims like [0:4:1][1:4:1].

    Uses numeric dimension ids only (ignores R/G and empty dims).
    """
    # For a repeated tensor dim across layout terms, apply_layout combines them by
    # summing stride-weighted contributions. So an inferred upper bound is:
    # size = 1 + sum((extent - 1) * stride) over all terms for that dim.
    dim_max_index: dict[int, int] = {}
    for token in re.findall(r"\[([^\]]+)\]", layout_str):
        parts = [p.strip() for p in token.split(":")]
        if len(parts) < 2:
            continue
        dim_id_raw, extent_raw = parts[0], parts[1]
        if not dim_id_raw.isdigit():
            continue
        try:
            dim_id = int(dim_id_raw)
            extent = int(extent_raw)
            stride = int(parts[2]) if len(parts) >= 3 else 1
        except ValueError:
            continue
        if dim_id < 0 or extent < 1 or stride < 1:
            continue
        contrib = (extent - 1) * stride
        dim_max_index[dim_id] = dim_max_index.get(dim_id, 0) + contrib

    if not dim_max_index:
        return None
    shape = [1] * (max(dim_max_index.keys()) + 1)
    for dim_id, max_index in dim_max_index.items():
        shape[dim_id] = int(max_index) + 1
    return tuple(shape)


def infer_n_from_layout_str(layout_str: str) -> int | None:
    """
    Infer HE vector slot count `n` from the layout string.

    Rule:
      - If there is a `;`, compute the product of extents of bracket terms after `;`.
      - Otherwise, compute the product of extents of all bracket terms.

    This treats bracket terms like:
      - [i:n:s] / [R:n:s]  -> extent = n
      - [R:n]             -> extent = n
      - [G:n]             -> extent = n
    """
    traversal_str = layout_str.split(";", 1)[1] if ";" in layout_str else layout_str

    extents: list[int] = []
    for token in re.findall(r"\[([^\]]+)\]", traversal_str):
        parts = [p.strip() for p in token.split(":")]
        if not parts:
            continue

        extent: int | None = None
        try:
            if len(parts) == 1:
                extent = int(parts[0])
            elif len(parts) == 2 and parts[0] in {"R", "G"}:
                extent = int(parts[1])
            elif len(parts) == 3:
                # [i:n:s] or [R:n:s]
                extent = int(parts[1])
        except ValueError:
            continue

        if extent is not None and extent >= 1:
            extents.append(int(extent))

    if not extents:
        return None

    out = 1
    for e in extents:
        out *= e
    return out


def extract_visualize_params(code: str) -> dict:
    """
    Parse top-level assignments from user editor code.
    Only literal values are allowed; no imports or calls.
    """
    stripped = code.strip()
    if not stripped:
        raise ValueError('Required: layout_str = "..." or a raw layout string.')

    # Convenience mode: allow raw layout text directly in the editor.
    # Example: roll(0,1) [0:4:1][1:4:1]
    if "=" not in stripped and "\n" not in stripped:
        inferred_shape = infer_shape_from_layout_str(stripped)
        inferred_n = infer_n_from_layout_str(stripped)
        params = {
            "layout_str": stripped,
            "tensor_shape": inferred_shape,
            "n": inferred_n
            if inferred_n is not None
            else (_product(inferred_shape) if inferred_shape else 16),
            "secret": False,
        }
        _enforce_web_limits(params)
        return params

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from e

    params: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Only simple single-target assignments are allowed.")
            name = node.targets[0].id
            if name not in _ALLOWED_ASSIGNMENTS:
                continue
            try:
                params[name] = _node_to_literal(node.value)
            except ValueError as e:
                raise ValueError(f"Invalid value for {name}: {e}") from e
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # allow docstring-style string literals at top level
            if isinstance(node.value.value, str):
                continue
            raise ValueError(
                "Only assignments and comments (string literals) are allowed at top level."
            )
        else:
            raise ValueError(
                f"Disallowed statement: use only assignments to "
                f"{', '.join(sorted(_ALLOWED_ASSIGNMENTS))}."
            )

    if "layout_str" not in params:
        raise ValueError('Required: layout_str = "..."')
    if not isinstance(params["layout_str"], str):
        raise ValueError("layout_str must be a string.")

    ts = params.get("tensor_shape")
    if ts is not None:
        if not isinstance(ts, (list, tuple)) or not all(
            isinstance(x, int) and x > 0 for x in ts
        ):
            raise ValueError(
                "tensor_shape must be a list/tuple of positive integers, or omit it."
            )
        params["tensor_shape"] = tuple(ts)
    else:
        inferred_shape = infer_shape_from_layout_str(params["layout_str"])
        if inferred_shape is not None:
            params["tensor_shape"] = inferred_shape

    n = params.get("n")
    if n is not None:
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
    else:
        inferred_n = infer_n_from_layout_str(params["layout_str"])
        if inferred_n is not None:
            params["n"] = inferred_n
        elif params.get("tensor_shape"):
            params["n"] = _product(params["tensor_shape"])
        else:
            params["n"] = 16

    sec = params.get("secret", False)
    if not isinstance(sec, bool):
        raise ValueError("secret must be True or False.")
    params["secret"] = sec

    _enforce_web_limits(params)
    return params


class RunRequest(BaseModel):
    code: str = Field(..., description="Editor contents (Python-style assignments).")


class RunModeRequest(BaseModel):
    mode: str = Field("visualize", description="visualize | demo | full")


app = FastAPI(title="Rotom — Web UI")

# Order: last added runs outermost on the request path.
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BrowserPostProtectionMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Rotom-Client"],
)


@app.get("/")
def landing():
    """Project landing page (startup-style: paper, visualizer CTA, GitHub)."""
    return FileResponse(_WEBSITE_DIR / "index.html")


@app.get("/visualizer")
def visualizer_page():
    """Interactive layout visualizer (Monaco + diagram)."""
    return FileResponse(_WEBSITE_DIR / "visualizer.html")


@app.post("/api/run")
def api_run(req: RunRequest):
    from layout_visualizer import visualize_for_web, visualize_layout

    try:
        p = extract_visualize_params(req.code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            if p.get("tensor_shape"):
                viz = visualize_for_web(
                    p["layout_str"],
                    p["n"],
                    p["tensor_shape"],
                    p["secret"],
                )
            else:
                visualize_layout(
                    p["layout_str"],
                    p["n"],
                    None,
                    p["secret"],
                )
                viz = None
    except Exception as e:
        raise _internal_server_error(e) from e

    return {"ok": True, "output": buf.getvalue(), "viz": viz}


@app.post("/api/run-demo")
def api_run_demo(req: RunModeRequest):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            if req.mode == "demo":
                from layout_visualizer import demo_layout_examples

                demo_layout_examples()
            elif req.mode == "full":
                from layout_visualizer import compare_layouts, demo_layout_examples

                demo_layout_examples()
                print("\n" + "=" * 60)
                print("COMPARISON EXAMPLE")
                print("=" * 60)
                layouts_to_compare = [
                    "[0:4:1][1:4:1]",
                    "[1:4:1][0:4:1]",
                    "roll(0,1) [0:4:1][1:4:1]",
                    "[R:4:1];[0:4:1][1:4:1]",
                ]
                compare_layouts(layouts_to_compare, 16, (4, 4))
            else:
                raise HTTPException(
                    status_code=400, detail='mode must be "demo" or "full"'
                )
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_server_error(e) from e

    return {"ok": True, "output": buf.getvalue()}
