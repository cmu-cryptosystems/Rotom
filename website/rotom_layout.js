// Minimal, browser-friendly port of Rotom's layout parsing + apply_layout pipeline
// used by `layout_visualizer.py` / `website/server.py`.
//
// Goal: allow `website/visualizer.html` to run fully client-side (GitHub Pages)
// without importing any Rotom Python code.

function prod(values) {
  let out = 1;
  for (const v of values) out *= Number(v);
  return out;
}

function isPowerOfTwo(x) {
  const n = Number(x);
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

export const DimType = Object.freeze({
  FILL: 0,
  EMPTY: 1,
});

export class Dim {
  constructor(dim, extent, stride = 1, dimType = DimType.FILL) {
    this.dim = dim; // number | null
    this.extent = Number(extent);
    this.stride = Number(stride);
    this.dim_type = dimType;
    if (this.dim_type === DimType.EMPTY) {
      if (this.dim !== null) throw new Error("Empty dim must have dim=null");
    }
    if (!(this.stride > 0)) throw new Error("Stride must be positive");
  }

  static parse(bracketed) {
    const terms = bracketed.replace("[", "").replace("]", "").split(":");
    if (terms.length === 1) {
      return new Dim(null, parseInt(terms[0], 10), 1, DimType.FILL);
    }
    if (terms.length === 2 && terms[0] === "G") {
      return new Dim(null, parseInt(terms[1], 10), 1, DimType.EMPTY);
    }
    if (terms.length === 2 && terms[0] === "R") {
      return new Dim(null, parseInt(terms[1], 10), 1, DimType.FILL);
    }
    if (terms.length === 3) {
      if (terms[0] === "R") {
        return new Dim(null, parseInt(terms[1], 10), parseInt(terms[2], 10), DimType.FILL);
      }
      return new Dim(parseInt(terms[0], 10), parseInt(terms[1], 10), parseInt(terms[2], 10), DimType.FILL);
    }
    throw new Error(`Unsupported Dim string: ${bracketed}`);
  }

  toString() {
    if (this.dim_type === DimType.EMPTY) return `[G:${this.extent}]`;
    if (this.dim === null) return `[R:${this.extent}:${this.stride}]`;
    return `[${this.dim}:${this.extent}:${this.stride}]`;
  }
}

export class Roll {
  constructor(dimToRoll, dimToRollBy) {
    if (dimToRoll === dimToRollBy) throw new Error("roll dims must differ");
    if (dimToRoll.extent !== dimToRollBy.extent) throw new Error("roll dims must have same extent");
    this.dim_to_roll = dimToRoll;
    this.dim_to_roll_by = dimToRollBy;
  }

  rollIndex(dims) {
    const a = dims.indexOf(this.dim_to_roll);
    const b = dims.indexOf(this.dim_to_roll_by);
    if (a < 0 || b < 0) throw new Error("roll dims must be in dims");
    return [a, b];
  }
}

export class Layout {
  constructor(rolls, dims, n, secret = false) {
    this.rolls = rolls;
    this.dims = dims;
    this.n = Number(n);
    this.secret = Boolean(secret);

    // Split dims into ct_dims and slot_dims (ported from `ir/layout.py`).
    this.ct_dims = [];
    this.slot_dims = [];
    let slotsLeft = Number(n);
    for (const dim of [...this.dims].reverse()) {
      if (!dim.extent) continue;
      if (slotsLeft <= 1) {
        this.ct_dims.unshift(dim);
      } else if (dim.extent > slotsLeft) {
        if (dim.extent % slotsLeft !== 0) throw new Error("dim extent must be multiple of n");
        const slotSplit = new Dim(dim.dim, slotsLeft, 1, dim.dim_type);
        const ctSplit = new Dim(dim.dim, dim.extent / slotsLeft, slotsLeft, dim.dim_type);
        this.slot_dims.unshift(slotSplit);
        this.ct_dims.unshift(ctSplit);
        slotsLeft = Math.floor(slotsLeft / dim.extent);
      } else if (dim.extent === slotsLeft) {
        this.slot_dims.unshift(dim);
        slotsLeft = Math.floor(slotsLeft / dim.extent);
      } else if (slotsLeft % dim.extent === 0) {
        slotsLeft = Math.floor(slotsLeft / dim.extent);
        this.slot_dims.unshift(dim);
      } else {
        // extent does not divide n -> keep in ct_dims
        this.ct_dims.unshift(dim);
      }
    }
    if (slotsLeft > 1) {
      this.slot_dims.unshift(Dim.parse(`[G:${slotsLeft}]`));
    }

    // In python, empty dims are removed from ct_dims.
    this.ct_dims = this.ct_dims.filter((d) => d.dim_type !== DimType.EMPTY);

    // Slot dims are expected to be power-of-2 (Rotom invariant). For the web UI,
    // enforce only when it is clearly intended; otherwise fail with a good error.
    for (const d of this.slot_dims) {
      if (!isPowerOfTwo(d.extent) || !isPowerOfTwo(d.stride)) {
        throw new Error(`slot dim must be power-of-2 (got extent=${d.extent}, stride=${d.stride})`);
      }
    }

    const allDims = this.getDims();
    for (const r of this.rolls) {
      if (!allDims.includes(r.dim_to_roll) || !allDims.includes(r.dim_to_roll_by)) {
        throw new Error("roll refers to unknown dimension");
      }
    }
  }

  getDims() {
    return [...this.ct_dims, ...this.slot_dims];
  }

  length() {
    return prod(this.getDims().map((d) => d.extent));
  }

  layoutStr() {
    const ctStr = this.ct_dims.map((d) => d.toString()).join("");
    const slotStr = this.slot_dims.map((d) => d.toString()).join("");
    const dims = this.getDims();
    const rolls = this.rolls
      .map((r) => `roll(${dims.indexOf(r.dim_to_roll)},${dims.indexOf(r.dim_to_roll_by)})`)
      .join(" ");
    let layout = ctStr ? `${ctStr};${slotStr}` : `${slotStr}`;
    if (rolls) layout = `${rolls} ${layout}`;
    return layout;
  }

  static fromString(layoutStr, n, secret = false) {
    const rollPattern = /roll\(([^,)]+),([^)]+)\)/g;
    const rollMatches = [...layoutStr.matchAll(rollPattern)].map((m) => [m[1], m[2]]);

    // Remove roll() terms before parsing bracket dims.
    let dimsStr = layoutStr.replace(/roll\([^)]+\)\s*/g, "").trim();

    let ctDimsStr = "";
    if (dimsStr.includes(";")) {
      const parts = dimsStr.split(";");
      ctDimsStr = (parts[0] || "").trim();
      dimsStr = (parts[1] || "").trim();
    }

    const dims = [];
    if (ctDimsStr) {
      const ctMatches = [...ctDimsStr.matchAll(/\[([^\]]+)\]/g)];
      for (const m of ctMatches) dims.push(Dim.parse(`[${m[1]}]`));
    }
    const slotMatches = [...dimsStr.matchAll(/\[([^\]]+)\]/g)];
    for (const m of slotMatches) dims.push(Dim.parse(`[${m[1]}]`));

    const rolls = [];
    const indices = rollMatches.map(([a, b]) => [parseInt(a, 10), parseInt(b, 10)]);
    for (const [fromIdx, toIdx] of indices) {
      if (fromIdx < 0 || toIdx < 0 || fromIdx >= dims.length || toIdx >= dims.length) {
        throw new Error("roll indices out of range");
      }
      rolls.push(new Roll(dims[fromIdx], dims[toIdx]));
    }

    return new Layout(rolls, dims, n, secret);
  }
}

export function inferNFromLayoutStr(layoutStr) {
  const traversal = layoutStr.includes(";") ? layoutStr.split(";", 2)[1] : layoutStr;
  const extents = [];
  for (const m of traversal.matchAll(/\[([^\]]+)\]/g)) {
    const token = m[1];
    const parts = token.split(":").map((p) => p.trim()).filter(Boolean);
    if (!parts.length) continue;
    let extent = null;
    if (parts.length === 1) extent = parseInt(parts[0], 10);
    else if (parts.length === 2) extent = parseInt(parts[1], 10);
    else if (parts.length === 3) extent = parseInt(parts[1], 10);
    if (Number.isFinite(extent) && extent >= 1) extents.push(extent);
  }
  if (!extents.length) return null;
  return extents.reduce((a, b) => a * b, 1);
}

export function inferShapeFromLayoutStr(layoutStr) {
  // Port of `website/server.py::infer_shape_from_layout_str`
  const dimMaxIndex = new Map(); // dimId -> maxIndex (inclusive)
  for (const m of layoutStr.matchAll(/\[([^\]]+)\]/g)) {
    const token = m[1];
    const parts = token.split(":").map((p) => p.trim());
    if (parts.length < 2) continue;
    const dimIdRaw = parts[0];
    const extentRaw = parts[1];
    if (!/^\d+$/.test(dimIdRaw)) continue;
    const dimId = parseInt(dimIdRaw, 10);
    const extent = parseInt(extentRaw, 10);
    const stride = parts.length >= 3 ? parseInt(parts[2], 10) : 1;
    if (!(dimId >= 0 && extent >= 1 && stride >= 1)) continue;
    const contrib = (extent - 1) * stride;
    dimMaxIndex.set(dimId, (dimMaxIndex.get(dimId) || 0) + contrib);
  }
  if (!dimMaxIndex.size) return null;
  const maxDim = Math.max(...dimMaxIndex.keys());
  const shape = Array(maxDim + 1).fill(1);
  for (const [dimId, maxIndex] of dimMaxIndex.entries()) shape[dimId] = Number(maxIndex) + 1;
  return shape;
}

function getSegments(dims) {
  let n = 1;
  for (const d of dims) n *= d.extent;
  const segments = new Map(); // i -> [count, extent, segmentLen]
  for (let i = 0; i < dims.length; i++) {
    const segmentLen = prod(dims.slice(i + 1).map((d) => d.extent));
    const extent = dims[i].extent;
    const count = n / segmentLen / extent;
    segments.set(i, [count, extent, segmentLen]);
  }
  return segments;
}

function getDimIndices(dims) {
  const segments = getSegments(dims);
  const all = [];
  for (let i = 0; i < dims.length; i++) {
    const [iLen, jLen, kLen] = segments.get(i);
    const idx = [];
    for (let ii = 0; ii < iLen; ii++) {
      for (let j = 0; j < jLen; j++) {
        for (let kk = 0; kk < kLen; kk++) idx.push(j);
      }
    }
    all.push(idx);
  }
  return all;
}

function mul(vec, n) {
  return vec.map((v) => (v == null ? null : v * n));
}

function addVec(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    out.push(x != null && y != null ? x + y : null);
  }
  return out;
}

function addVecs(a, b) {
  const out = [];
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (x != null && y != null) out.push(x + y);
    else if (x != null) out.push(x);
    else if (y != null) out.push(y);
    else out.push(null);
  }
  return out;
}

function addVecsOfVecs(a, b) {
  return a.map((x, i) => addVecs(x, b[i]));
}

function getCtIdxsByDim(ctDims, dim) {
  const dimIndex = ctDims.indexOf(dim);
  if (dimIndex < 0) throw new Error("dim not in ctDims");
  const dimIndices = getDimIndices(ctDims);
  const indices = [...dimIndices[dimIndex]];
  const numCt = indices.length;
  if (numCt === dim.extent) return [Array.from({ length: numCt }, (_, j) => j)];
  const groups = [];
  for (let i = 0; i < dim.extent; i++) {
    groups.push(indices.map((v, j) => (v === i ? j : null)).filter((x) => x != null));
  }
  return groups;
}

function rowMajorLinearIndex(coords, shape) {
  let acc = 0;
  for (let i = 0; i < coords.length; i++) acc = acc * Number(shape[i]) + Number(coords[i]);
  return acc;
}

function tensorGet(tensor, coords) {
  let node = tensor;
  for (const c of coords) node = node[c];
  return node;
}

export function applyLayout(ptTensor, layout, { returnSlotMapping = false } = {}) {
  const layoutLen = Math.max(layout.length(), layout.n);
  const dims = layout.getDims();
  let dimIndices = getDimIndices(dims);

  // Apply rolls.
  for (const roll of layout.rolls) {
    const [a, b] = roll.rollIndex(dims);
    const mod = roll.dim_to_roll.extent;
    const rolled = [];
    for (let i = 0; i < layoutLen; i++) {
      rolled.push((dimIndices[a][i] + dimIndices[b][i]) % mod);
    }
    dimIndices[a] = rolled;
  }

  // Apply strides.
  for (let i = 0; i < dimIndices.length; i++) dimIndices[i] = mul(dimIndices[i], dims[i].stride);

  // Empty dims: keep 0, map non-zero to null.
  for (let i = 0; i < dims.length; i++) {
    if (dims[i].dim_type !== DimType.EMPTY) continue;
    dimIndices[i] = dimIndices[i].map((j) => (!j ? j : null));
  }

  // Combine per-dimension contributions.
  const indicesMap = new Map(); // dimId -> vec
  for (let i = 0; i < dims.length; i++) {
    const d = dims[i];
    if (d.dim == null) continue;
    if (indicesMap.has(d.dim)) indicesMap.set(d.dim, addVec(indicesMap.get(d.dim), dimIndices[i]));
    else indicesMap.set(d.dim, dimIndices[i]);
  }
  for (let i = 0; i < dims.length; i++) {
    const d = dims[i];
    if (!(d.dim == null && d.dim_type === DimType.EMPTY)) continue;
    for (const key of indicesMap.keys()) {
      indicesMap.set(key, addVec(indicesMap.get(key), dimIndices[i]));
    }
  }

  const maxDimId = indicesMap.size ? Math.max(...indicesMap.keys()) : -1;
  const baseIndices = Array.from({ length: layoutLen }, () => Array(maxDimId + 1).fill(0));
  for (const [dimId, indices] of indicesMap.entries()) {
    for (let i = 0; i < indices.length; i++) baseIndices[i][dimId] = indices[i];
  }

  // Split by ciphertexts.
  let baseByCts = [];
  const ctCount = Math.floor(layoutLen / layout.n);
  for (let i = 0; i < ctCount; i++) baseByCts.push(baseIndices.slice(i * layout.n, (i + 1) * layout.n));

  // Combine cts if any ct_dim is empty (normally none due to Layout filtering).
  let combined = baseByCts.slice();
  let ctDims = layout.ct_dims.slice();
  for (const ctDim of layout.ct_dims) {
    if (ctDim.dim_type !== DimType.EMPTY) continue;
    const newCombined = [];
    const groups = getCtIdxsByDim(ctDims, ctDim);
    for (const group of groups) {
      let base = baseByCts[group[0]];
      for (const idx of group.slice(1)) base = addVecsOfVecs(base, baseByCts[idx]);
      newCombined.push(base);
    }
    ctDims = ctDims.filter((d) => d !== ctDim);
    combined = newCombined;
  }
  baseByCts = combined;

  // Tensor shape / ndim.
  const shape = inferArrayShape(ptTensor);
  const ndim = shape.length;
  const cts = [];
  const maps = returnSlotMapping ? [] : null;

  for (let ctIndex = 0; ctIndex < baseByCts.length; ctIndex++) {
    const ctIndices = baseByCts[ctIndex];
    const ct = [];
    const slotMap = returnSlotMapping ? [] : null;
    for (let slotI = 0; slotI < ctIndices.length; slotI++) {
      const index = ctIndices[slotI];
      const effective = index.slice();
      while (effective.length < ndim) effective.push(0);

      if (effective.slice(0, ndim).some((x) => x == null)) {
        ct.push(0);
        if (returnSlotMapping) {
          slotMap.push({ slot: slotI, kind: "gap", coords: null, linear: null, label: "gap (unused)" });
        }
        continue;
      }

      const coords = effective.slice(0, ndim).map((x) => Number(x));
      const oob = coords.some((c, i) => c >= shape[i]);
      if (oob) {
        ct.push(0);
        if (returnSlotMapping) {
          slotMap.push({ slot: slotI, kind: "oob", coords, linear: null, label: "out of range" });
        }
        continue;
      }

      const value = ndim === 0 ? ptTensor : tensorGet(ptTensor, coords);
      ct.push(value);
      if (returnSlotMapping) {
        const linear = ndim === 0 ? 0 : rowMajorLinearIndex(coords, shape);
        const label =
          ndim === 0
            ? "T[]"
            : ndim === 1
              ? `T[${coords[0]}]`
              : ndim === 2
                ? `T[${coords[0]},${coords[1]}]`
                : `T[${coords.join(",")}]`;
        slotMap.push({ slot: slotI, kind: "tensor", coords, linear, label });
      }
    }
    cts.push(ct);
    if (returnSlotMapping) maps.push(slotMap);
  }

  if (returnSlotMapping) return [cts, maps];
  return [cts, null];
}

function inferArrayShape(arr) {
  const shape = [];
  let node = arr;
  while (Array.isArray(node)) {
    shape.push(node.length);
    node = node[0];
  }
  return shape;
}

function flattenNumbers(nested) {
  if (Array.isArray(nested)) return nested.flatMap(flattenNumbers);
  return [Number(nested)];
}

function inferLayoutOrderHint(layoutStrResolved, tensorShape) {
  if (!Array.isArray(tensorShape) || tensorShape.length !== 2) return "other";
  const compact = String(layoutStrResolved).replaceAll(" ", "");
  if (compact.includes("[0:") && compact.includes("[1:")) {
    const idx0 = compact.indexOf("[0:");
    const idx1 = compact.indexOf("[1:");
    if (idx0 !== -1 && idx1 !== -1) return idx0 < idx1 ? "row-major" : "column-major";
  }
  return "other";
}

function inferColorGrouping(slotMaps, tensorShape) {
  if (!Array.isArray(tensorShape) || tensorShape.length !== 2) return { mode: "value" };
  const coordsSeq = [];
  for (const ct of slotMaps) {
    for (const entry of ct) {
      if (!entry || entry.kind !== "tensor") continue;
      const coords = entry.coords;
      if (!Array.isArray(coords) || coords.length < 2) continue;
      coordsSeq.push([Number(coords[0]), Number(coords[1])]);
    }
  }
  if (coordsSeq.length < 2) return { mode: "row" };
  const deltas = [];
  let prev = coordsSeq[0];
  for (const curr of coordsSeq.slice(1)) {
    const dr = curr[0] - prev[0];
    const dc = curr[1] - prev[1];
    if (dr !== 0 || dc !== 0) deltas.push([dr, dc]);
    prev = curr;
  }
  if (!deltas.length) return { mode: "row" };
  const singleDimCounts = [0, 0];
  const multi = [];
  for (const [dr, dc] of deltas) {
    const changed = (dr !== 0 ? 1 : 0) + (dc !== 0 ? 1 : 0);
    if (changed === 1) {
      if (dr !== 0) singleDimCounts[0] += 1;
      else singleDimCounts[1] += 1;
    } else if (changed >= 2) {
      multi.push([dr, dc]);
    }
  }
  if (multi.length >= Math.max(...singleDimCounts)) {
    const diagStep = multi.find(([dr, dc]) => Math.abs(dr) === 1 && Math.abs(dc) === 1) || multi[0];
    return { mode: diagStep[0] * diagStep[1] >= 0 ? "diagonal-sum" : "diagonal-diff" };
  }
  const fastestDim = singleDimCounts[0] >= singleDimCounts[1] ? 0 : 1;
  return { mode: fastestDim === 1 ? "row" : "column" };
}

function buildTensorFromShape(shape) {
  if (!shape || !shape.length) return 0;
  const total = prod(shape);
  const flat = Array.from({ length: total }, (_, i) => i);
  // reshape
  function build(dim, offset) {
    const extent = shape[dim];
    if (dim === shape.length - 1) {
      return flat.slice(offset, offset + extent);
    }
    const step = prod(shape.slice(dim + 1));
    const out = [];
    for (let i = 0; i < extent; i++) out.push(build(dim + 1, offset + i * step));
    return out;
  }
  return build(0, 0);
}

function emitVisualizePrints(layoutStr, tensorShape, layout, tensor, packed) {
  const lines = [];
  lines.push(`=== Layout: ${layoutStr} ===`);
  lines.push(`Test tensor shape: ${JSON.stringify(tensorShape)}`);
  lines.push("Original tensor:");
  lines.push(JSON.stringify(tensor));
  lines.push(`Layout: ${layout.layoutStr()}`);
  lines.push("Packed vector:");
  if (Array.isArray(packed) && packed.length > 1) {
    packed.forEach((ct, i) => lines.push(`Ciphertext ${i}: ${JSON.stringify(ct)}`));
  } else {
    lines.push(JSON.stringify(packed));
  }
  lines.push("");
  return lines.join("\n");
}

export function visualizeForWeb(layoutStrInput, n = null, tensorShape = null, secret = false) {
  const inferredN = n == null ? inferNFromLayoutStr(layoutStrInput) : Number(n);
  const useN = inferredN != null ? inferredN : 16;
  const layout = Layout.fromString(layoutStrInput, useN, secret);
  if (!tensorShape) return { output: "", viz: null };

  const tensor = buildTensorFromShape(tensorShape.map((x) => Number(x)));
  const [packed, slotMaps] = applyLayout(tensor, layout, { returnSlotMapping: true });

  const packedOut = packed.map((ct, i) => {
    const entries = ct.map((val, j) => {
      const m = slotMaps[i][j];
      return { value: val, kind: m.kind, label: m.label, coords: m.coords, linear: m.linear, slot: m.slot };
    });
    return { id: i, slots: ct.slice(), entries };
  });

  const flatVals = [...flattenNumbers(tensor)];
  for (const pv of packedOut) flatVals.push(...pv.slots.map((x) => Number(x)));
  let vmin = flatVals.length ? Math.min(...flatVals) : 0.0;
  let vmax = flatVals.length ? Math.max(...flatVals) : 1.0;
  if (vmin === vmax) vmax = vmin + 1.0;

  const viz = {
    layout_str_input: layoutStrInput,
    layout_str_resolved: layout.layoutStr(),
    layout_order_hint: inferLayoutOrderHint(layout.layoutStr(), tensorShape),
    color_grouping: inferColorGrouping(slotMaps, tensorShape),
    n: useN,
    secret: Boolean(secret),
    tensor_shape: tensorShape.map((x) => Number(x)),
    tensor,
    tensor_ndim: tensorShape.length,
    tensor_index_order: "C",
    num_ciphertexts: packedOut.length,
    packed: packedOut,
    value_range: { min: vmin, max: vmax },
  };

  const output = emitVisualizePrints(layoutStrInput, tensorShape, layout, tensor, packed);
  return { output, viz };
}

export function parseEditorInput(code) {
  // Mirrors `website/server.py::extract_visualize_params`, but in JS.
  const stripped = String(code || "").trim();
  if (!stripped) throw new Error('Required: layout_str = "..." or a raw layout string.');

  // Raw single-line layout string convenience mode.
  if (!stripped.includes("=") && !stripped.includes("\n")) {
    const inferredShape = inferShapeFromLayoutStr(stripped);
    const inferredN = inferNFromLayoutStr(stripped);
    const n = inferredN != null ? inferredN : inferredShape ? prod(inferredShape) : 16;
    return { layout_str: stripped, tensor_shape: inferredShape, n, secret: false };
  }

  const params = {};
  const allowed = new Set(["layout_str", "n", "tensor_shape", "secret"]);
  const lines = stripped.split("\n");
  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    // allow python-style docstring literal lines
    if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) continue;
    const m = t.match(/^([a-zA-Z_]\w*)\s*=\s*(.+)$/);
    if (!m) throw new Error("Only simple assignments are allowed.");
    const name = m[1];
    if (!allowed.has(name)) continue;
    const rhs = m[2].trim();
    if (name === "layout_str") {
      const s = parseStringLiteral(rhs);
      if (typeof s !== "string") throw new Error("layout_str must be a quoted string.");
      params.layout_str = s;
    } else if (name === "n") {
      const v = parseInt(rhs, 10);
      if (!(Number.isInteger(v) && v > 0)) throw new Error("n must be a positive integer.");
      params.n = v;
    } else if (name === "secret") {
      if (rhs === "True" || rhs === "true") params.secret = true;
      else if (rhs === "False" || rhs === "false") params.secret = false;
      else throw new Error("secret must be True or False.");
    } else if (name === "tensor_shape") {
      const ts = parseTupleOrListOfInts(rhs);
      params.tensor_shape = ts;
    }
  }
  if (params.layout_str == null) throw new Error('Required: layout_str = "..."');

  if (params.tensor_shape == null) {
    const inferredShape = inferShapeFromLayoutStr(params.layout_str);
    if (inferredShape != null) params.tensor_shape = inferredShape;
  }
  if (params.n == null) {
    const inferredN = inferNFromLayoutStr(params.layout_str);
    if (inferredN != null) params.n = inferredN;
    else if (params.tensor_shape) params.n = prod(params.tensor_shape);
    else params.n = 16;
  }
  if (params.secret == null) params.secret = false;
  return params;
}

function parseStringLiteral(rhs) {
  const m = rhs.match(/^(['"])(.*)\1$/);
  if (!m) return null;
  // Very small escape handling for \n, \t, \", \'
  const quote = m[1];
  let body = m[2];
  body = body.replaceAll("\\n", "\n").replaceAll("\\t", "\t");
  body = body.replaceAll(`\\${quote}`, quote).replaceAll("\\\\", "\\");
  return body;
}

function parseTupleOrListOfInts(rhs) {
  // Accept "(4, 4)" or "[4,4]" or "(4,4,)".
  const t = rhs.trim();
  const isTuple = t.startsWith("(") && t.endsWith(")");
  const isList = t.startsWith("[") && t.endsWith("]");
  if (!isTuple && !isList) throw new Error("tensor_shape must be a list/tuple of positive integers, or omit it.");
  const inner = t.slice(1, -1).trim();
  if (!inner) throw new Error("tensor_shape must be non-empty when provided.");
  const parts = inner
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x.length);
  const out = parts.map((p) => parseInt(p, 10));
  if (!out.every((x) => Number.isInteger(x) && x > 0)) {
    throw new Error("tensor_shape must be a list/tuple of positive integers, or omit it.");
  }
  return out;
}

