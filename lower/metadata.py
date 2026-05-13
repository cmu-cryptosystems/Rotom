"""Small helpers for preserving source scopes in serialized HE instructions."""


def kernel_metadata(kernel, default_op: str) -> str:
    term = getattr(kernel.layout, "term", None)
    scope = getattr(term, "orbit_scope", None)
    if not scope:
        return ""
    op_name = getattr(term, "orbit_op", default_op)
    layer = getattr(term, "orbit_layer", None)
    parts = [f"scope={scope}", f"op={op_name}"]
    if layer is not None:
        parts.append(f"layer={layer}")
    return ";".join(parts)


def pack_metadata(kernel, index: int) -> str:
    metadata = kernel_metadata(kernel, "pack")
    if metadata:
        return f"{index} {metadata}"
    return f"{index} {kernel}"
