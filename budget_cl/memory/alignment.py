"""Record sizes under packed and aligned ABIs.

The paper reports PACKED sizes (203 B raw, 21 B post-pool) and states the
alignment assumption explicitly, because a packed-state claim that silently
depends on a favourable ABI is not reproducible on another toolchain.

Both replay structs are all int8/uint8, so packed size, natural ABI size, and
1-byte stride coincide. Under wider alignment they do not, which is why the
padded strides are reported rather than assumed away.
"""

from __future__ import annotations

__all__ = ["align_to", "alignment_report"]


def align_to(n: int, alignment: int) -> int:
    """Round ``n`` up to a multiple of ``alignment``.

    Raises:
        ValueError: if ``alignment`` is not positive.
    """
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    if alignment == 1:
        return int(n)
    return int(((n + alignment - 1) // alignment) * alignment)


def alignment_report(record_bytes: int) -> dict[str, int]:
    """Stride of one record under each ABI assumption the paper discusses."""
    return {
        "packed": int(record_bytes),
        "aligned_4": align_to(record_bytes, 4),
        "aligned_8": align_to(record_bytes, 8),
    }
