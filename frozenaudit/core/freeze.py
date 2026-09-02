"""Step 1 of the protocol: write the decision rules down and hash them.

An audit is only an audit if the criteria predate the numbers. This module
provides the mechanical part of that promise: a deterministic hash over a
canonicalised description of your protocol, recorded before any score is read
and re-checkable afterwards.

Nothing here is chemistry-specific.

Why a hash rather than a git commit: the object being frozen is usually a
dict assembled at runtime from several sources (thresholds pulled from a
model bundle, gates declared in a script, a reference set read from disk).
A commit pins the code that builds it, not the thing built. Two runs of the
same code against a silently updated reference file give the same commit and
different protocols; they give different freeze hashes.

Canonicalisation rules, so that a hash is comparable across machines:

* mappings are emitted with sorted keys
* floats are formatted with :func:`repr`, which round-trips exactly in
  CPython, rather than with ``%.17g`` or similar
* numpy scalars and arrays are converted to plain Python via ``tolist()``
* sets become sorted lists; tuples become lists (JSON has no tuple)
* anything else raises rather than falling back to ``str()``, because a
  fallback would let two different objects hash the same

The last rule is the one that matters. A canonicaliser that quietly calls
``str()`` on an unknown object produces stable-looking hashes that do not
actually pin the object's contents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["canonical", "freeze_hash", "FreezeLock", "FreezeMismatch"]


class FreezeMismatch(Exception):
    """Raised when a protocol does not match the hash it was locked with."""

    def __init__(self, label: str, expected: str, got: str) -> None:
        super().__init__(
            f"{label}: protocol has changed since it was frozen\n"
            f"  locked with: {expected}\n"
            f"  now hashes:  {got}\n"
            "Either you are running a different protocol than the one whose "
            "results you are about to compare against, or the lock is stale. "
            "Do not reconcile numbers across this boundary."
        )
        self.label, self.expected, self.got = label, expected, got


def _numpy_scalar(value: Any) -> Any:
    """Convert a numpy scalar/array to plain Python, or return None if it is not one.

    numpy is imported lazily so that this module works in an environment that
    does not have it; the audit core should not require an array library just
    to hash a dict of thresholds.
    """
    item = getattr(value, "item", None)
    tolist = getattr(value, "tolist", None)
    if tolist is not None and getattr(value, "shape", None) is not None:
        return tolist()
    if item is not None and getattr(value, "dtype", None) is not None:
        return item()
    return None


def canonical(obj: Any) -> Any:
    """Reduce ``obj`` to JSON-safe primitives under the rules in the module docstring.

    Raises
    ------
    TypeError
        If a value has no defined canonical form. This is deliberate: see the
        module docstring on why a ``str()`` fallback would be unsound.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # repr round-trips exactly in CPython; format() and %g do not.
        return {"__float__": repr(obj)}
    if isinstance(obj, dict):
        out = {}
        for key in sorted(obj, key=lambda k: (type(k).__name__, str(k))):
            if not isinstance(key, (str, int, bool)) and key is not None:
                raise TypeError(
                    f"freeze: mapping key {key!r} of type {type(key).__name__} "
                    "has no canonical form; convert keys to str first"
                )
            out[str(key)] = canonical(obj[key])
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [canonical(v) for v in sorted(obj, key=lambda v: (type(v).__name__, str(v)))]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_sha256__": hashlib.sha256(bytes(obj)).hexdigest()}
    converted = _numpy_scalar(obj)
    if converted is not None:
        return canonical(converted)
    raise TypeError(
        f"freeze: object of type {type(obj).__name__} has no canonical form. "
        "Convert it to a dict/list/scalar describing exactly what you mean to "
        "freeze -- a fallback to str() would let two different objects hash "
        "the same."
    )


def freeze_hash(obj: Any) -> str:
    """SHA-256 over the canonical form of ``obj``.

    >>> freeze_hash({"threshold": 0.70}) == freeze_hash({"threshold": 0.70})
    True
    >>> freeze_hash({"threshold": 0.70}) == freeze_hash({"threshold": 0.7000001})
    False
    """
    payload = json.dumps(canonical(obj), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass
class FreezeLock:
    """A recorded hash of a protocol, plus the label it goes by in a report.

    Typical use, before any scoring::

        lock = FreezeLock.record("my_protocol", {"gates": [...], "cuts": {...}})
        print(lock.sha256)          # paste into the methods section

    and afterwards, before comparing anything to published numbers::

        lock.verify(protocol_dict)  # raises FreezeMismatch if it drifted

    ``short`` is what a paper usually carries: the first 16 hex characters,
    which is what the accompanying manuscript reports for its two locks.
    """

    label: str
    sha256: str
    n_keys: int = 0
    note: str = ""
    _canonical_preview: dict = field(default_factory=dict, repr=False)

    @classmethod
    def record(cls, label: str, obj: Any, note: str = "") -> "FreezeLock":
        canon = canonical(obj)
        return cls(label=label, sha256=freeze_hash(obj),
                   n_keys=len(canon) if isinstance(canon, dict) else 0,
                   note=note,
                   _canonical_preview={k: type(v).__name__
                                       for k, v in list(canon.items())[:20]}
                   if isinstance(canon, dict) else {})

    @property
    def short(self) -> str:
        """First 16 hex characters, the form a methods section usually quotes."""
        return self.sha256[:16]

    def verify(self, obj: Any) -> bool:
        """Raise :class:`FreezeMismatch` unless ``obj`` still hashes to this lock."""
        got = freeze_hash(obj)
        if got != self.sha256:
            raise FreezeMismatch(self.label, self.sha256, got)
        return True

    def matches(self, obj: Any) -> bool:
        """Non-raising form of :meth:`verify`."""
        return freeze_hash(obj) == self.sha256

    def as_dict(self) -> dict:
        return {"label": self.label, "sha256": self.sha256, "short": self.short,
                "n_keys": self.n_keys, "note": self.note}
