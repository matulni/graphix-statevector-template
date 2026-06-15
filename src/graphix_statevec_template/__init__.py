"""Template for efficient Graphix statevector backend."""

from __future__ import annotations

from graphix_statevec_template.graphix_statevec_template import (
    Statevec,
    StatevectorBackend,
    _gpu_available,
)

__all__ = ["Statevec", "StatevectorBackend", "_gpu_available"]
