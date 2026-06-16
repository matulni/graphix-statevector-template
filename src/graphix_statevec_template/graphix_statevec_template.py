"""MBQC state vector backend simulator."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import numpy as np
import numpy.typing as npt
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.sim.statevec import Statevec as SVLegacy
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.data import Data

try:
    import cupy as _cupy
    _cupy.cuda.Device(0).compute_capability  # type: ignore[attr-defined]  # noqa: B018
    cp: Any = _cupy
    _GPU: bool = True
except Exception:  # noqa: BLE001
    cp = np
    _GPU = False


def _gpu_available() -> bool:
    """Return True if CuPy and a CUDA GPU are available.

    Returns
    -------
    bool
        True when the GPU backend is active.
    """
    return _GPU


# Module-level gate constants (numpy; converted to cp at use-time via cp.asarray)
_CZ = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    dtype=np.complex128,
)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)


class Statevec(DenseState):
    """GPU-accelerated statevector with a flat pre-allocated buffer.

    The quantum state is stored as a flat complex128 array of length
    ``2**max_space``. Only the first ``2**_nqubit`` elements are active;
    the rest is zero-padded capacity reserve so that :meth:`tensor` and
    :meth:`add_nodes` avoid reallocation on every call.

    All gate operations use :func:`cp.tensordot` on the reshaped active region,
    preserving graphix's qubit convention (qubit ``i`` = axis ``i``).

    Attributes
    ----------
    psi : array-like of complex128
        Flat buffer of length ``2**max_space``.
        Active region: ``psi[:2**_nqubit]``.
    max_space : int
        Allocated capacity in qubits. ``len(psi) == 2**max_space``.
    _nqubit : int
        Number of active qubits.
    """

    psi: Any  # cp.ndarray when GPU available, np.ndarray otherwise

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
        max_space: int | None = None,
    ) -> None:
        """Initialise statevector.

        Delegates state construction to :class:`graphix.sim.statevec.Statevec`
        then copies the result into a GPU (or CPU fallback) flat buffer.

        Parameters
        ----------
        data : Data, optional
            Input state. Accepts a :class:`graphix.states.State`, a list of
            states, or a flat complex128 array. Defaults to
            ``BasicStates.PLUS``.
        nqubit : int | None, optional
            Number of qubits. Inferred from ``data`` when ``None``.
        max_space : int | None, optional
            Pre-allocated buffer capacity in qubits. Defaults to ``nqubit``.
        """
        sv_legacy = SVLegacy(data, nqubit)
        self._nqubit: int = sv_legacy.nqubit
        flat = sv_legacy.flatten().astype(np.complex128)

        self.max_space: int = max(max_space if max_space is not None else self._nqubit, self._nqubit)

        buf = cp.zeros(1 << self.max_space, dtype=cp.complex128)
        buf[: len(flat)] = cp.asarray(flat)
        self.psi = buf

    def __str__(self) -> str:
        """Return a string description."""
        active = self.psi[: 1 << self._nqubit]
        return f"Statevec object with statevector {active} and length {1 << self._nqubit}."

    # ── private helpers ───────────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        """Reallocate buffer when ``needed`` qubits exceed current capacity.

        Parameters
        ----------
        needed : int
            Required number of qubits after the next operation.
        """
        if needed <= self.max_space:
            return
        new_max = max(self.max_space + 1, needed)
        new_buf = cp.zeros(1 << new_max, dtype=cp.complex128)
        new_buf[: 1 << self._nqubit] = self.psi[: 1 << self._nqubit]
        self.psi = new_buf
        self.max_space = new_max

    def _apply_gate(
        self,
        gate_np: npt.NDArray[np.complex128],
        qargs: tuple[int, ...],
    ) -> None:
        """Apply a gate to the specified qubits via tensordot.

        Parameters
        ----------
        gate_np : np.ndarray
            Gate matrix of shape ``(2**k, 2**k)`` for a k-qubit gate.
        qargs : tuple of int
            Target qubit indices, matching graphix axis ordering.
        """
        n = self._nqubit
        k = len(qargs)
        gate = cp.asarray(gate_np).reshape([2] * (2 * k))
        psi_t = self.psi[: 1 << n].reshape([2] * n)

        contracted = cp.tensordot(
            gate,
            psi_t,
            axes=(list(range(k, 2 * k)), list(qargs)),
        )
        self.psi[: 1 << n] = cp.moveaxis(contracted, list(range(k)), list(qargs)).ravel()

    # ── DenseState interface ──────────────────────────────────────────────

    @property
    @override
    def nqubit(self) -> int:
        """Return the number of active qubits."""
        return self._nqubit

    @override
    def flatten(self) -> Matrix:
        """Return the active state as a flat numpy array.

        Returns
        -------
        np.ndarray
            Complex128 array of length ``2**nqubit``.
        """
        active = self.psi[: 1 << self._nqubit]
        if _GPU:
            return np.asarray(active.get(), dtype=np.complex128)
        return np.asarray(active, dtype=np.complex128)

    def tensor(self, other: Statevec) -> None:
        r"""In-place tensor product ``self ⊗ other``.

        Parameters
        ----------
        other : Statevec
            Statevector to tensor with.
        """
        new_n = self._nqubit + other._nqubit
        self._ensure_capacity(new_n)
        result = cp.kron(
            self.psi[: 1 << self._nqubit],
            other.psi[: 1 << other._nqubit],
        )
        self.psi[: 1 << new_n] = result
        self._nqubit = new_n

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add qubits initialised in ``data`` via tensor product.

        Parameters
        ----------
        nqubit : int
            Number of qubits to add.
        data : Data
            Initial state for the new qubits.
        """
        self.tensor(Statevec(nqubit=nqubit, data=data))

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """Apply a CZ gate between two qubits.

        Parameters
        ----------
        edge : tuple of int
            ``(control, target)`` qubit indices.
        """
        self._apply_gate(_CZ, edge)

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit gate.

        Parameters
        ----------
        op : np.ndarray
            2x2 unitary matrix.
        i : int
            Target qubit index.
        """
        self._apply_gate(np.asarray(op, dtype=np.complex128), (i,))

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Apply a multi-qubit gate.

        Parameters
        ----------
        op : np.ndarray
            ``2**k X 2**k`` matrix for a k-qubit gate.
        qargs : sequence of int
            Target qubit indices.
        """
        self._apply_gate(np.asarray(op, dtype=np.complex128), tuple(qargs))

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        r"""Return the expectation value of a single-qubit observable.

        Computes :math:`\langle\psi|O|\psi\rangle`.

        Parameters
        ----------
        op : np.ndarray
            2X2 Hermitian operator.
        loc : int
            Target qubit index.

        Returns
        -------
        complex
            Expectation value.
        """
        n = self._nqubit
        psi_flat = self.psi[: 1 << n]
        op_gpu = cp.asarray(np.asarray(op, dtype=np.complex128))

        psi_op = cp.tensordot(op_gpu, psi_flat.reshape([2] * n), axes=([1], [loc]))
        psi_op = cp.moveaxis(psi_op, 0, loc).ravel()

        result = cp.dot(psi_flat.conj(), psi_op)
        if _GPU:
            return complex(result.item())
        return complex(result)

    @override
    def remove_qubit(self, qarg: int) -> None:
        r"""Remove a separable qubit after measurement.

        Scans the |0⟩ then |1⟩ branch of qubit ``qarg`` and keeps the
        non-zero-norm one, normalising the result.

        Parameters
        ----------
        qarg : int
            Qubit index to remove.

        Raises
        ------
        ValueError
            If both branches have zero norm — qubit is not separable.
        """
        n = self._nqubit
        psi_t = self.psi[: 1 << n].reshape([2] * n)
        idx: list[int | slice] = [slice(None)] * n

        for val in (0, 1):
            idx[qarg] = val
            branch = psi_t[tuple(idx)].ravel()
            nrm2 = float(cp.sum(cp.abs(branch) ** 2))
            if not math.isclose(nrm2, 0.0, abs_tol=1e-15):
                self._nqubit -= 1
                self.psi[: 1 << self._nqubit] = branch / math.sqrt(nrm2)
                self.psi[1 << self._nqubit :] = 0.0
                return

        msg = f"Both branches of qubit {qarg} have zero norm — qubit may not be separable."
        raise ValueError(msg)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap two qubits.

        Parameters
        ----------
        qubits : tuple of int
            ``(qubit_a, qubit_b)`` indices.
        """
        self._apply_gate(_SWAP, qubits)


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC state vector backend using GPU acceleration via CuPy."""

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))
