"""GPU-accelerated MBQC state vector backend built on NVIDIA cuQuantum (cuStateVec).

The quantum state is stored as a flat :mod:`cupy` array of constant size ``2**max_space`` and
``complex128`` dtype. Only the first ``2**nqubit`` entries are meaningful; the remaining entries
are padding that lets the active register grow and shrink (as ``N`` and ``M`` commands are
processed) without reallocating GPU memory.

Graphix uses the most-significant-bit convention (qubit ``0`` is axis ``0``), whereas cuStateVec
uses the least-significant-bit convention. Indices are converted with :func:`_msb_to_lsb` before
every cuStateVec call.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TypedDict, Unpack, override

import cupy as cp
import cuquantum.bindings.custatevec as cusv
import numpy as np
from cuquantum import cudaDataType
from cuquantum.bindings.custatevec import ComputeType, MatrixLayout
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix, NodeIndex
from graphix.sim.statevec import Statevec as SVLegacy
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.branch_selector import BranchSelector
    from graphix.sim.data import Data

_C64 = cudaDataType.CUDA_C_64F
_COMPUTE = ComputeType.COMPUTE_64F
_ROW = MatrixLayout.ROW
_INV_SQRT2 = 1 / math.sqrt(2)

# Host-side gate constants. The device copies are built lazily (see `_cz_device`, `_plus_device`)
# so that importing this module never touches the GPU (e.g. on CI runners without a device).
_CZ_HOST = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    dtype=np.complex128,
)
_PLUS_HOST = np.array([_INV_SQRT2, _INV_SQRT2], dtype=np.complex128)


def _gpu_available() -> bool:
    """Return ``True`` if a CUDA device is available, ``False`` otherwise."""
    try:
        return bool(cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:  # noqa: BLE001  any CUDA error means no usable device
        return False


@functools.cache
def _handle() -> int:
    """Return a process-wide cuStateVec handle, created lazily on first use."""
    handle: int = cusv.create()
    return handle


@functools.cache
def _cz_device() -> Any:
    """Return the CZ gate as a device matrix, allocated once."""
    return cp.asarray(_CZ_HOST)


@functools.cache
def _plus_device() -> Any:
    """Return the single-qubit ``|+>`` state as a device vector, allocated once."""
    return cp.asarray(_PLUS_HOST)


def _msb_to_lsb(targets: Sequence[int], nqubit: int) -> tuple[int, ...]:
    """Convert Graphix MSB qubit indices to cuStateVec LSB index bits.

    Parameters
    ----------
    targets : Sequence[int]
        Qubit indices in the Graphix (MSB) convention.
    nqubit : int
        Number of active qubits.

    Returns
    -------
    tuple[int, ...]
        Corresponding index bits in the cuStateVec (LSB) convention.
    """
    return tuple(nqubit - 1 - t for t in targets)


class Statevec(DenseState):
    """State vector stored on the GPU and manipulated through cuStateVec.

    Attributes
    ----------
    psi : cupy.ndarray of cupy.complex128
        Flat device array of constant size ``2**max_space``. Only the first ``2**nqubit`` entries
        are meaningful; the rest is padding.
    max_space : int
        Maximum Hilbert-space exponent supported without reallocation. It determines the size of
        ``psi``. For pattern simulations it corresponds to :meth:`graphix.pattern.Pattern.max_space`.
    """

    psi: Any  # cupy.ndarray; cupy is untyped, so it is treated as ``Any``.
    max_space: int
    _nqubit: int

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
        max_space: int | None = None,
    ) -> None:
        """Initialize the state vector.

        The input ``data`` is first parsed by :class:`graphix.sim.statevec.Statevec` (reusing all
        of its input validation), and the resulting amplitudes are copied to the GPU.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. See :class:`graphix.sim.statevec.Statevec`.
        nqubit : int | None, optional
            Number of qubits to prepare. If ``None`` (default), it is inferred from ``data``.
        max_space : int | None, optional
            Capacity of the internal buffer, as an exponent of two. If ``None`` (default), it
            equals the number of qubits. Must be at least the number of qubits.
        """
        sv_graphix = SVLegacy(data, nqubit)
        host = np.asarray(sv_graphix.flatten(), dtype=np.complex128)
        n = sv_graphix.nqubit

        if max_space is None:
            max_space = n
        elif max_space < n:
            raise ValueError(f"max_space ({max_space}) must be at least the number of qubits ({n}).")

        self.max_space = max_space
        self._nqubit = n
        self.psi = cp.zeros(1 << max_space, dtype=cp.complex128)
        self.psi[: 1 << n] = cp.asarray(host)

    def __str__(self) -> str:
        """Return a string description."""
        sv = cp.asnumpy(self._active)
        return f"Statevec object with statevector {sv} and length {len(sv)}."

    @property
    def _active(self) -> Any:
        """Return the meaningful slice ``psi[: 2**nqubit]`` (a device view)."""
        return self.psi[: 1 << self._nqubit]

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of active qubits."""
        return self._nqubit

    @override
    def flatten(self) -> Matrix:
        """Return the active state as a host (NumPy) array of length ``2**nqubit``."""
        result: Matrix = cp.asnumpy(self._active)
        return result

    def _ensure_capacity(self, nqubit: int) -> None:
        """Grow ``psi`` in place if it cannot hold ``2**nqubit`` amplitudes."""
        needed = 1 << nqubit
        if needed > self.psi.size:
            grown = cp.zeros(needed, dtype=cp.complex128)
            grown[: self.psi.size] = self.psi
            self.psi = grown
            self.max_space = nqubit

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add qubits to the state vector and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            Number of qubits to add.
        data : Data
            State in which to initialize the new qubits. See :meth:`graphix.sim.statevec.Statevec.add_nodes`.

        Notes
        -----
        The common case of adding a single ``|+>`` qubit (when processing an ``N`` command) is
        handled with a specialized device-only path that avoids any host-to-device transfer.
        """
        if nqubit == 1 and data is BasicStates.PLUS:
            new_n = self._nqubit + 1
            combined = cp.kron(self._active, _plus_device())
            self._ensure_capacity(new_n)
            self.psi[: 1 << new_n] = combined
            self._nqubit = new_n
        else:
            self.tensor(Statevec(nqubit=nqubit, data=data))

    def _apply_matrix(self, matrix: Any, targets: Sequence[int]) -> None:
        """Apply a device matrix to ``targets`` (Graphix MSB indices) via cuStateVec."""
        n = self._nqubit
        bits = list(_msb_to_lsb(targets, n))
        n_targets = len(bits)
        handle = _handle()
        ws_size = cusv.apply_matrix_get_workspace_size(
            handle=handle,
            sv_data_type=_C64,
            n_index_bits=n,
            matrix=matrix.data.ptr,
            matrix_data_type=_C64,
            layout=_ROW,
            adjoint=0,
            n_targets=n_targets,
            n_controls=0,
            compute_type=_COMPUTE,
        )
        ws = cp.zeros(ws_size, dtype=cp.uint8)
        ws_ptr = ws.data.ptr if ws_size > 0 else 0
        cusv.apply_matrix(
            handle=handle,
            sv=self.psi.data.ptr,
            sv_data_type=_C64,
            n_index_bits=n,
            matrix=matrix.data.ptr,
            matrix_data_type=_C64,
            layout=_ROW,
            adjoint=0,
            targets=bits,
            n_targets=n_targets,
            controls=[],
            control_bit_values=[],
            n_controls=0,
            compute_type=_COMPUTE,
            extra_workspace=ws_ptr,
            extra_workspace_size_in_bytes=ws_size,
        )

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit operation to qubit ``i``."""
        self._apply_matrix(cp.asarray(op, dtype=cp.complex128), (i,))

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Apply a multi-qubit operation to ``qargs``.

        This method is not required by the pattern simulator (only by the circuit simulator). It
        is left unimplemented because it cannot be commented out (it is an abstract method of
        :class:`graphix.sim.base_backend.DenseState`).
        """
        raise NotImplementedError

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """Apply a CZ gate to the two qubits of ``edge``."""
        self._apply_matrix(_cz_device(), edge)

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Return the expectation value of a single-qubit operator on the normalized state."""
        n = self._nqubit
        matrix = cp.asarray(op, dtype=cp.complex128)
        bits = list(_msb_to_lsb((loc,), n))
        handle = _handle()
        ws_size = cusv.compute_expectation_get_workspace_size(
            handle=handle,
            sv_data_type=_C64,
            n_index_bits=n,
            matrix=matrix.data.ptr,
            matrix_data_type=_C64,
            layout=_ROW,
            n_basis_bits=1,
            compute_type=_COMPUTE,
        )
        ws = cp.zeros(ws_size, dtype=cp.uint8)
        ws_ptr = ws.data.ptr if ws_size > 0 else 0
        expectation = np.empty(1, dtype=np.complex128)
        cusv.compute_expectation(
            handle=handle,
            sv=self.psi.data.ptr,
            sv_data_type=_C64,
            n_index_bits=n,
            expectation_value=expectation.ctypes.data,
            expectation_data_type=_C64,
            matrix=matrix.data.ptr,
            matrix_data_type=_C64,
            layout=_ROW,
            basis_bits=bits,
            n_basis_bits=1,
            compute_type=_COMPUTE,
            extra_workspace=ws_ptr,
            extra_workspace_size_in_bytes=ws_size,
        )
        # `compute_expectation` returns <psi|op|psi> for the unnormalized state, so divide by the
        # squared norm to match the normalized convention of the reference backend.
        norm2 = float(cp.sum(cp.abs(self._active) ** 2))
        return complex(expectation[0]) / norm2

    @override
    def remove_qubit(self, qarg: int) -> None:
        """Remove a separable qubit from the system (see :meth:`graphix.sim.statevec.Statevec.remove_qubit`)."""
        n = self._nqubit
        tensor = self._active.reshape((2,) * n)
        for value in (0, 1):
            index = (slice(None),) * qarg + (value,)
            branch = tensor[index].reshape(-1)
            norm2 = float(cp.sum(cp.abs(branch) ** 2))
            if not math.isclose(norm2, 0, abs_tol=1e-15):
                break
        else:
            raise ValueError(f"Both branches of qubit {qarg} have zero norm; the qubit may not be separable.")
        new_n = n - 1
        self.psi[: 1 << new_n] = branch / math.sqrt(norm2)
        self._nqubit = new_n

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap two qubits."""
        first, second = qubits
        if first == second:
            return
        n = self._nqubit
        tensor = self._active.reshape((2,) * n)
        self.psi[: 1 << n] = cp.swapaxes(tensor, first, second).reshape(-1)

    def tensor(self, other: Statevec) -> None:
        r"""Tensor the state with ``other`` (``self`` :math:`\otimes` ``other``)."""
        new_n = self._nqubit + other._nqubit
        combined = cp.kron(self._active, other._active)
        self._ensure_capacity(new_n)
        self.psi[: 1 << new_n] = combined
        self._nqubit = new_n


class DenseStateBackendKwargs(TypedDict, total=False):
    """Keyword arguments accepted by :class:`graphix.sim.base_backend.DenseStateBackend`."""

    node_index: NodeIndex
    branch_selector: BranchSelector
    symbolic: bool


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC state vector backend running on the GPU through cuStateVec."""

    state: Statevec = dataclasses.field(default_factory=lambda: Statevec(nqubit=0))

    @classmethod
    def with_capacity(
        cls, max_qubits: int, state: Statevec | None = None, **kwargs: Unpack[DenseStateBackendKwargs]
    ) -> Self:
        """Initialize the backend with a preallocated state-vector capacity.

        Parameters
        ----------
        max_qubits : int
            Maximum number of qubits supported without reallocation. For pattern simulation this
            corresponds to :meth:`graphix.pattern.Pattern.max_space`.
        state : Statevec | None, optional
            Initial backend state. If ``None`` (default), a 0-qubit state is used.
        **kwargs
            Options for :class:`graphix.sim.base_backend.DenseStateBackend`.

        Returns
        -------
        Self
            Backend instance with capacity for up to ``max_qubits`` qubits.
        """
        state_init = (
            Statevec(nqubit=0, max_space=max_qubits)
            if state is None
            else Statevec(state.flatten(), max_space=max_qubits)
        )
        return cls(state_init, **kwargs)
