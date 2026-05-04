"""MBQC state vector backend simulator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.sim.statevec import Statevec as SVLegacy
from graphix.states import BasicStates

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.data import Data


class Statevec(DenseState):
    """Statevector object.

    Attributes
    ----------
    psi : numpy.ndarray of numpy.complex128
        Complex-valued 1-dimensional array representing the quantum statevector.
        Throughout the simulation ``psi`` has constant size ``2**max_space``. Only the first ``2**nqubit`` complex values have meaning.

    max_space : int
        Maximum Hilbert space size allowed for internal computations. It determines the size of ``psi``. For circuit simulations, it corresponds to the number of qubits, while for pattern simulations it corresponds to the pattern's maximum space.

    _nqubit : int
        Number of active qubits at any given time.
    """

    psi: Matrix  # TODO: Update type annotation appropiately.

    def __init__(self, data: Data = BasicStates.PLUS, nqubit: int | None = None) -> None:
        """Initialize statevector objects.

        See :class:`graphix.sim.statevec.Statevec` for additional information.

        Parameters
        ----------
        data : Data, optional
            Input data to prepare the state. Can be a classical description or a numerical input, defaults to `graphix.states.BasicStates.PLUS`
        nqubit : int | None, optional
            Number of qubits to prepare. If ``None`` (default), it's inferred from ``data``.
        """
        sv_graphix = SVLegacy(data, nqubit)  # noqa: F841

        # TODO
        # For simplicity, the __init__ method in this template re-uses the constructor of the existing statevector in Graphix.
        # This method should convert `sv_graphix` to the appropriate internal representation of the new backend.

    def __str__(self) -> str:
        """Return a string description."""
        sv = self.psi
        return f"Statevec object with statevector {sv} and length {len(sv)}."

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits."""
        # TODO
        raise NotImplementedError

    @override
    def flatten(self) -> Matrix:
        """Return flattened state.

        A view of only the first ``2**self.nqubit`` elements of ``self.psi`` is returned.
        """
        # TODO
        raise NotImplementedError

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add nodes (qubits) to the state vector and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`, initializes the new nodes jointly.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        sv_to_add = Statevec(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """Connect graph nodes.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices
        """
        # TODO
        raise NotImplementedError

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        """Apply a multi-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n matrix
        qargs : list of int
            target qubits' indices
        """
        # This method is not required for the pattern simulator,
        # only by the circuit simulator.
        # It cannot be commented out because it's an abstract method
        # of DenseState.
        raise NotImplementedError

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit operation.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 matrix
        i : int
            qubit index
        """
        # TODO
        raise NotImplementedError

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Return the expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        loc : int
            target qubit index

        Returns
        -------
        complex : expectation value.
        """
        # TODO
        raise NotImplementedError

    @override
    def remove_qubit(self, qarg: int) -> None:
        r"""Remove a separable qubit from the system and assemble a statevector for remaining qubits.

        This results in the same result as partial trace, if the qubit *qarg* is separable from the rest.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with sum taken over
        :math:`i \in [ 0 \dots 00,\ 0\dots 01,\ \dots,\
        1 \dots 11 ]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
           \end{align}

        (after normalization) for :math:`k =` qarg. If the :math:`k` th qubit is in :math:`\ket{1}` state,
        above will return zero amplitudes; in such a case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}` .

        .. warning::
            This method assumes the qubit with index *qarg* to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        Parameters
        ----------
        qarg : int
            qubit index
        """
        # TODO
        raise NotImplementedError

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap qubits.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # TODO
        raise NotImplementedError

    def tensor(self, other: Statevec) -> None:
        r"""Tensor product state with other qubits.

        Results in ``self`` :math:`\otimes` ``other``.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevec`
            Statevector to be tensored with ``self``.
        """
        # TODO
        raise NotImplementedError


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC state vector backend simulator."""

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))
