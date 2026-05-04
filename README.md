# graphix-statevec-template

Template for efficient Graphix statevector backend simulators.

## Editable installation using uv

This template supports [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/matulni/graphix-statevec-template.git
cd graphix-statevec-template
uv sync --extra dev
```

This creates a virtual environment and installs all development and extra dependencies from the `pyproject.toml` and `uv` lockfile.

## Development

To develop a minimal working statevector bakend for Graphix' pattern simulator, all methods of `Statevec` marked with the `# TODO` label should be implemented. These include:

- `__init__`
- `entangle`
- `evolve_single`
- `expectation_single`
- `remove_qubit`
- `swap`
- `tensor`
- `flatten`
and the property `nqubit`.

Additionally, an appropriate internal representation for the statevector (`psi` attribute) must be chosen.

The method `evolve` (for multi-qubit operations) is not required for the pattern simulation.

## Testing and benchmarking

This template comes with a minimal set of unit tests and benchmarks: 

```bash
cd graphix-statevec-template
uv pytest tests/
uv pytest benchmarks/
```

All tests and benchmarks are decorated with `@pytest.mark.skip(reason="Not Implemented")` which should be removed once the corresponding methods have been implemented.

Note that `graphix-statevec-template/benchmarks/bench_mqtbench.py` relies on the `graphix-mqtbench` plugin which is in an early development stage.

## Usage

Custom statevector backends based on this templace can be passed as any other backend to the Graphix pattern simulator:

```python
from graphix_statevec_template import Statevec, StatevectorBackend
from graphix.transpiler import Circuit

qc = Circuit(2)
qc.cz(0, 1)
qc.h(0)

pattern = qc.transpile().pattern
backend = StatevectorBackend()
sv = pattern.simulate_pattern(backend=backend)
```