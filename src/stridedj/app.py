"""Composition root: builds every adapter from `Settings` and wires them into the harness (dependency injection).

The only module that knows which concrete adapters exist. Tests, evals and `--demo` build the same graph with fakes."""
