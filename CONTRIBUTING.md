# Contributing

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/smeinecke/image-match.git
cd image-match
uv sync --all-extras
```

## Validate

```bash
make validate   # ruff format+lint, radon/xenon, bandit, pyright, vulture
make test       # unit tests
make test-cov   # unit tests with coverage (must stay above 80%)
make docs       # build Sphinx docs (warnings are errors)
```

## Integration tests

Integration tests run against live Elasticsearch, OpenSearch and MongoDB in Docker:

```bash
make test-integration-local   # starts services, runs tests, stops services
```

This includes the Toxiproxy fault-injection tier (`tests/test_faults_integration.py`),
which exercises real transport failures (proxy kill, latency, connection resets)
between the drivers and OpenSearch. `make db-up` starts the toxiproxy service
alongside the databases.

## Mutation testing

`mutmut` runs the non-integration suite against generated mutants. It is
deliberately not part of PR validation (a full run takes a while) — it runs
weekly and on demand via the `mutation` workflow, or locally with:

```bash
make mutation
```

Lines that can never change behaviour (CLI help text, `print` reporting, the
vestigial `handle_mpo` parameter, `typing.cast`) are excluded via
`do_not_mutate_patterns` in `pyproject.toml`.

## Pull requests

- Keep changes in logically grouped commits.
- All checks (`make validate`, unit tests, integration tests, docs build)
  run in CI on every PR and must pass.
- New public API needs unit tests and an integration test where possible.
- Preserve compatibility with existing indexed data: signatures, word
  encodings and stored record formats must not change silently.

## Releases

Releases are cut from tags. The tag must match `__version__` in
`src/image_match/__init__.py`:

```bash
git tag v2.0.1 && git push origin v2.0.1
```

The release workflow builds the wheel/sdist, attests build provenance,
publishes to PyPI via OIDC trusted publishing, and creates a GitHub
Release with the artifacts attached.
