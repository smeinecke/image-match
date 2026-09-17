[![CI](https://github.com/smeinecke/image-match/actions/workflows/check.yml/badge.svg)](https://github.com/smeinecke/image-match/actions/workflows/check.yml)
[![PyPI](https://img.shields.io/pypi/v/image-match.svg)](https://pypi.python.org/pypi/image-match)
[![Release](https://img.shields.io/github/v/release/smeinecke/image-match)](https://github.com/smeinecke/image-match/releases)
[![Documentation Status](https://github.com/smeinecke/image-match/actions/workflows/docs.yml/badge.svg)](https://smeinecke.github.io/image-match/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.13-blue)](https://pypi.org/project/image-match/)

![image-match](https://cloud.githubusercontent.com/assets/6517700/17741093/41040a64-649b-11e6-8499-48b78ddca56b.png)

**image-match** finds approximate (near-duplicate) image matches in a corpus.
It generates a compact 648-dimensional perceptual signature per image
([Goldberg et al.](http://www.cs.cmu.edu/~hcwong/Pdfs/icip02.ps)) and stores it
in a database backend that scales to billions of images — sustaining insertion
rates of up to 10,000 images/s.

> **Note:** this algorithm detects *near-duplicates* — think copyright
> violation detection. It is **not** meant for conceptually similar images
> (see [this issue](https://github.com/edjo-labs/image-match/issues/62) and
> [this video](https://www.youtube.com/watch?v=DfWLBzArzKE)). For a different
> approach, see the [libpuzzle reference implementation](https://github.com/jedisct1/libpuzzle)
> or Pavlov's [containerized fork](https://github.com/pavlovml/match).

## Features

- **Multiple backends**: Elasticsearch, OpenSearch 2.x/3.x, MongoDB
- **Sync + async drivers** for Elasticsearch and OpenSearch
- **OpenSearch k-NN**: approximate nearest-neighbour search on the signature
  vector, plus a migration tool for existing word indexes
- **Orientation matching**: optionally match rotations, mirrors and color
  inversions of the query image
- **Metadata filtering**: store arbitrary metadata and pre-filter searches
- **Signature-only mode**: generate and compare signatures without a database

## Install

```bash
pip install "image-match[opensearch]"   # or [elasticsearch], [mongo]
```

Backend client libraries are optional extras — install only what you use
(`elasticsearch-async`, `opensearch-async` for the async drivers). Signature
generation alone needs no extra.

## Quick start

```python
from opensearchpy import OpenSearch
from image_match.opensearch_driver import SignatureOpenSearch

ses = SignatureOpenSearch(OpenSearch("http://localhost:9201"))

ses.add_image("photo_a.jpg")
ses.add_image("https://example.org/photo_b.jpg")

hits = ses.search_image("photo_a.jpg")
# [{'id': ..., 'path': 'photo_a.jpg', 'dist': 0.0, ...}, ...]
```

`dist` is the normalized signature distance: `0.0` is identical, `< 0.40` is
very likely the same image. A `docker-compose.yml` with Elasticsearch,
OpenSearch (2.x and 3.x) and MongoDB is included for local development.

## Documentation

Full docs at [smeinecke.github.io/image-match](https://smeinecke.github.io/image-match/):

- [Getting started](https://smeinecke.github.io/image-match/start.html) /
  [Quickstart](https://smeinecke.github.io/image-match/quickstart.html)
- [Image signatures and distances](https://smeinecke.github.io/image-match/signatures.html) —
  how the algorithm works
- [Storing and searching](https://smeinecke.github.io/image-match/searches.html)
- [Backends](https://smeinecke.github.io/image-match/backends.html) —
  including writing your own driver
- [Migrating to OpenSearch k-NN](https://smeinecke.github.io/image-match/migration.html)
- [API reference](https://smeinecke.github.io/image-match/api.html)

## Development

Uses [uv](https://docs.astral.sh/uv/) for dependency management and Ruff for
linting. Tests run in tiers — unit, live integration (Docker Compose, incl.
Toxiproxy fault injection) and mutation testing (`mutmut`, weekly CI):

```bash
uv sync --all-extras        # install everything
make validate               # format, lint, types, security, complexity
make test                   # unit tests
make test-integration-local # live backends via docker compose
make mutation               # mutation testing (slow)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache-2.0 — see [LICENSES.md](LICENSES.md).
