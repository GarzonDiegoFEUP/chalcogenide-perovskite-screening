Generating the docs
----------

Uses [MkDocs](http://www.mkdocs.org/) with [Material theme](https://squidfunk.github.io/mkdocs-material/).

> **Note:** Docs dependencies require Python >= 3.9. They are installed separately from the project's Python 3.11 environment.

### Quick start (from repo root)

    make docs

### Manual setup

Install dependencies (use the project's Python 3.11 environment or any Python >= 3.9):

    pip install -r docs/requirements.txt

Build:

    mkdocs build

Serve locally:

    mkdocs serve
