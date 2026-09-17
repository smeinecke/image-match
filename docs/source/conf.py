from importlib.metadata import version as _pkg_version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
project = "image_match"
copyright = "2016, Ryan Henderson"
author = "Ryan Henderson"
try:
    release = _pkg_version("image-match")
except Exception:
    release = "0.0.0"
version = release
language = "en"
exclude_patterns = []
pygments_style = "sphinx"
todo_include_todos = True
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
htmlhelp_basename = "image_matchdoc"

latex_elements = {}
latex_documents = [
    (master_doc, "image_match.tex", "image_match Documentation", "Ryan Henderson", "manual"),
]

man_pages = [(master_doc, "image_match", "image_match Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "image_match",
        "image_match Documentation",
        author,
        "image_match",
        "One line description of project.",
        "Miscellaneous",
    ),
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
