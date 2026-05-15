import os
import subprocess
import sys
import onnx_light

project = "onnx-light"
author = "onnx-light contributors"
release = onnx_light.__version__

# ---------------------------------------------------------------------------
# Generate operator documentation pages (docs/operators/*.rst).
# The generator uses the ``onnx`` package to retrieve schema data; if that
# package is absent the step is skipped silently.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext"))
import gen_operators  # noqa: E402

_operators_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "operators")
gen_operators.generate(_operators_dir)

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.duration",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_runpython.epkg",
    "sphinx_runpython.runpython",
    "sphinx_runpython.runmermaid",
    "sphinxcontrib.mermaid",
    "matplotlib.sphinxext.plot_directive",
]

# Run Doxygen to generate XML for Breathe.
_docs_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_docs_dir)
_doxygen_output_dir = os.path.join(_repo_root, "dist", "doxygen")
_doxygen_xml_dir = os.path.join(_doxygen_output_dir, "xml")
os.makedirs(_doxygen_xml_dir, exist_ok=True)
_doxygen_result = subprocess.run(
    ["doxygen", os.path.join(_docs_dir, "Doxyfile")],
    cwd=_docs_dir,
    capture_output=True,
    env={**os.environ, "DOXYGEN_OUTPUT_DIR": _doxygen_output_dir},
)
if _doxygen_result.returncode != 0:
    import warnings

    warnings.warn(
        f"Doxygen exited with code {_doxygen_result.returncode}:\n"
        + _doxygen_result.stderr.decode(errors="replace"),
        stacklevel=1,
    )

# Breathe configuration
breathe_projects = {"onnx-light": _doxygen_xml_dir}
breathe_default_project = "onnx-light"

sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "nested_sections": True,
}

# templates_path = ["_templates"]
exclude_patterns = ["build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/onnx-light",
    "logo": {"image_light": "_static/logo.svg", "image_dark": "_static/logo.svg"},
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
}

suppress_warnings = [
    "intersphinx.external",
    "duplicate_declaration.cpp",
    "duplicate_declaration.c",
    "ref.python",
    "source_code_parser.cpp",
]

epkg_dictionary = {
    "C++ onnx-light examples": "https://github.com/xadupre/onnx-light/tree/main/examples",
    "onnx": "https://github.com/onnx/onnx",
}
