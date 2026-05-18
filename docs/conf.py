import os
import subprocess
import sys
import onnx_light

project = "onnx-light"
author = "onnx-light contributors"
release = onnx_light.__version__

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

# Paths used by Breathe (must be set at module level so Breathe can read them).
_docs_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_docs_dir)
_doxygen_output_dir = os.path.join(_repo_root, "dist", "doxygen")
_doxygen_xml_dir = os.path.join(_doxygen_output_dir, "xml")

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


def _on_builder_inited(app) -> None:
    """Generates operator RST pages and runs Doxygen when Sphinx initialises its builder."""
    from onnx_light.doc import generate_operators_doc
    from sphinx.util import logging

    logger = logging.getLogger(__name__)

    # Run Doxygen to generate XML for Breathe.
    logger.info("[doxygen] running doxygen to generate XML for Breathe ...")
    os.makedirs(_doxygen_xml_dir, exist_ok=True)
    doxygen_result = subprocess.run(
        ["doxygen", os.path.join(_docs_dir, "Doxyfile")],
        cwd=_docs_dir,
        capture_output=True,
        env={**os.environ, "DOXYGEN_OUTPUT_DIR": _doxygen_output_dir},
    )
    stdout = doxygen_result.stdout.decode(errors="replace")
    stderr = doxygen_result.stderr.decode(errors="replace")
    if stdout:
        logger.info("[doxygen] stdout:\n%s", stdout)
    if stderr:
        logger.info("[doxygen] stderr:\n%s", stderr)
    if doxygen_result.returncode != 0:
        logger.warning("[doxygen] exited with code %d", doxygen_result.returncode)
    else:
        logger.info("[doxygen] completed successfully (return code 0)")

    operators_dir = os.path.join(app.srcdir, "operators")
    generate_operators_doc(operators_dir, progress_callback=logger.info)


def setup(app) -> None:
    """Registers Sphinx event hooks used by this configuration."""
    app.connect("builder-inited", _on_builder_inited)
