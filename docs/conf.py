import datetime
import os
import subprocess
import sys
from docutils import nodes
from docutils.parsers.rst import Directive
import onnx_light

project = "onnx-light"
author = "onnx-light contributors"
release = onnx_light.__version__

extensions = [
    "breathe",
    "myst_parser",
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
    "sphinx_datatables",
    "sphinx_design",
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

# The Sphinx C++ domain parser does not know about this project's own visibility
# macro; declare it as an "id attribute" so it is skipped instead of raising a
# parsing error when documenting backend test case registration functions.
cpp_id_attributes = ["ONNX_LIGHT_BACKEND_TEST_LOCAL"]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        os.path.join(os.path.dirname(__file__), "examples", "runtime"),
        os.path.join(os.path.dirname(__file__), "examples", "proto"),
        os.path.join(os.path.dirname(__file__), "examples", "core"),
        os.path.join(os.path.dirname(__file__), "examples", "gradient"),
        os.path.join(os.path.dirname(__file__), "examples", "compute"),
    ],
    # path where to save gallery generated examples
    "gallery_dirs": [
        "auto_examples_runtime",
        "auto_examples_proto",
        "auto_examples_core",
        "auto_examples_gradient",
        "auto_examples_compute",
    ],
    # no parallelization to avoid conflict with environment variables
    "parallel": 1,
    # sorting
    "within_subsection_order": "ExampleTitleSortKey",
    # errors
    "abort_on_example_error": True,
    # recommendation
    "recommender": {"enable": True, "n_examples": 3, "min_df": 3, "max_df": 0.9},
    # ignore capture for matplotib axes
    "ignore_repr_types": "matplotlib\\.(text|axes)",
    # robubstness
    "reset_modules_order": "both",
    "reset_modules": ("matplotlib",),
}


# templates_path = ["_templates"]
exclude_patterns = ["build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["svg_zoom.js"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "github_url": "https://github.com/xadupre/onnx-light",
    "logo": {"image_light": "_static/logo.svg", "image_dark": "_static/logo.svg"},
    # Show every top-level navigation link on the same level instead of
    # collapsing the extra ones into a "More" dropdown.
    "header_links_before_dropdown": 10,
}
# pydata-sphinx-theme renders content tables inside scrollable containers.
copybutton_selector = "div.highlight pre, div.pst-scrollable-table-container > table"

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
}

suppress_warnings = [
    "intersphinx.external",
    "duplicate_declaration.cc",
    "duplicate_declaration.cpp",
    "duplicate_declaration.c",
    "ref.python",
    "source_code_parser.cc",
]

epkg_dictionary = {
    "Breathe": "https://breathe.readthedocs.io/",
    "C++ onnx-light examples": "https://github.com/xadupre/onnx-light/tree/main/examples",
    "Doxygen": "https://www.doxygen.nl/",
    "FlatBuffers": "https://flatbuffers.dev/",
    "libFuzzer": "https://llvm.org/docs/LibFuzzer.html",
    "libFuzzer command line": "https://llvm.org/docs/LibFuzzer.html#options",
    "libFuzzer entry point": "https://llvm.org/docs/LibFuzzer.html#fuzz-target",
    "lib_onnx_backend_test": (
        "https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_extensions/backend_test"
    ),
    "lib_onnx_kernels": (
        "https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_extensions/kernels"
    ),
    "Mermaid": "https://mermaid.js.org/",
    "mermaid.js": "https://mermaid.js.org/",
    "onnx": "https://github.com/onnx/onnx",
    "onnx_light/onnx_extensions/backend_test/cases": (
        "https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_extensions/backend_test/cases"
    ),
    "onnx_light/onnx_extensions/kernels": (
        "https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_extensions/kernels"
    ),
    "onnxruntime": "https://github.com/microsoft/onnxruntime",
    "OSS-Fuzz": "https://github.com/google/oss-fuzz",
    "protobuf": "https://protobuf.dev/",
    "Protocol Buffers": "https://protobuf.dev/",
    "sphinx-datatables": "https://pypi.org/project/sphinx-datatables/",
    "unittests/cc/onnx_extensions/backend_test": (
        "https://github.com/xadupre/onnx-light/tree/main/unittests/cc/onnx_extensions/backend_test"
    ),
    "unittests/cc/onnx_extensions/kernels": (
        "https://github.com/xadupre/onnx-light/tree/main/unittests/cc/onnx_extensions/kernels"
    ),
    "unittests/onnxl_vs_ort/test_backend_with_onnxruntime.py": (
        "https://github.com/xadupre/onnx-light/blob/main/unittests/onnxl_vs_ort/"
        "test_backend_with_onnxruntime.py"
    ),
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


def _build_to_svg_example(code: str) -> tuple[str, str]:
    """Builds the rendered SVG used in documentation from provided example code.

    Args:
        code: Python code executed to populate variable ``svg``.

    Returns:
        tuple[str, str]: Input Python code snippet and rendered SVG content.
    """
    namespace: dict[str, object] = {}
    exec(code, namespace)
    svg = namespace.get("svg")
    if not isinstance(svg, str):
        raise RuntimeError("The to-svg example must define 'svg' as a string.")
    return code, svg


class ToSvgExampleDirective(Directive):
    """Renders the `to_svg` example code and the resulting SVG image."""

    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False

    def run(self):
        """Returns the literal Python example and the rendered SVG output.

        Returns:
            list[nodes.Node]: Literal code block and raw HTML SVG nodes.
        """
        code = "\n".join(self.content).strip()
        if not code:
            raise self.error("to-svg-example requires Python code in the directive body.")
        code, svg = _build_to_svg_example(code)
        literal = nodes.literal_block(code, code)
        literal["language"] = "python"
        return [literal, nodes.raw("", svg, format="html")]


def _gallery_example_source(docname: str) -> str | None:
    """Returns the example ``.py`` source path for a generated gallery ``docname``.

    Args:
        docname: The Sphinx document name (for example
            ``auto_examples_proto/plot_onnx_time``).

    Returns:
        The absolute path to the example script, or None when ``docname`` does
        not correspond to a generated gallery example.
    """
    for examples_dir, gallery_dir in zip(
        sphinx_gallery_conf["examples_dirs"], sphinx_gallery_conf["gallery_dirs"]
    ):
        prefix = f"{gallery_dir}/"
        if docname.startswith(prefix):
            relative = docname[len(prefix) :]
            source = os.path.join(examples_dir, f"{relative}.py")
            return source if os.path.exists(source) else None
    return None


def _example_last_modified_date(path: str) -> str:
    """Returns the last modification date of ``path`` as an ISO ``YYYY-MM-DD`` string.

    The git commit date is used when available so the displayed date reflects
    when the example was last updated; it falls back to the file modification
    time when git information cannot be retrieved.

    Args:
        path: The absolute path to the example script.

    Returns:
        The last modification date formatted as ``YYYY-MM-DD``.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", path],
            cwd=_repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        date = result.stdout.strip()
        if date:
            return date
    except (subprocess.SubprocessError, OSError):
        pass
    return datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()


def _append_example_date(app, docname: str, source: list[str]) -> None:
    """Appends the source example's last modification date at the bottom of the page."""
    example_source = _gallery_example_source(docname)
    if example_source is None:
        return
    date = _example_last_modified_date(example_source)
    source[0] += f"\n\n.. rubric:: Example last updated\n\n:Date: {date}\n"


def setup(app) -> None:
    """Registers Sphinx hooks and custom directives used by this configuration."""
    app.connect("builder-inited", _on_builder_inited)
    app.connect("source-read", _append_example_date)
    app.add_directive("to-svg-example", ToSvgExampleDirective)
