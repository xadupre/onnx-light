import sys
import onnx_light

project = "onnx-light"
author = "onnx-light contributors"
release = onnx_light.__version__

extensions = [
    "sphinx.ext.autodoc",
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
    "sphinx_runpython.runpython",
    "sphinx_runpython.runmermaid",
    "matplotlib.sphinxext.plot_directive",
]

sphinx_gallery_conf = {"examples_dirs": ["examples"], "gallery_dirs": ["auto_examples"]}

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

suppress_warnings = ["intersphinx.external"]
