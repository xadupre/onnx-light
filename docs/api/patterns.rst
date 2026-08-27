.. _l-api-pattern-catalog:

Pattern catalogue
=================

This table is generated from the patterns registered by the installed
``onnx_light`` package. Summaries and rewrite graphs come directly from the
C++ Doxygen comments, so adding a registered pattern or changing its graph
automatically updates this page.

.. runpython::
    :rst:

    from onnx_light.doc import render_rst_pattern_catalog

    print(render_rst_pattern_catalog())
