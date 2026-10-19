#!/usr/bin/env bash

rm -rf dump_test*
rm -f eval*.onnx
rm -f plot*.onnx
rm -f plot*.png
rm -f temp*.onnx
rm -f test*.data
rm -f test*.onnx
rm -f dummy*.onnx
rm -f onnx*.md
rm -f onnx_light/onnx_py/*.so
rm -f Serialize*
rm -f bench_qwen*
rm -f qmoe*
find . -type d -name "__pycache__" -exec rm -rf {} +

if [ "$1" == "--all" ]; then
    rm -rf dist/
    rm -rf lib/
    rm -rf include/
    rm -rf build/
    rm -rf build-*/ build_*/ build2/ build_tests/ _skbuild/
    rm -rf htmlcov/ cover/ .coverage .coverage.*
    rm -rf .ruff_cache/
    rm -rf .pytest_cache/
    rm -rf docs/_build/ -rf
    rm -rf docs/auto_* -rf
    rm -rf docs/operators
    rm -rf docs/examples/*/*.onnx
    rm -rf docs/examples/*/*.data
    rm -rf docs/examples/*/*.xlsx
    rm -rf docs/examples/*/*.csv
    rm -rf docs/examples/*/*.png
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
fi
