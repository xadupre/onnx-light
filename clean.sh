rm dump_test* -rf
rm eval*.onnx
rm plot*.onnx
rm plot*.png
rm temp*.onnx
rm test*.data
rm test*.onnx
rm dummy*.onnx
rm onnx*.md
rm onnx_light/onnx_py/*.so
find . -type d -name "__pycache__" -exec rm -rf {} +

if [ "$1" == "--all" ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} 
    rm -rf dist/
    rm -rf lib/
    rm -rf include/
    rm -rf build/
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
