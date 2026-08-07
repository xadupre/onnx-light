echo "--"
echo "clean branches"
echo "--"
git branch | grep -v "^\*\|main$" | xargs git branch -D
git branch
echo "--"
echo "-- Builds cpp examples"
echo "--"
cd examples
ONNX_GIT_TAG=v1.21.0 ./build.sh || exit 1
cd ..
echo "--"
echo "-- Builds inline (no protobuf in benchmarks)"
echo "--"
CMAKE_ARGS="-DONNX_LIGHT_BUILD_TESTS=ON" python setup.py build_ext --inplace || exit 1
echo "--"
echo "-- Runs one test."
echo "--"
ctest --test-dir build/temp/ --output-on-failure -R MatchesOnnxLib || exit 1
echo "--"
echo "-- Builds the documentation"
echo "--"
python -m sphinx docs dist/html -j 2 || exit 1
