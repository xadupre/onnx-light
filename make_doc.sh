echo "--"
echo "-- Builds cpp examples"
echo "--"
cd examples
ONNX_GIT_TAG=v1.21.0 ./build.sh 
cd ..
echo "--"
echo "-- Builds inline"
echo "--"
python setup.py build_ext --inplace
echo "--"
echo "-- Builds the documentation"
echo "--"
python -m sphinx docs dist/html -j auto
