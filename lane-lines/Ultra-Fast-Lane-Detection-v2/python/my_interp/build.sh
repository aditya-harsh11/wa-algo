rm -rf build &&
rm -rf *.egg-info &&
rm -rf dist &&
python setup.py build_ext --inplace &&
pip install .
