#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

# Install a system package required by our library
# yum install -y atlas-devel

# Compile wheels
for python in /opt/python/cp36*/bin/python; do
    "${python}" -m pip wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
#  running tests from source directory fails due to some
#  pytest path manipulation issue or something,
#  running tests from /tmp works
cp -r /io/data /io/tests /tmp
for python in /opt/python/cp36*/bin/python; do
    "${python}" -m pip install astropy numpy scipy joblib pytest
    "${python}" -m pip install hdstats[test] --no-index -f /io/wheelhouse
    (cd "/tmp"; "${python}" -m pytest ./tests)
done
