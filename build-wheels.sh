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
for PYBIN in /opt/python/cp36*/bin; do
#    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
for PYBIN in /opt/python/cp36*/bin/; do
    "${PYBIN}/pip" install astropy numpy scipy joblib pytest
    "${PYBIN}/pip" install hdstats[test] --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/pytest" /io/tests)
done
