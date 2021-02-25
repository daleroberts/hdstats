define HELP
make test|doc|sdist|upload|manylinux_wheels|clean

  test             - compile in place and run tests
  doc              - build documentation
  sdist            - build source distribution
  manylinux_wheels - build highly compatible binary wheels (uses docker)
endef
export HELP

help:
	@echo "$$HELP"

inplace:
	python3 setup.py build_ext -i

test: inplace
	pytest

clean:
	@rm -fr build dist
	@rm -fr hdstats/*.so
	@rm -fr hdstats/dtw.c
	@rm -fr hdstats/geomedian.c
	@rm -fr hdstats/pcm.c
	@rm -fr hdstats/ts.c
	@rm -fr hdstats.egg-info
	@rm -fr hdstats/__pycache__
	@rm -fr .pytest_cache
	@rm -fr .ipynb_checkpoints
	@rm -fr .hypothesis
	@rm -fr .eggs
	@rm -fr tests/__pycache__

doc: docs/README_.md docs/plots.py
#-- requires `pip3 install readme2tex cairosvg`
	@python3 -m readme2tex --output README.md --svgdir docs --project hdstats --usepackage "stix" --rerender docs/README_.md
	@python3 docs/plots.py
#-- hack to make images work
	@for f in $(wildcard docs/*.svg); do cairosvg -d 300 $$f -o $${f/svg/png}; done
	@sed -i~ -e 's/svg/png/g; s/rawgit/github/g; s/master/raw\\\/master/g' README.md
	@rm -fr *~
	git rm --ignore-unmatch --cached $(wildcard docs/*.svg) $(wildcard docs/*.png)
	git add $(wildcard docs/*.svg) $(wildcard docs/*.png)
	git add README.md docs/README_.md
	git commit -m 'Update README'
	git push

sdist:
	@rm -fr dist/
	python3 setup.py sdist

upload:
	python3 setup.py sdist register upload

DOCKER_IMAGE = quay.io/pypa/manylinux2010_x86_64
PLAT = manylinux2010_x86_64

manylinux_wheels:
	docker pull ${DOCKER_IMAGE}
	docker run --rm -e PLAT=${PLAT} -v `pwd`:/io ${DOCKER_IMAGE} ${PRE_CMD} /io/build-wheels.sh
	@echo "Wheels are in ./wheelhouse"
	@ls -la ./wheelhouse


.PHONY: manylinux_wheels sdist clean test inplace
