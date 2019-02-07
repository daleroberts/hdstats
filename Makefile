
inplace:
	python3 setup.py build_ext -i

test: inplace
	pytest

clean:
	@rm -fr build dist
	@rm -fr hdstats/*.so
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
