# Requires tectonic (https://tectonic-typesetting.github.io)
all: figs
	XDG_CACHE_HOME=$${XDG_CACHE_HOME:-/tmp/xdg-cache} tectonic -X compile paper.tex
figs:
	cd .. && python3 paper/make_figs.py
.PHONY: all figs
