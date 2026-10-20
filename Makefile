
.PHONY: install run debug clean lint package pclean

VENV = /goinfre/fadarwis/venvs/call_me_maybe

install:
	uv sync
run:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv run python -m src
debug:

clean:

lint:	
