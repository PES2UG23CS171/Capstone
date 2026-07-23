#!/bin/zsh
# voiceiso launcher — always uses the project venv (the #1 setup mistake is
# running with the system python, which silently drops DFN3 to passthrough).
cd "$(dirname "$0")"
exec .venv_poc/bin/python -m app.main "$@"
