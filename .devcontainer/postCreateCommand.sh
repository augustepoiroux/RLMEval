#!/usr/bin/env bash

git config --global --add safe.directory '*'

echo \"Installing Python dependencies\"...
uv sync
