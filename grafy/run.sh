#!/usr/bin/env bash
# run.sh — spouštěcí skript pro Linux/macOS
# usage: ./run.sh [command] [file]
# example: ./run.sh summary 01.tg

CMD=${1:-summary}
FILE=${2:-01.tg}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate

python -m pip install --upgrade pip

PKG_DIR="./packages"
if [ ! -d "$PKG_DIR" ]; then
  echo "Složka packages nenalezena. Zkopírujte offline balíčky do ./packages"
else
  python -m pip install --no-index --find-links "$PKG_DIR" -r requirements.txt
fi

python3 graph_tool.py "$CMD" "$FILE"
