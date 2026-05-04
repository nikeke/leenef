#!/usr/bin/env bash
set -euo pipefail

# Common Debian packages:
#   sudo apt install pandoc texlive-xetex texlive-latex-recommended \
#       texlive-latex-extra texlive-fonts-recommended lmodern

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/SSUP-I.md"
OUT="${1:-$ROOT/build/SSUP-I.pdf}"

regen_figures=0
if [[ "${1:-}" == "--regen-figures" ]]; then
    regen_figures=1
    OUT="${2:-$ROOT/build/SSUP-I.pdf}"
fi

if [[ ! -f "$DOC" ]]; then
    echo "error: cannot find $DOC" >&2
    exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "error: pandoc is required" >&2
    echo "install with: sudo apt install pandoc" >&2
    exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
    echo "error: xelatex is required" >&2
    echo "install with: sudo apt install texlive-xetex texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern" >&2
    exit 1
fi

if ! kpsewhich lmodern.sty >/dev/null 2>&1; then
    echo "error: LaTeX package lmodern.sty is required" >&2
    echo "install with: sudo apt install lmodern" >&2
    exit 1
fi

if [[ "$regen_figures" -eq 1 ]]; then
    python3 "$ROOT/docs/generate_paper_i_figures.py"
fi

mkdir -p "$(dirname "$OUT")"

tmp_header="$(mktemp)"
trap 'rm -f "$tmp_header"' EXIT

cat >"$tmp_header" <<'EOF'
\usepackage{microtype}
\usepackage{bookmark}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\captionsetup{font=small}
\setlength{\emergencystretch}{3em}
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
EOF

pandoc \
    "$DOC" \
    --from=markdown+tex_math_dollars+footnotes+pipe_tables+table_captions+raw_tex \
    --to=pdf \
    --pdf-engine=xelatex \
    --standalone \
    --resource-path="$ROOT:$ROOT/docs:$ROOT/docs/figures" \
    --include-in-header="$tmp_header" \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --variable=linestretch:1.05 \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --variable=urlcolor:blue \
    --output="$OUT"

echo "wrote $OUT"
