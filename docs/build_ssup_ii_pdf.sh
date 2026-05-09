#!/usr/bin/env bash
set -euo pipefail

# Build SSUP-II.pdf from docs/SSUP-II.md.
#
# Common Debian packages:
#   sudo apt install pandoc texlive-xetex texlive-latex-recommended \
#       texlive-latex-extra texlive-fonts-recommended lmodern

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/docs/SSUP-II.md"
OUT="$ROOT/build/SSUP-II.pdf"

if [[ $# -ge 1 ]]; then
    OUT="$1"
fi

if [[ ! -f "$DOC" ]]; then
    echo "error: cannot find $DOC" >&2
    exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "error: pandoc is required" >&2
    exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
    echo "error: xelatex is required" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"

tmpdir="$(mktemp -d)"
tmp_header="$tmpdir/header.tex"
tmp_tex="$tmpdir/SSUP-II.tex"
tmp_log="$tmpdir/xelatex.log"
trap 'rm -rf "$tmpdir"' EXIT

cat >"$tmp_header" <<'EOF'
\usepackage{amsmath}
\usepackage{amssymb}
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
printf '\\graphicspath{{%s/}{%s/figures/}}\n' "$ROOT/docs" "$ROOT/docs" >>"$tmp_header"

pandoc \
    "$DOC" \
    --from=markdown+tex_math_dollars+footnotes+pipe_tables+table_captions+raw_tex \
    --to=latex \
    --standalone \
    --resource-path="$ROOT:$ROOT/docs:$ROOT/docs/figures" \
    --include-in-header="$tmp_header" \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --variable=linestretch:1.05 \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --variable=urlcolor:blue \
    --output="$tmp_tex"

run_xelatex() {
    if ! xelatex -halt-on-error -interaction=nonstopmode -output-directory "$tmpdir" "$tmp_tex" >"$tmp_log" 2>&1; then
        cat "$tmp_log" >&2
        exit 1
    fi
}

run_xelatex
run_xelatex

cp "$tmpdir/SSUP-II.pdf" "$OUT"
echo "wrote $OUT"
