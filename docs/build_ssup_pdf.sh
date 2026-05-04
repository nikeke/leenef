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

tmpdir="$(mktemp -d)"
tmp_header="$tmpdir/header.tex"
tmp_tex="$tmpdir/SSUP-I.tex"
tmp_log="$tmpdir/xelatex.log"
trap 'rm -rf "$tmpdir"' EXIT

font_args=()
if command -v fc-match >/dev/null 2>&1; then
    if fc-match "DejaVu Serif" >/dev/null 2>&1; then
        font_args+=(-V "mainfont=DejaVu Serif" -V "sansfont=DejaVu Sans")
    fi
    if fc-match "DejaVu Sans Mono" >/dev/null 2>&1; then
        font_args+=(-V "monofont=DejaVu Sans Mono")
    fi
fi

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
    --lua-filter="$ROOT/docs/ssup_math.lua" \
    --resource-path="$ROOT:$ROOT/docs:$ROOT/docs/figures" \
    --include-in-header="$tmp_header" \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt \
    --variable=linestretch:1.05 \
    --variable=colorlinks:true \
    --variable=linkcolor:blue \
    --variable=urlcolor:blue \
    "${font_args[@]}" \
    --output="$tmp_tex"

if ! xelatex -halt-on-error -interaction=nonstopmode -output-directory "$tmpdir" "$tmp_tex" >"$tmp_log" 2>&1; then
    cat "$tmp_log" >&2
    exit 1
fi

if ! xelatex -halt-on-error -interaction=nonstopmode -output-directory "$tmpdir" "$tmp_tex" >"$tmp_log" 2>&1; then
    cat "$tmp_log" >&2
    exit 1
fi

cp "$tmpdir/SSUP-I.pdf" "$OUT"

echo "wrote $OUT"
