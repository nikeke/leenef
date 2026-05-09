#!/usr/bin/env bash
set -euo pipefail

# Common Debian packages:
#   sudo apt install pandoc texlive-xetex texlive-latex-recommended \
#       texlive-latex-extra texlive-fonts-recommended lmodern
#
# Examples:
#   bash docs/build_ssup_pdf.sh docs/SSUP-I.md
#   bash docs/build_ssup_pdf.sh --verbose docs/SSUP-II.md docs/SSUP-II.pdf
#   bash docs/build_ssup_pdf.sh --regen-figures docs/SSUP-I.md

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC=""
OUT=""
ARXIV_OUT=""
regen_figures=0
verbose=0

usage() {
    cat <<EOF
usage: $(basename "$0") [options] [source.md] [output.pdf]

Build an SSUP PDF from a Markdown source.  If source.md is omitted,
docs/SSUP-I.md is used.  If output.pdf is omitted, the output is
build/<source-basename>.pdf.

Options:
  --regen-figures              Regenerate Paper-I figures before building.
  --arxiv-package [archive]    Create an arXiv source tarball.
  --verbose                    Print pandoc/xelatex commands and show missing-character warnings.
  -h, --help                   Show this help.
EOF
}

abspath() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$PWD" "$path"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regen-figures)
            regen_figures=1
            shift
            ;;
        --verbose)
            verbose=1
            shift
            ;;
        --arxiv-package)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                ARXIV_OUT="$(abspath "$2")"
                shift 2
            else
                ARXIV_OUT="__DEFAULT__"
                shift
            fi
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "error: unknown option $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$DOC" ]]; then
                DOC="$(abspath "$1")"
            elif [[ -z "$OUT" ]]; then
                OUT="$(abspath "$1")"
            else
                echo "error: too many positional arguments" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$DOC" ]]; then
    DOC="$ROOT/docs/SSUP-I.md"
fi

if [[ ! -f "$DOC" ]]; then
    echo "error: cannot find $DOC" >&2
    exit 1
fi

stem="$(basename "$DOC" .md)"
if [[ -z "$OUT" ]]; then
    OUT="$ROOT/build/$stem.pdf"
fi
if [[ "$ARXIV_OUT" == "__DEFAULT__" ]]; then
    ARXIV_OUT="$ROOT/build/$stem-arxiv.tar.gz"
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
    if [[ "$(basename "$DOC")" != "SSUP-I.md" ]]; then
        echo "error: --regen-figures is currently defined only for docs/SSUP-I.md" >&2
        exit 1
    fi
    python3 "$ROOT/docs/generate_paper_i_figures.py"
fi

mkdir -p "$(dirname "$OUT")"
if [[ -n "$ARXIV_OUT" ]]; then
    mkdir -p "$(dirname "$ARXIV_OUT")"
fi

tmpdir="$(mktemp -d)"
tmp_header="$tmpdir/header.tex"
tmp_tex="$tmpdir/$stem.tex"
tmp_log="$tmpdir/xelatex.log"
trap 'rm -rf "$tmpdir"' EXIT

run_cmd() {
    if [[ "$verbose" -eq 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    fi
    "$@"
}

write_header() {
    local mode="$1"
    local header_path="$2"

    cat >"$header_path" <<'EOF'
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
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}
\captionsetup{font=small}
\setlength{\emergencystretch}{3em}
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
EOF
    if [[ "$mode" == "pdf" ]]; then
        printf '\\graphicspath{{%s/}{%s/figures/}}\n' "$ROOT/docs" "$ROOT/docs" >>"$header_path"
    fi
}

generate_tex() {
    local mode="$1"
    local output_tex="$2"
    local header_path="$3"

    write_header "$mode" "$header_path"

    run_cmd pandoc \
        "$DOC" \
        --from=markdown+tex_math_dollars+footnotes+pipe_tables+table_captions+raw_tex \
        --to=latex \
        --standalone \
        --resource-path="$ROOT:$ROOT/docs:$ROOT/docs/figures" \
        --include-in-header="$header_path" \
        --variable=geometry:margin=1in \
        --variable=fontsize:11pt \
        --variable=linestretch:1.05 \
        --variable=colorlinks:true \
        --variable=linkcolor:blue \
        --variable=urlcolor:blue \
        --output="$output_tex"
}

print_missing_characters() {
    local log_path="$1"

    if grep -q 'Missing character:' "$log_path"; then
        echo "missing-character warnings:" >&2
        grep 'Missing character:' "$log_path" | sort -u >&2
        return 1
    fi
}

run_xelatex() {
    local tex_path="$1"
    local log_path="$2"

    if [[ "$verbose" -eq 1 ]]; then
        run_cmd xelatex -halt-on-error -interaction=nonstopmode -output-directory "$(dirname "$tex_path")" "$tex_path" 2>&1 | tee "$log_path"
    elif ! xelatex -halt-on-error -interaction=nonstopmode -output-directory "$(dirname "$tex_path")" "$tex_path" >"$log_path" 2>&1; then
        cat "$log_path" >&2
        exit 1
    fi
    print_missing_characters "$log_path" || return 1

    if [[ "$verbose" -eq 1 ]]; then
        run_cmd xelatex -halt-on-error -interaction=nonstopmode -output-directory "$(dirname "$tex_path")" "$tex_path" 2>&1 | tee "$log_path"
    elif ! xelatex -halt-on-error -interaction=nonstopmode -output-directory "$(dirname "$tex_path")" "$tex_path" >"$log_path" 2>&1; then
        cat "$log_path" >&2
        exit 1
    fi
    print_missing_characters "$log_path" || return 1
}

print_arxiv_metadata() {
    local pdf_path="$1"
    local figure_count="$2"
    local archive_path="$3"

    local title subtitle author_line email_line combined_title abstract_text abstract_plain page_count abstract_tmp

    title="$(awk '/^# / {sub(/^# /, ""); print; exit}' "$DOC")"
    subtitle="$(awk 'BEGIN{seen_title=0} /^# / {seen_title=1; next} seen_title && /^## / {sub(/^## /, ""); print; exit}' "$DOC")"
    author_line="$(awk 'BEGIN{after_subtitle=0} /^## / && !after_subtitle {after_subtitle=1; next} after_subtitle && NF && $0 !~ /^\[\^/ && $0 !~ /^---/ {gsub(/[[:space:]]+$/, ""); print; exit}' "$DOC")"
    email_line="$(awk 'BEGIN{after_subtitle=0; seen_author=0} /^## / && !after_subtitle {after_subtitle=1; next} after_subtitle && NF && $0 !~ /^\[\^/ && $0 !~ /^---/ {gsub(/[[:space:]]+$/, ""); if (seen_author == 0) {seen_author=1; next} print; exit}' "$DOC")"
    abstract_text="$(
        awk '
            /^## Abstract$/ {in_abstract=1; next}
            in_abstract && /^---$/ {exit}
            in_abstract {print}
        ' "$DOC" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
    )"
    abstract_tmp="$tmpdir/arxiv-abstract.md"
    printf '%s\n' "$abstract_text" >"$abstract_tmp"
    abstract_plain="$(
        pandoc "$abstract_tmp" \
            --from=markdown+tex_math_dollars+footnotes+raw_tex \
            --to=plain | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
    )"
    page_count="$(pdfinfo "$pdf_path" | awk -F': *' '/^Pages:/ {print $2; exit}')"
    email_line="$(printf '%s' "$email_line" | sed 's/\[\^[^]]*\]//g')"

    combined_title="$title"
    if [[ -n "$subtitle" ]]; then
        combined_title="$title: $subtitle"
    fi

    cat <<EOF

arXiv upload package: $archive_path
Suggested processor on arXiv: xelatex
Main source file: $stem.tex
Comments: ${page_count:-unknown pages} pages, $figure_count figures

Metadata deduced from the manuscript:
- Title: $title
- Subtitle: $subtitle
- Suggested combined title: $combined_title
- Authors: $author_line
- Contact line: $email_line
- Abstract (plain text for the form): $abstract_plain
- Abstract (source form): $abstract_text

You will still need to choose or confirm on arXiv:
- primary subject class / cross-lists
- license
- comments edits, if any
- journal reference / DOI / report number, if applicable
- whether to use TeX Live 2025 or 2023 (this package is intended for xelatex; 2025 is the natural choice)
EOF
}

generate_tex "pdf" "$tmp_tex" "$tmp_header"
run_xelatex "$tmp_tex" "$tmp_log"

cp "$tmpdir/$stem.pdf" "$OUT"

echo "wrote $OUT"

if [[ -n "$ARXIV_OUT" ]]; then
    arxiv_dir="$tmpdir/arxiv-src"
    arxiv_header="$tmpdir/arxiv-header.tex"
    arxiv_tex="$arxiv_dir/$stem.tex"
    mkdir -p "$arxiv_dir"

    generate_tex "arxiv" "$arxiv_tex" "$arxiv_header"

    mapfile -t used_figures < <(
        grep -oE 'includegraphics(\[[^]]*\])?\{[^}]+\}' "$arxiv_tex" \
            | sed -E 's/.*\{([^}]+)\}/\1/' \
            | sort -u
    )

    for figure_path in "${used_figures[@]}"; do
        mkdir -p "$arxiv_dir/$(dirname "$figure_path")"
        cp "$ROOT/docs/$figure_path" "$arxiv_dir/$figure_path"
    done

    tar -C "$arxiv_dir" -czf "$ARXIV_OUT" "$stem.tex" "${used_figures[@]}"
    echo "wrote $ARXIV_OUT"
    print_arxiv_metadata "$OUT" "${#used_figures[@]}" "$ARXIV_OUT"
fi
