#!/usr/bin/env bash
#
# Downloads the benchmark tables that are not in the repo from the shared Google
# Drive folders into data_src/<benchmark>/dataset/: the KramaBench CSVs (of which
# git tracks only the smaller datasets') and the DataAgentBench database files.
# Only curl is required.
#
# Each Drive folder holds one subfolder per benchmark, with its files either in a
# dataset/ subfolder (KramaBench) or directly (DataAgentBench), e.g.
#   astronomy/dataset/<file>.csv    -> data_src/astronomy/dataset/<file>.csv
#   patents/patent_publication.db   -> data_src/patents/dataset/patent_publication.db
#
# Existing files are never overwritten. Interrupted downloads are kept as
# <file>.part and resumed on the next run.
#
# Usage:
#   data_src/download.sh                     # every benchmark
#   data_src/download.sh astronomy patents   # only these benchmarks
#   data_src/download.sh --dry-run

set -uo pipefail

DRIVE_FOLDERS=(
    "1jL2T8N8X8ncFyERdZ1lg0DcKZDwFqKaG"  # KramaBench
    "14WMGOuBzuL2S5hgGReTe-r4IE2OcEekU"  # DataAgentBench
)
DATA_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
BENCHMARKS=()
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) BENCHMARKS+=("${arg%/}") ;;
    esac
done

MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

html_unescape() {
    sed -e 's/&amp;/\&/g' -e 's/&#39;/'"'"'/g' -e 's/&quot;/"/g' -e 's/&lt;/</g' -e 's/&gt;/>/g'
}

wanted() {
    [ ${#BENCHMARKS[@]} -eq 0 ] && return 0
    local b
    for b in "${BENCHMARKS[@]}"; do [ "$b" = "$1" ] && return 0; done
    return 1
}

# Prints "<kind>\t<id>\t<name>" for each entry of a public Drive folder.
list_folder() {
    local html
    html="$(curl -fsSL "https://drive.google.com/embeddedfolderview?id=$1")" || return 1
    printf '%s' "$html" \
        | grep -oE 'https://drive\.google\.com/(drive/folders|file/d)/[A-Za-z0-9_-]+|flip-entry-title">[^<]+' \
        | sed -E 's#^https://drive\.google\.com/drive/folders/#folder	#; s#^https://drive\.google\.com/file/d/#file	#; s#^flip-entry-title">##' \
        | paste - - \
        | html_unescape
}

download_file() {
    local id="$1" dest="$2" part="$2.part" status
    mkdir -p "$(dirname "$dest")" || return 1
    status="$(curl -fL --progress-bar -C - -o "$part" -w '%{http_code} %{content_type}' \
        "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t")" || return 1
    case "$status" in
        416\ *) ;;  # the .part from an earlier run is already complete
        *\ text/html*) rm -f "$part"; echo "Drive returned an HTML page instead of the file" >&2; return 1 ;;
    esac
    [ -e "$dest" ] && { rm -f "$part"; return 0; }
    mv "$part" "$dest"
}

# Appends "<id>\t<benchmark>/dataset/<file>" for each table file of a benchmark's Drive folder.
add_benchmark() {
    local folder_id="$1" name="$2" entries kind id fname
    entries="$(list_folder "$folder_id")" || { echo "Failed to list $name/ on Drive." >&2; return 1; }
    local dataset_id
    dataset_id="$(awk -F'\t' '$1 == "folder" && $3 == "dataset" {print $2}' <<< "$entries")"
    if [ -n "$dataset_id" ]; then
        entries="$(list_folder "$dataset_id")" || { echo "Failed to list $name/dataset/ on Drive." >&2; return 1; }
    fi
    while IFS=$'\t' read -r kind id fname; do
        [ "$kind" = "file" ] && printf '%s\t%s\n' "$id" "$name/dataset/$fname" >> "$MANIFEST"
    done <<< "$entries"
}

echo "Listing the Google Drive folders..."
for root in "${DRIVE_FOLDERS[@]}"; do
    entries="$(list_folder "$root")" || { echo "Failed to list the Drive folder $root." >&2; exit 1; }
    while IFS=$'\t' read -r kind id name; do
        [ "$kind" = "folder" ] && wanted "$name" || continue
        add_benchmark "$id" "$name" || exit 1
    done <<< "$entries"
done

total="$(wc -l < "$MANIFEST" | tr -d ' ')"
[ "$total" -gt 0 ] || { echo "No files found${BENCHMARKS[*]:+ for: ${BENCHMARKS[*]}}."; exit 0; }

i=0; n_available=0; n_downloaded=0; failed=()
while IFS=$'\t' read -r id rel; do
    i=$((i + 1))
    dest="$DATA_SRC/$rel"
    if [ -s "$dest" ]; then
        echo "[$i/$total] $rel  already available"
        n_available=$((n_available + 1))
    elif [ "$DRY_RUN" -eq 1 ]; then
        echo "[$i/$total] $rel  would download"
    else
        echo "[$i/$total] $rel  downloading"
        if download_file "$id" "$dest" < /dev/null; then
            n_downloaded=$((n_downloaded + 1))
        else
            echo "[$i/$total] $rel  FAILED (partial data kept as .part for resume)"
            failed+=("$rel")
        fi
    fi
done < "$MANIFEST"

echo
echo "Done: $n_available already available, $n_downloaded downloaded, ${#failed[@]} failed."
if [ ${#failed[@]} -gt 0 ]; then
    printf '  failed: %s\n' "${failed[@]}"
    echo "Run it again to retry; partial downloads resume."
    exit 1
fi
