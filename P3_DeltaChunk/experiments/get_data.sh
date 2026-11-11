#!/usr/bin/env bash
# EditStream-RAG raw data: simplewiki 2026-11-01 full-history dump (.7z) + pinned git clones.
# Resumable: curl -C - continues a partial download; clones are skipped if present.
set -euo pipefail
P3="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$P3/data/raw"; mkdir -p "$RAW" "$P3/results" "$P3/logs"
DUMP=simplewiki-20261101-pages-meta-history.xml.7z
BASE=https://dumps.wikimedia.org/simplewiki/20261101

part_wiki() {
  want=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["simplewiki"]["sha1"])' "$P3/results/data_versions.json")
  if [ -f "$RAW/$DUMP.ok" ] && [ -f "$RAW/$DUMP" ]; then
    got=$(shasum -a 1 "$RAW/$DUMP" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then echo "wiki: sha1 OK $got"; return; fi
    echo "wiki: existing archive hash does not match the pinned release" >&2
    return 1
  fi
  for attempt in 1 2 3 4 5; do
    curl -L --fail -C - --retry 5 --retry-delay 10 -o "$RAW/$DUMP" "$BASE/$DUMP" && break || sleep 30
  done
  got=$(shasum -a 1 "$RAW/$DUMP" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then echo "wiki: SHA1 MISMATCH got=$got want=$want"; exit 1; fi
  # bsdtar 3.5.3 rejects this archive's entry name; list it with py7zr (data/sevenzip_cat.py streams it)
  "$P3/.venv/bin/python" -c "import py7zr,sys;print(py7zr.SevenZipFile(sys.argv[1]).list())" "$RAW/$DUMP"
  echo "$got" > "$RAW/$DUMP.ok"; echo "wiki: sha1 OK $got"
}

part_git() {
  for name in k8s cpython; do
    read -r url sha < <(python3 -c 'import json, sys; p = json.load(open(sys.argv[1]))[sys.argv[2]]; print(p["url"], p["head_sha"])' "$P3/results/data_versions.json" "$name")
    repo="$RAW/$name"
    if [ ! -d "$repo/.git" ]; then
      git clone --quiet --no-checkout --filter=blob:limit=2m "$url.git" "$repo"
      git -C "$repo" update-ref --no-deref HEAD "$sha"
    elif [ "$(git -C "$repo" rev-parse HEAD)" != "$sha" ]; then
      echo "$name: existing clone is not at pinned revision $sha ($repo)" >&2
      return 1
    fi
    echo "$name: $sha"
  done
}

case "${1:-all}" in
  wiki) part_wiki ;;
  git) part_git ;;
  all) part_git & git_pid=$!; part_wiki; wait "$git_pid" ;;
esac
