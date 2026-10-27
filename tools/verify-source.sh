#!/usr/bin/env bash
# Prove that strata/ is exactly upstream v0.2.2 plus patches/*.patch, nothing else.
# Clones upstream into a temp dir, applies the patches with `git am`, and diffs.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

git clone -q --filter=blob:none --no-checkout https://github.com/Niko1221/Strata "$tmp/upstream"
git -C "$tmp/upstream" checkout -q 011c9e60
git -C "$tmp/upstream" -c user.name=verify -c user.email=verify@localhost am -q "$here"/patches/0*.patch

mkdir "$tmp/expected"
git -C "$tmp/upstream" archive HEAD | tar -x -C "$tmp/expected"
if diff -r "$tmp/expected" "$here/strata" --exclude=build-p100 --exclude=.venv >/dev/null; then
    echo "OK: strata/ == v0.2.2 (011c9e60) + $(ls "$here"/patches/0*.patch | wc -l) patches"
else
    diff -rq "$tmp/expected" "$here/strata" --exclude=build-p100 --exclude=.venv
    echo "MISMATCH: strata/ differs from v0.2.2 + patches" >&2
    exit 1
fi
