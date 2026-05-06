#!/usr/bin/env sh
set -eu

REPO_URL="https://github.com/bodaay/HuggingFaceModelDownloader.git"
REF="${HFDOWNLOADER_REF:-master}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
INSTALL_DIR="${MODEL_MANAGER_HFDOWNLOADER_INSTALL_DIR:-$SCRIPT_DIR/bin}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to fetch HuggingFaceModelDownloader." >&2
  exit 1
fi

if ! command -v go >/dev/null 2>&1; then
  echo "Go is required to build hfdownloader. Install Go, then rerun this script." >&2
  exit 1
fi

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hfdownloader.XXXXXX")
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

mkdir -p "$INSTALL_DIR"
echo "Fetching $REPO_URL ($REF)"
git clone --depth 1 "$REPO_URL" "$TMP_DIR/HuggingFaceModelDownloader"
cd "$TMP_DIR/HuggingFaceModelDownloader"
git fetch --depth 1 origin "$REF" >/dev/null 2>&1 || true
git checkout "$REF" >/dev/null 2>&1 || true

echo "Building hfdownloader without Docker"
go build -o "$INSTALL_DIR/hfdownloader" ./cmd/hfdownloader
cp LICENSE "$INSTALL_DIR/HuggingFaceModelDownloader.LICENSE"

echo "Installed: $INSTALL_DIR/hfdownloader"
"$INSTALL_DIR/hfdownloader" version || true
