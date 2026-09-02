#!/usr/bin/env bash
#
# Cut a release: bump SDK_VERSION / pyproject.toml, stamp CHANGELOG, commit, tag.
#
# Usage:
#   scripts/release.sh 1.1.0
#   scripts/release.sh 1.1.0 --dry-run

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-}"
DRY_RUN=0
shift || true
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: scripts/release.sh <major.minor.patch> [--dry-run]" >&2
  exit 2
fi

DATE="$(date +%F)"
TAG="v${VERSION}"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "tag $TAG already exists" >&2
  exit 1
fi

CURRENT="$(sed -n 's/^SDK_VERSION = "\(.*\)"$/\1/p' trtc_asr/sdkinfo.py)"
if [[ -z "$CURRENT" ]]; then
  echo "cannot read SDK_VERSION from trtc_asr/sdkinfo.py" >&2
  exit 1
fi

stamp_changelog() {
  python3 - "$VERSION" "$DATE" <<'PY'
import sys
from pathlib import Path
version, date = sys.argv[1], sys.argv[2]
path = Path("CHANGELOG.md")
text = path.read_text(encoding="utf-8")
heading = f"## [{version}] - {date}"
if heading in text:
    sys.exit(0)
old = "## [未发布]"
if old not in text:
    raise SystemExit("CHANGELOG.md has no '## [未发布]' section to stamp")
replacement = f"## [未发布]\n\n## [{version}] - {date}"
path.write_text(text.replace(old, replacement, 1), encoding="utf-8")
PY
}

echo "==> $CURRENT -> $VERSION"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "would update trtc_asr/sdkinfo.py and pyproject.toml"
  echo "would stamp CHANGELOG.md as ## [$VERSION] - $DATE"
  echo "would commit and tag $TAG"
  exit 0
fi

perl -i -pe "s/SDK_VERSION = \"$CURRENT\"/SDK_VERSION = \"$VERSION\"/" trtc_asr/sdkinfo.py
perl -i -pe "s/^version = \"$CURRENT\"/version = \"$VERSION\"/" pyproject.toml
stamp_changelog

git add trtc_asr/sdkinfo.py pyproject.toml CHANGELOG.md
git commit -m "chore: release $VERSION"
git tag -a "$TAG" -m "Release $VERSION"

echo "tagged $TAG. push with:"
echo "  git push origin HEAD && git push origin $TAG"
