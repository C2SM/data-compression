#!/bin/bash
set -euo pipefail

pip install --upgrade pip

# Install dc_toolkit
pip install -e .
CC="$(command -v mpicc)" pip install --no-binary=mpi4py mpi4py

# ---------------- EBCC: always hard reset to origin's default branch ----------------
EBCC_DIR="EBCC"
EBCC_REMOTE="https://github.com/spcl/EBCC.git"

echo "[EBCC] Target remote: $EBCC_REMOTE"

# if folder exists but not a git repo, remove it
if [ -d "$EBCC_DIR" ] && [ ! -d "$EBCC_DIR/.git" ]; then
  echo "[EBCC] '$EBCC_DIR' exists but is not a git repo. Removing..."
  rm -rf "$EBCC_DIR"
fi

if [ ! -d "$EBCC_DIR/.git" ]; then
  echo "[EBCC] Cloning fresh..."
  git clone --recursive "$EBCC_REMOTE" "$EBCC_DIR"
fi

pushd "$EBCC_DIR" >/dev/null

# Ensure correct origin URL
CUR_URL="$(git remote get-url origin || true)"
if [ "$CUR_URL" != "$EBCC_REMOTE" ]; then
  echo "[EBCC] Repointing origin from '$CUR_URL' to '$EBCC_REMOTE'"
  git remote set-url origin "$EBCC_REMOTE"
fi

# Fetch latest (including submodule metadata)
echo "[EBCC] Fetching origin..."
git fetch --prune --tags origin
git submodule sync --recursive

# Determine default branch with robust fallbacks
get_default_branch() {
  local defb
  defb="$(git remote show origin 2>/dev/null | awk '/HEAD branch/ {print $NF}')" || true
  if [ -z "${defb:-}" ]; then
    # Fallbacks if remote HEAD not advertised
    for b in main master; do
      if git ls-remote --heads origin "$b" | grep -q "$b"; then
        defb="$b"; break
      fi
    done
  fi
  if [ -z "${defb:-}" ]; then
    echo "[EBCC] ERROR: Could not determine origin default branch." >&2
    exit 1
  fi
  printf "%s" "$defb"
}

DEFAULT_BRANCH="$(get_default_branch)"
REMOTE_REF="origin/$DEFAULT_BRANCH"
echo "[EBCC] Remote default branch: $DEFAULT_BRANCH"

# Record pre-update SHA (if any)
LOCAL_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'none')"
REMOTE_SHA="$(git rev-parse "$REMOTE_REF")"
echo "[EBCC] Local:  $LOCAL_SHA"
echo "[EBCC] Remote: $REMOTE_SHA"

# Force to default branch regardless of current branch/detached state
# 1) ensure local branch exists and tracks origin/<default>
if git show-ref --verify --quiet "refs/heads/$DEFAULT_BRANCH"; then
  git checkout -q "$DEFAULT_BRANCH"
else
  git checkout -q -B "$DEFAULT_BRANCH" "$REMOTE_REF"
fi
git branch --set-upstream-to="$REMOTE_REF" "$DEFAULT_BRANCH" >/dev/null 2>&1 || true

# 2) nuke local changes & untracked, then hard reset to remote tip
echo "[EBCC] Discarding local changes and untracked files..."
git reset --hard
git clean -fdx

echo "[EBCC] Hard resetting to $REMOTE_REF..."
git reset --hard "$REMOTE_REF"

# Update/init submodules to the exact commits
echo "[EBCC] Updating submodules..."
git submodule update --init --recursive

# Recompute post-reset SHA
NEW_SHA="$(git rev-parse HEAD)"
echo "[EBCC] Now at: $NEW_SHA"

# Only reinstall if the commit changed
if [ "$LOCAL_SHA" = "$NEW_SHA" ]; then
  echo "[EBCC] No upstream changes. Skipping reinstall."
else
  echo "[EBCC] Upstream changed ($LOCAL_SHA -> $NEW_SHA). Reinstalling EBCC (editable, zarr)..."
  pip install -e ".[zarr]"
fi

popd >/dev/null
# ---------------- end EBCC section ----------------
