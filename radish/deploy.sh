#!/bin/bash
set -e

SITE_DIR="$(cd "$(dirname "$0")" && pwd)"
GITHUB_DIR="$SITE_DIR/../irenetrampoline.github.io"
DEST_DIR="$GITHUB_DIR/static/radish"
EDIT_DEST_DIR="$GITHUB_DIR/static/radish-edit"

echo "Building Radish..."
cd "$SITE_DIR"
export PATH="/usr/local/opt/node@22/bin:$PATH"
npm run build
npx vite build --config vite.config.edit.js

echo "Copying to site static directory..."
cp dist/index.html "$DEST_DIR/index.html"
mkdir -p "$EDIT_DEST_DIR"
cp dist-edit/edit.html "$EDIT_DEST_DIR/index.html"

echo "Pushing to GitHub..."
cd "$GITHUB_DIR"
git add static/radish/index.html static/radish-edit/index.html
git commit -m "Update Radish app"
git push

echo "Done! GitHub Pages will deploy shortly."
