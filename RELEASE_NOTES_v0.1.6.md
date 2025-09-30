# Release v0.1.6

## Summary

This release includes workflow improvements and version bump to v0.1.6.

## Changes Since v0.1.5

- Update OpenCV CUDA workflow to manual (#457)
- Bump version to 0.1.6

## What Was Done

1. ✅ Updated version in `pyproject.toml` from 0.1.5 to 0.1.6
2. ✅ Updated version in `ui/package.json` from 0.1.5 to 0.1.6
3. ✅ Verified the release workflow builds successfully
4. ✅ Tested UI build process and comfyuikit artifact creation

## Release Workflow

The release workflow (`.github/workflows/release.yaml`) is already configured and will automatically:

1. **Trigger**: When a tag starting with `v*` is pushed (e.g., `v0.1.6`)
2. **Build UI**: Install dependencies and build the Next.js UI project
3. **Create Artifact**: Package the built UI files from `nodes/web/static/` into `comfystream-uikit.zip`
4. **Upload Artifacts**: Upload the zip file to GitHub Actions
5. **Create Release**: Create a GitHub release with:
   - The `comfystream-uikit.zip` artifact attached
   - Auto-generated release notes from commits since the last release
   - Mark as latest release

## Next Steps (To Be Done by Repository Maintainer)

To complete the release creation, a repository maintainer needs to:

### Option 1: Create Release via GitHub UI (Recommended)
1. Go to https://github.com/livepeer/comfystream/releases/new
2. Click "Choose a tag" and type `v0.1.6` (this will create the tag)
3. Set "Target" to the main branch (after this PR is merged)
4. Title: `v0.1.6`
5. Click "Generate release notes" to auto-populate the description
6. Review and edit the release notes if needed
7. Click "Publish release"

### Option 2: Create Release via Git Tag (After PR Merge)
```bash
# After this PR is merged to main
git checkout main
git pull origin main
git tag v0.1.6
git push origin v0.1.6
```

The GitHub Actions workflow will automatically:
- Build the UI
- Create comfystream-uikit.zip
- Attach it to the release
- Generate release notes

## Build Verification

The build process was tested and verified:
- ✅ UI dependencies install successfully (524 packages)
- ✅ Build completes without errors
- ✅ Static files generated in `nodes/web/static/` (32 files)
- ✅ `comfystream-uikit.zip` created successfully (531 KB)
- ✅ Archive contains all expected UI assets

## Installation After Release

Once the release is published, users can install comfystream v0.1.6:

```bash
# Install from release tag
pip install git+https://github.com/livepeer/comfystream.git@v0.1.6

# Or install latest
pip install git+https://github.com/livepeer/comfystream.git
```

The `install.py` script will automatically download the `comfystream-uikit.zip` artifact from the release.
