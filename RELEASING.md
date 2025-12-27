# Releasing Coral

This document describes how to release a new version of Coral.

## Release Checklist

Before releasing, ensure:

- [ ] All tests pass: `uv run pytest`
- [ ] Code is formatted: `uv run ruff format src/ tests/`
- [ ] Linting passes: `uv run ruff check src/ tests/`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] Coverage is above 80%: `uv run pytest --cov=coral --cov-fail-under=80`
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated

## Version Numbering

Coral follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Pre-release versions use suffixes:
- Alpha: `1.0.0-alpha.1`
- Beta: `1.0.0-beta.1`
- Release Candidate: `1.0.0-rc.1`

## Release Process

### 1. Update Version Numbers

Update the version in **two places**:

```bash
# pyproject.toml
version = "X.Y.Z"

# src/coral/__init__.py
__version__ = "X.Y.Z"
```

Or use the provided script:

```bash
./scripts/bump_version.py X.Y.Z
```

### 2. Update CHANGELOG.md

Move items from `[Unreleased]` to the new version section:

```markdown
## [Unreleased]

## [X.Y.Z] - YYYY-MM-DD

### Added
- New features...

### Changed
- Changes in existing functionality...

### Fixed
- Bug fixes...

### Deprecated
- Soon-to-be removed features...

### Removed
- Removed features...

### Security
- Security fixes...
```

Update the links at the bottom:

```markdown
[Unreleased]: https://github.com/parkerdgabel/coral/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/parkerdgabel/coral/releases/tag/vX.Y.Z
```

### 3. Commit the Release

```bash
git add pyproject.toml src/coral/__init__.py CHANGELOG.md
git commit -m "chore: release version X.Y.Z"
```

### 4. Create and Push Tag

```bash
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin main
git push origin vX.Y.Z
```

### 5. Automated Release

Once the tag is pushed, GitHub Actions will automatically:

1. **Validate** version consistency across files
2. **Run tests** on Python 3.9, 3.11, and 3.12
3. **Build** the distribution packages
4. **Publish to TestPyPI** (for verification)
5. **Publish to PyPI** (requires approval)
6. **Create GitHub Release** with changelog notes

### 6. Verify the Release

After the workflow completes:

```bash
# Verify PyPI installation
pip install coral-ml==X.Y.Z

# Quick smoke test
python -c "import coral; print(coral.__version__)"
```

## Manual Release (If Needed)

If automated release fails, you can release manually:

```bash
# Build
uv build --no-sources

# Publish to TestPyPI first
UV_PUBLISH_TOKEN=$TEST_PYPI_TOKEN uv publish \
  --publish-url https://test.pypi.org/legacy/ dist/*

# Verify on TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ coral-ml==X.Y.Z

# Publish to PyPI
UV_PUBLISH_TOKEN=$PYPI_TOKEN uv publish dist/*
```

## Hotfix Releases

For urgent fixes to a released version:

```bash
# Create hotfix branch from tag
git checkout -b hotfix/X.Y.Z vX.Y.Z

# Make fixes...
# Update version to X.Y.(Z+1)
# Commit and tag
git commit -m "fix: critical bug description"
git tag -a vX.Y.(Z+1) -m "Hotfix release X.Y.(Z+1)"

# Push
git push origin hotfix/X.Y.Z
git push origin vX.Y.(Z+1)

# Merge back to main
git checkout main
git merge hotfix/X.Y.Z
git push origin main
```

## GitHub Secrets Configuration

The following secrets must be configured in GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `PYPI_TOKEN` | PyPI API token for publishing |
| `TEST_PYPI_TOKEN` | TestPyPI API token for testing |
| `CODECOV_TOKEN` | Codecov token for coverage reports (optional) |

### Creating PyPI Tokens

1. Go to [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
2. Create a token scoped to the `coral-ml` project
3. Add as `PYPI_TOKEN` in GitHub Secrets

4. Go to [test.pypi.org/manage/account/token/](https://test.pypi.org/manage/account/token/)
5. Create a token scoped to the `coral-ml` project
6. Add as `TEST_PYPI_TOKEN` in GitHub Secrets

## Environment Protection Rules

Configure environment protection in GitHub:

### `testpypi` Environment
- No required reviewers
- Allow all branches with tags

### `pypi` Environment
- Required reviewers: 1 (recommended for production releases)
- Allow only `v*.*.*` tags

## Troubleshooting

### Version Mismatch Error

If the release fails due to version mismatch:

```bash
# Check versions
grep 'version' pyproject.toml
grep '__version__' src/coral/__init__.py
git describe --tags

# Fix and re-tag
git tag -d vX.Y.Z
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin :vX.Y.Z
git push origin vX.Y.Z
```

### PyPI Upload Failure

If PyPI upload fails:

1. Check the token hasn't expired
2. Verify the version doesn't already exist on PyPI
3. Ensure package name is available

### Rollback a Release

If a release needs to be yanked:

```bash
# On PyPI (cannot delete, only yank)
# Go to pypi.org/manage/project/coral-ml/releases/
# Click on the version and select "Yank"

# Delete GitHub release
gh release delete vX.Y.Z

# Delete tag (optional, keeps history)
git tag -d vX.Y.Z
git push origin :vX.Y.Z
```

## Post-Release

After a successful release:

1. **Announce** the release on relevant channels
2. **Update** any pinned versions in examples/documentation
3. **Close** related GitHub milestones
4. **Create** a new milestone for the next release
