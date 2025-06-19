# Publishing Coral to PyPI

This guide covers how to build and publish the Coral package to PyPI using `uv`.

## Prerequisites

1. **PyPI Account**: Create an account at [pypi.org](https://pypi.org)
2. **API Token**: Generate an API token from your PyPI account settings
3. **uv**: Ensure uv is installed (`pip install uv`)

## Building the Package

Before publishing, build the distribution packages:

```bash
# Build both wheel and source distributions
uv build

# Build without local sources (recommended for publishing)
uv build --no-sources
```

This creates distribution files in the `dist/` directory.

## Publishing to PyPI

### 1. Set Authentication Token

Set your PyPI token as an environment variable:

```bash
export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN_HERE"
```

Alternatively, add to your shell profile (`~/.bash_profile` or `~/.zshrc`):

```bash
echo 'export UV_PUBLISH_TOKEN="pypi-YOUR_TOKEN_HERE"' >> ~/.bash_profile
```

### 2. Publish the Package

```bash
# Publish to PyPI
uv publish

# Or explicitly specify files
uv publish dist/*
```

## Testing with TestPyPI

Before publishing to the main PyPI repository, test with TestPyPI:

### 1. Create TestPyPI Account

Register at [test.pypi.org](https://test.pypi.org)

### 2. Configure TestPyPI in pyproject.toml

Uncomment the test configuration in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "testpypi"
url = "https://test.pypi.org/simple/"
publish-url = "https://test.pypi.org/legacy/"
explicit = true
```

### 3. Set TestPyPI Token

```bash
export UV_PUBLISH_TOKEN="pypi-YOUR_TEST_TOKEN_HERE"
```

### 4. Publish to TestPyPI

```bash
# Publish to TestPyPI
uv publish --index testpypi
```

### 5. Test Installation

```bash
# Install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ coral
```

## Version Management

Before publishing a new version:

1. Update version in `pyproject.toml`
2. Update CHANGELOG (if maintained)
3. Commit changes
4. Tag the release:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

## Automated Publishing (GitHub Actions)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    
    - name: Build package
      run: uv build --no-sources
    
    - name: Publish to PyPI
      env:
        UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
      run: uv publish
```

Add your PyPI token to GitHub secrets as `PYPI_TOKEN`.

## Troubleshooting

### Authentication Issues

If you get authentication errors:
- Verify token is correctly set: `echo $UV_PUBLISH_TOKEN`
- Ensure token has appropriate scope
- Check token hasn't expired

### Package Name Conflicts

If the package name is taken:
- Choose a different name in `pyproject.toml`
- Consider namespacing (e.g., `coral-ml`, `coral-weights`)

### Build Issues

If build fails:
- Run `uv build` locally to check for errors
- Ensure all files are included via `MANIFEST.in` if needed
- Verify `pyproject.toml` syntax

## Post-Publishing

After successful publishing:

1. Verify installation: `uv pip install coral`
2. Test basic functionality
3. Update documentation with installation instructions
4. Announce release (GitHub releases, social media, etc.)

## Security Notes

- **Never commit tokens** to version control
- Use environment variables or CI/CD secrets
- Rotate tokens periodically
- Use 2FA on PyPI account