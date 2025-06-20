# Coral CLI Safetensors Commands

Coral v1.0.0 introduces three new CLI commands for working with the popular Safetensors format, enabling seamless interoperability between Coral and other ML tools.

## Commands Overview

### 1. `coral import-safetensors`

Import weights from a Safetensors file into your Coral repository.

```bash
coral import-safetensors <file> [options]
```

**Arguments:**
- `file`: Path to the Safetensors file to import

**Options:**
- `--weights WEIGHT [WEIGHT ...]`: Specific weights to import (default: all)
- `--exclude WEIGHT [WEIGHT ...]`: Weights to exclude from import
- `--no-metadata`: Don't preserve Safetensors metadata

**Example:**
```bash
# Import all weights from a model
coral import-safetensors model.safetensors

# Import only specific layers
coral import-safetensors model.safetensors --weights encoder.weight encoder.bias

# Import all except output layers
coral import-safetensors model.safetensors --exclude classifier.weight classifier.bias
```

### 2. `coral export-safetensors`

Export weights from your Coral repository to a Safetensors file.

```bash
coral export-safetensors <output> [options]
```

**Arguments:**
- `output`: Output file path (`.safetensors` extension added if missing)

**Options:**
- `--weights WEIGHT [WEIGHT ...]`: Specific weights to export (default: all)
- `--no-metadata`: Don't include Coral metadata in the output
- `--metadata KEY=VALUE`: Add custom metadata (can be used multiple times)

**Example:**
```bash
# Export all weights in the current branch
coral export-safetensors trained_model.safetensors

# Export specific weights with custom metadata
coral export-safetensors checkpoint.safetensors \
  --weights encoder decoder \
  --metadata author="Your Name" \
  --metadata training_steps=10000
```

### 3. `coral convert`

Convert between different weight file formats. Automatically detects format based on file extensions.

```bash
coral convert <input> <output> [options]
```

**Arguments:**
- `input`: Input file path
- `output`: Output file path

**Options:**
- `--weights WEIGHT [WEIGHT ...]`: Specific weights to convert (default: all)
- `--no-metadata`: Don't preserve/include metadata

**Supported Conversions:**
- Safetensors → NPZ, HDF5
- NPZ → Safetensors
- Coral repository → Safetensors

**Example:**
```bash
# Convert Safetensors to NPZ
coral convert model.safetensors model.npz

# Convert NPZ to Safetensors
coral convert weights.npz weights.safetensors

# Export entire Coral repository as Safetensors
coral convert ./my-coral-repo exported_repo.safetensors

# Convert to HDF5 format
coral convert model.safetensors model.h5
```

## Integration with Coral Features

### Automatic Deduplication

When importing Safetensors files, Coral automatically:
- Detects duplicate weights across imports
- Applies delta encoding for similar weights
- Shows deduplication statistics after import

```bash
$ coral import-safetensors large_model.safetensors
Importing weights from large_model.safetensors...
✓ Successfully imported 1024 weight(s)
✓ Deduplicated 512 weight(s) (50.0% reduction)
```

### Metadata Preservation

Coral preserves and enhances metadata during conversions:

**On Import:**
- Original Safetensors metadata is preserved
- Metadata fields are prefixed with `safetensors.` if not standard
- Model name, version, and description are mapped to Coral fields

**On Export:**
- Coral branch and commit information is included
- Similarity threshold and delta encoding status are recorded
- Custom metadata can be added via `--metadata` flags

### Version Control Integration

Imported weights are automatically committed to your Coral repository:

```bash
$ coral import-safetensors model_v1.safetensors
$ coral log --oneline
abc12345 Import 256 weights from model_v1.safetensors

$ coral import-safetensors model_v2.safetensors
$ coral log --oneline
def67890 Import 256 weights from model_v2.safetensors
abc12345 Import 256 weights from model_v1.safetensors
```

## Common Workflows

### 1. Migrating from Safetensors to Coral

```bash
# Initialize a new Coral repository
coral init my-model-repo
cd my-model-repo

# Import your Safetensors models
coral import-safetensors ../models/base_model.safetensors
coral import-safetensors ../models/finetuned_v1.safetensors
coral import-safetensors ../models/finetuned_v2.safetensors

# View deduplication results
coral status
```

### 2. Exporting for Model Sharing

```bash
# Create a branch for release
coral branch release-v1.0
coral checkout release-v1.0

# Export with metadata
coral export-safetensors release_v1.0.safetensors \
  --metadata version="1.0.0" \
  --metadata release_date="2024-01-15" \
  --metadata license="MIT"
```

### 3. Batch Conversion

```bash
# Convert multiple Safetensors files to NPZ
for file in *.safetensors; do
  coral convert "$file" "${file%.safetensors}.npz"
done
```

### 4. Selective Weight Management

```bash
# Import only encoder weights from a large model
coral import-safetensors large_model.safetensors \
  --weights encoder.* \
  --exclude encoder.embeddings.*

# Export only decoder weights
coral export-safetensors decoder_only.safetensors \
  --weights decoder.*
```

## Performance Tips

1. **Batch Operations**: Import multiple related models together to maximize deduplication benefits

2. **Filter Early**: Use `--weights` and `--exclude` during import to avoid storing unnecessary weights

3. **Compression**: Coral's HDF5 backend provides additional compression beyond Safetensors

4. **Progress Monitoring**: All commands show progress bars for large operations

## Error Handling

The commands provide helpful error messages:

```bash
$ coral import-safetensors missing.safetensors
Error: File not found: missing.safetensors

$ coral export-safetensors output.st --metadata invalid
Error: Invalid metadata format: invalid (expected key=value)

$ coral convert file.txt output.safetensors
Error: Unsupported conversion from .txt to .safetensors
Supported conversions:
  - .safetensors → .npz, .h5, .hdf5
  - .npz → .safetensors
  - Coral repository → .safetensors
```