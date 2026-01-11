---
title: Concordance DAT/OPT Import Patterns
tags:
  - docviewer
  - concordance
  - legal-discovery
  - data-import
  - dat-opt
  - s3
created: 2025-12-22
contributors:
  - chriskd <2326567+chriskd@users.noreply.github.com>
source_project: memex
---

# Concordance DAT/OPT Import Patterns

Legal discovery productions often use the Concordance format. Here's what we learned importing DOJ EFTA datasets.

## File Format Details

### DAT Files (Document Metadata)
- **Encoding**: UTF-8 with BOM (`utf-8-sig`)
- **Delimiter**: `\x14` (DC4 control character)
- **Quote Character**: `þ` (thorn, 0xFE)
- **Line Endings**: CRLF (Windows)

### OPT Files (Image-to-Page Mappings)
- Standard CSV format
- Maps bates numbers to image file paths
- Columns: bates_number, volume, path, is_first_page, unused, unused, pages_count

## Common Header Variations

Different productions use different header names:

| Expected | Variation Found |
|----------|-----------------|
| `Bates Begin` | `Begin Bates` |
| `Bates End` | `End Bates` |

**Fix before import:**
```bash
sed '1s/Begin Bates/Bates Begin/; 1s/End Bates/Bates End/' input.DAT > fixed.DAT
```

## Path Transformation for S3

OPT files contain Windows paths like `IMAGES\0001\EFTA00000001.pdf`. For S3 storage, transform to:
```
s3://bucket/dataset-slug/0001/EFTA00000001.pdf
```

Python snippet:
```python
raw_path = row[2]  # IMAGES\0001\file.pdf
normalized = raw_path.replace('\\', '/').replace('IMAGES/', '')
s3_path = f"s3://efta-images/{slug}/{normalized}"
```

## Import Workflow

1. Fix headers if needed (sed)
2. Import DAT with `--skip-opt` (document metadata)
3. Transform OPT paths to S3 URIs
4. Insert image records directly to DB

## Related Issues

- [[epstein-s5g]] - Header variation support
- [[epstein-559]] - S3 path transformation feature