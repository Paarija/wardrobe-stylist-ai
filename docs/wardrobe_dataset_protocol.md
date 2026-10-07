# Wardrobe-photo evaluation protocol

The app needs a target-domain dataset of ordinary wardrobe photos. Product-image validation and Fashionpedia crops do not measure that use case. No representative public wardrobe-photo result is claimed yet.

## Labels and review

The single-item classifier uses four labels:

- `top`: all upper-body garments, including jackets and waistcoats
- `bottom`: trousers, jeans, shorts, and skirts
- `shoes`: a single shoe or pair of shoes
- `dress`: one-piece garments that replace both a top and bottom

Every row must be reviewed by a person. Automatically predicted labels do not qualify. Record the image source and confirm consent; private photos remain outside Git.

## Collection design

Collect several garments per class across plain and cluttered backgrounds, daylight and indoor lighting, front and oblique angles, hanging and laid-flat presentations, and different phone cameras. Assign stable `garment_id` and `capture_session` values.

Keep every photo of one garment and every photo from one capture session in a single split. This prevents nearly identical views from leaking across training, development, and final test data. Use development data for model and confidence-threshold decisions. Open the test split once after those decisions are fixed.

Start from [the manifest example](../examples/wardrobe_dataset_manifest.csv), replace its example rows, and validate it:

```shell
python validate_wardrobe_dataset.py data/wardrobe_dataset/manifest.csv --output reports/wardrobe_dataset_validation.json
```

The validator requires reviewed labels and consent, checks exact duplicate hashes, and rejects garment or capture-session overlap across splits. A later dataset version should add perceptual duplicate review and record a version identifier in its validation report.

## Minimum reporting

Report split and class counts, provenance, precision, recall, macro F1, confusion matrices, confidence calibration, representative errors, and inference time. Keep participant photos private unless their consent explicitly covers publication. The current ten-photo local check is useful for debugging but is too small and narrow to satisfy this protocol.
