# Training and evaluation

This page records the experiments behind the shipped checkpoints. Dataset files belong under data/training/ and are ignored by Git. Commands use paths relative to the repository root. Install the extra readers first:

~~~shell
python -m pip install -r requirements-training.txt
~~~

For the exact direct dependency versions used for the recorded local runs, install `requirements-lock.txt` instead. Run `python scripts/check_models.py` to load both shipped checkpoints and execute real inference.

## Garment classifier

Download the [Fashion Product Images (Small) parquet](https://huggingface.co/datasets/Transformersx/fashion-product-images-small/tree/main/data) to data/training/fashion-products.parquet. The source dataset is described by its [publisher](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small). For single-item uploads, the training script maps all upper-body garments—including sweaters, waistcoats, jackets, and blazers—to **top**. Outfit segmentation retains a separate **outerwear** category for optional layers.

~~~shell
python train_classifier.py --dataset data/training/fashion-products.parquet --output models
~~~

The shipped model uses a frozen ImageNet-initialized MobileNetV2 backbone and trains a 128-unit four-class head. The saved [training report](../models/garment_metrics.json) records 806 training and 202 validation product images, with 93.07% validation accuracy. That validation split also controlled early stopping.

For a separate natural-photo check, download the [Fashionpedia validation parquet](https://huggingface.co/datasets/detection-datasets/fashionpedia/tree/main/data) and [original instance annotations](https://github.com/cvdfoundation/fashionpedia) as data/training/fashionpedia-val.parquet and data/training/fashionpedia-val-annotations.json:

~~~shell
python evaluate_classifier.py --fashionpedia-images data/training/fashionpedia-val.parquet --fashionpedia-annotations data/training/fashionpedia-val-annotations.json --per-class 40 --seed 51 --output reports/classifier_fashionpedia_independent.json
~~~

The [independent report](../reports/classifier_fashionpedia_independent.json) records 160 distinct photos: 53.75% accuracy and 0.5262 macro F1 for crops retaining context, and 53.12% accuracy and 0.5118 macro F1 for mask-isolated garments. It also records per-class precision and recall, confusion matrices, calibration, uncertainty, inference time, and indexed high-confidence mistakes. Fashionpedia outerwear annotations count as tops for this classifier evaluation. These are not the user's own clothing photos.

To evaluate personal clothing photos, create data/personal_test/labels.csv from the [template](../examples/personal_labels_template.csv), put photos in the same folder, and list one image and its true label per row:

~~~csv
image,label
shirt1.jpg,top
jacket1.jpg,top
~~~

Use only top, bottom, shoes, or dress. Keep these photos out of training and account adaptation. Then run:

~~~shell
python evaluate_classifier.py --manifest data/personal_test/labels.csv --output reports/classifier_personal.json
~~~

The personal folder and report are ignored by Git. For a representative collection, follow the [wardrobe-photo protocol](wardrobe_dataset_protocol.md) and validate its manifest with `validate_wardrobe_dataset.py`.

Personal adaptation uses only items explicitly saved through **Edit details**, which marks their labels as human reviewed. It also requires a separate held-out manifest. The app activates the adapted checkpoint only when held-out macro F1 exceeds the global model:

~~~shell
python adapt_classifier.py --user-id 1 --dataset data/training/fashion-products.parquet --evaluation-manifest data/adaptation_test/labels.csv
~~~

## Colour calibration

Colour detection removes edge-connected backgrounds, clusters foreground pixels in CIE LAB space, and can learn a user's corrected colour signatures. The [personal tuning report](../reports/color_calibration_tuning.json) selected a strict LAB distance of 3 using leave-one-item-out evaluation over 20 named wardrobe items. The result was 14/20 (70%); the set is too small and personal for a public accuracy claim. Reproduce it against a local account with:

~~~shell
python tune_color_calibration.py --user-id 1 --output reports/color_calibration_tuning.json
~~~

## Outfit segmentation

The included ATR checkpoint is copied from [mattmdjaga/segformer_b0_clothes](https://huggingface.co/mattmdjaga/segformer_b0_clothes). It predicts top, bottom, shoes, and dress after the app's label mapping; it has no separate outerwear class.

The local six-class checkpoint in models/fashionpedia-segformer/ was trained from that ATR checkpoint on the official Fashionpedia training split. To repeat the run, download its training image parquet and original instance annotations as data/training/fashionpedia-train-00000.parquet and data/training/fashionpedia-train-annotations.json, then run:

~~~shell
python train_fashionpedia.py --images data/training/fashionpedia-train-00000.parquet --annotations data/training/fashionpedia-train-annotations.json --base models/atr-segformer --output models/fashionpedia-segformer
~~~

The [training report](../models/fashionpedia-segformer/training_report.json) records 640 training images, 128 validation images, three CPU epochs, and the label weights. The standalone model's top IoU is near zero on the separate Fashionpedia holdout. It is **not used alone** in the app. The selected ensemble keeps the ATR mask and changes a top pixel to outerwear when the local model's outerwear probability reaches 0.62. This rule is in models/fashionpedia-ensemble/ensemble.json.

The [threshold development report](../reports/fashionpedia_threshold_refined.json) records the 100 image IDs and all candidate thresholds. To recompute it on those same IDs, repeat the --outerwear-threshold option for each value:

~~~shell
python evaluate_fashionpedia.py --images data/training/fashionpedia-val.parquet --annotations data/training/fashionpedia-val-annotations.json --include-report reports/fashionpedia_threshold_refined.json --checkpoint models/fashionpedia-segformer --base models/atr-segformer --outerwear-threshold 0.52 --outerwear-threshold 0.54 --outerwear-threshold 0.56 --outerwear-threshold 0.58 --outerwear-threshold 0.60 --outerwear-threshold 0.62 --outerwear-threshold 0.64 --outerwear-threshold 0.66 --outerwear-threshold 0.68 --output reports/recomputed_thresholds.json
~~~

For a newly trained checkpoint, review the recomputed scores, then pass that report explicitly to the selector. It rejects reports whose hashes do not match the exact base and Fashionpedia weights:

~~~shell
python select_fashionpedia_ensemble.py --report reports/recomputed_thresholds.json
~~~

The bundled [selection summary](../reports/fashionpedia_accuracy_summary.json) records the current component hashes and selected threshold.

The [150-photo holdout report](../reports/fashionpedia_accuracy_holdout.json) records image IDs that can be reused with --include-report. On this sample, five-class mean IoU was 0.4481 for ATR, 0.4396 for the standalone local checkpoint, and 0.5065 for the selected ensemble. Standalone top IoU was 0.0002; the ensemble's top IoU was 0.3279. To rerun on the recorded IDs:

~~~shell
python evaluate_fashionpedia.py --images data/training/fashionpedia-val.parquet --annotations data/training/fashionpedia-val-annotations.json --include-report reports/fashionpedia_accuracy_holdout.json --checkpoint models/fashionpedia-segformer --base models/atr-segformer --outerwear-threshold 0.62 --output reports/recomputed_holdout.json
~~~

Earlier ATR fine-tuning and hyperparameter comparisons are recorded in [validation](../reports/segmentation_tuning_validation.json), [holdout](../reports/segmentation_tuning_holdout.json), and [selection](../reports/segmentation_tuning_summary.json) reports. `python tune_segmentation.py` now reuses IDs from the committed validation report and has no dependency on removed `models/outfit-segformer` artifacts. Horizontal augmentation swaps every ATR left/right shoe, leg, and arm label. The locally fine-tuned candidates were not selected. See the [model analysis](model_analysis.md) for the ensemble's top-versus-outerwear tradeoff.

## Recommendation ratings

Occasion ranking uses recorded Pinterest inspiration cues: casual denim and tees for everyday/college, structured separates and neutral footwear for office, and dresses, floral pieces, statement colours, and dark footwear for parties. The cues are manually encoded and versioned rather than scraped live. Each occasion links to its Pinterest search, while the current colour trends come from the official [Pinterest Predicts 2026 report](https://business.pinterest.com/pdf/pinterest-predicts/2026-trend-report/).

The app's **Rate outfits** page asks for 1–5 ratings in a hidden order. When the wardrobe permits, it shows two rule-ranked outfits, two colour-only baseline outfits, and two random valid outfits without revealing their source. After people rate them, run:

~~~shell
python evaluate_recommendations.py --output reports/recommendation_study.json
~~~

The report records participant count, candidate-pool size, rating variation, per-method mean ratings with descriptive intervals, and pairwise rules-versus-baseline preferences. No multi-participant result has been collected yet, so no recommendation-quality score is claimed. Local ratings stay in SQLite; the generated report is ignored by Git.
