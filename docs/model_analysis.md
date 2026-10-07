# Model analysis and decisions

## Garment classification

The shipped classifier keeps an ImageNet-pretrained MobileNetV2 backbone frozen and trains a 128-unit head for `top`, `bottom`, `shoes`, and `dress`. All upper-body garments, including jackets and waistcoats, map to `top`. This project implements the label mapping, sampling, head training, independent evaluation, personal adaptation gate, and app integration; it does not claim authorship of MobileNetV2 or its ImageNet weights.

The product-image validation result is 93.07%, but the independent Fashionpedia result is much lower:

| Evaluation | Accuracy | Macro F1 | ECE | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Context crops, 160 photos | 0.5375 | 0.5262 | 0.2508 | 0.6655 |
| Isolated crops, 160 photos | 0.5312 | 0.5118 | 0.2655 | 0.7619 |

The context-crop class F1 scores are top 0.4762, bottom 0.5983, shoes 0.5185, and dress 0.5117. High-confidence errors remain: examples include shoes and dresses predicted as bottoms above 99% confidence. Confidence is displayed as information rather than used as an automatic acceptance rule. The full [classifier report](../reports/classifier_fashionpedia_independent.json) includes precision, recall, F1, confusion matrices, calibration bins, uncertainty, inference timing, and indexed high-confidence errors.

The ten-photo local wardrobe check reaches 8/10 after the four-class retraining, but it is private, small, and not representative. It is excluded from public performance claims. The [wardrobe dataset protocol](wardrobe_dataset_protocol.md) defines the evidence still needed.

## Outfit segmentation

The base parser is the pretrained ATR SegFormer-B0 checkpoint. A locally trained six-class Fashionpedia checkpoint was intended to add outerwear, but its standalone top IoU is 0.0002 on the 150-image holdout. The final rule keeps ATR predictions and changes only high-confidence ATR top pixels to outerwear.

| Model | Five-class mIoU | Top IoU | Outerwear IoU |
| --- | ---: | ---: | ---: |
| ATR base | 0.4481 | 0.3632 | 0.0000 |
| Local Fashionpedia model | 0.4396 | 0.0002 | 0.3629 |
| Selected 0.62 ensemble | 0.5065 | 0.3279 | 0.3276 |

The ensemble improves the five-class mean by adding outerwear, while top IoU decreases by 0.0353. This is a measured tradeoff rather than evidence that the standalone model is generally better. The likely top/outerwear competition needs a new training experiment; it has not been resolved.

Checkpoint selection verifies that the evaluation report hashes match both exact weight files. Invalid optional configuration falls back to verified ATR. The shipped [holdout report](../reports/fashionpedia_accuracy_holdout.json) records fixed IDs, hashes, and per-class results.

On the current CPU environment, a cold start plus one blank-image inference took about 19.8 seconds for classification and 22.7 seconds for the two-model segmentation ensemble. These single-run timings include model loading and are operational smoke measurements, not benchmark estimates. Garment-crop usefulness still needs human review on consented outfit photos.

## Recommendation evaluation

Recommendations are deterministic scoring rules. The rating page now draws blinded candidates from the app rules, a colour-only baseline, and random valid combinations. It records the full candidate-pool size and uses two candidates per method when available. Reports include participant count, rating variation, method means with descriptive intervals, and pairwise rules-versus-baseline preferences.

No participant result is claimed yet. A useful pilot requires multiple people to rate the same balanced procedure; one account cannot establish general usefulness.

## Reproduction boundary

Exact direct dependency versions are in [requirements-lock.txt](../requirements-lock.txt). Dataset files are excluded because of size, provenance, and privacy. Split IDs, preprocessing, seeds, hyperparameters, hashes, and output reports are retained where redistribution permits. Run `python scripts/check_models.py` for real checkpoint loading and inference.
