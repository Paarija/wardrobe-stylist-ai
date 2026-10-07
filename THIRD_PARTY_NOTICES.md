# Third-party notices

## SegFormer-B0 clothes checkpoint

The files in `models/atr-segformer/` are copied from [mattmdjaga/segformer_b0_clothes](https://huggingface.co/mattmdjaga/segformer_b0_clothes). The publisher identifies the checkpoint as MIT-licensed. `models/fashionpedia-segformer/` is a local adaptation of that checkpoint, and `models/fashionpedia-ensemble/` combines their predictions. Copyright in the upstream model remains with its original rights holders. The upstream model card does not document every earlier pretraining source, so this notice does not assert a separate license for that lineage.

The upstream model was trained on the ATR human parsing dataset. Fashionpedia images and annotations used for the local adaptation are not included in this repository.

## Garment classifier

`models/garment_classifier.keras` contains a frozen MobileNetV2 backbone initialized with Keras ImageNet weights and a head trained in this project. The source is [Keras Applications](https://keras.io/api/applications/mobilenet/); its implementation is distributed through TensorFlow/Keras. Training images came from [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small), whose publisher labels the dataset MIT. No training images are included here. The ImageNet initialization is identified separately because the project's MIT license does not establish rights to third-party weights or images.

The project's `LICENSE` covers original project code and does not replace upstream notices or terms.
