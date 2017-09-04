## Learning CNN-LSTM Architectures for Image Caption Generation

This code contains a Tensorflow implementation of the CNN-LSTM architecture used to attain state-of-the-art performance on the MSCOCO dataset. We achieve a BLEU-4 score of 24.4 and CIDEr score of 81.7 compared to 27.7 and 85.5 by Google's implementation. Qualitative analysis of the generated captions indicate that the model is able to sensibly caption a wide variety of images from the MSCOCO dataset.

### Other files
`model.py` contains the `Model` class that contains the CNN-LSTM architecture (using Tensorflow's dynamic_rnn API) and various helper functions for generating captions. `evaluate_captions.py` is a helper script to generate aggregated JSON files that can then be used for hyperparameter tuning. `image_feature_cnn.py` contains the helper functions we use to load up the GoogleNet batch normalization CNN model and turn images into 1024 x 1 vectors.
