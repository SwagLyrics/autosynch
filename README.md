# autosynch repository

The `train` branch is used for training new models. Mixtures should be placed in `mad_twinnet_train/dataset/mixtures` and vocals should be placed in `mad_twinnet_train/dataset/sources`. They should be named such that when sorted alphabetically, the files correspond with one another (i.e. the first file sorted alphabetically in `vocals` should be the vocals for the first file sorted alphabetically in `mixtures`). Run `mad_twinnet_train/scripts/training.py` to use.

If you are using the MedleyDB dataset like I am, you can place `MedleyDB_V1.tar.gz` (or the second version if you'd like) into the `resources` directory, then run `extract.py` to extract just the mixtures and the vocal melodies into the `dataset` directory.
