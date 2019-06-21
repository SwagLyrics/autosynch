# autosynch repository

## installation
To install, do the following:
- `git clone` this branch (use `-b phase1 --single-branch` to clone just this branch)
- `cd autosynch`
- `pip install -r requirements.txt`

## usage
To use, you must put a trained model into the `src/mad_twinnet/outputs/states`
directory. The files of the model must be exactly:
- rnn_enc.pt
- rnn_dec.pt
- fnn.pt
- denoiser.pt

To download the pre-trained model from the authors of MaD TwinNet, go [here](https://doi.org/10.5281/zenodo.1164592).
A new model based on more popular music is in the works and is coming soon.

After the model is installed, the two main functions are get_vocal_syllables()
and eval_by_syllable() in `src/eval.py`. Run `src/eval.py` as `__main__` to get
an example run on Liz Nelson's *Rainfall* and Marvin Gaye's *I Heard It Through the
Grapevine*.
