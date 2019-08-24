# autosynch

[![Build Status](https://travis-ci.com/SwagLyrics/autosynch.svg?branch=master)](https://travis-ci.com/SwagLyrics/autosynch) [![codecov](https://codecov.io/gh/SwagLyrics/autosynch/branch/master/graph/badge.svg)](https://codecov.io/gh/SwagLyrics/autosynch)

**Check out [my blog](https://medium.com/@chriswang030) on my progress and process throughout GSoC 2019!**

## about
Given an audio file of some recognizable song, autosynch will try to align its
lyrics to their temporal location in the song. The song lyrics must be available
on [Genius](https://genius.com).

This project is still in its early stages and is inaccurate in many cases. Optimization is a work in progress, but feel free to try it out, modify it, or contribute!

Developed during Google Summer of Code 2019 with CCExtractor.

## installation
To install, do the following:
```
git clone
cd autosynch
pip install -r requirements.txt
```

*Note: autosynch is supported only on Python 3.6+.*

## dependencies
Using autosynch requires a trained model for vocal isolation as well as
PortAudio. For mp3 support, SoX is required. On MacOS/Linux, get everything by
executing:
```
chmod a+x setup.sh
./setup.sh
```

### download model
If you would like to download the weights manually or get a different version,
check here:
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3334973.svg)](https://doi.org/10.5281/zenodo.3334973) for weights trained on MedleyDB V1
- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3351632.svg)](https://doi.org/10.5281/zenodo.3351632) for weights trained on MedleyDB V2

Weights must be placed into `autosynch/mad_twinnet/outputs/states`.

### install portaudio
On Mac:
```
brew install portaudio
```

On Linux:
```
sudo apt-get update
sudo apt-get install portaudio19-dev
```

### install sox
*Note: Installing SoX is optional and only required for processing mp3 files.*

On Mac:
```
brew install ffmpeg
```

On Linux:
```
sudo apt install ffmpeg
```

## usage
To play a song with its lyrics displayed at its calculated position:
```
python autosynch/playback.py [audio_file.wav] [artist] [song_title]
```
It will take a few minutes to perform the alignment process. To save the
alignment data to eliminate processing time in future plays of the same audio,
add the flag `-s SAVE_DIR`, where `SAVE_DIR` is the directory you want to save
the alignment data.

If you have already generated and saved an alignment data file:
```
python autosynch/playback.py [audio_file.wav] -f [align_file.yml]
```

If you would like to process an mp3 file, see [this section](#install-sox).
Running with an mp3 will automatically generate a wav file in the same directory.

*Note: If you did not use* `setup.sh`*, first make sure you set your Python*
*environment correctly with* `export PYTHONPATH=$PYTHONPATH:./` *from the outer*
`autosynch` *directory.*

## demos
**Bruno Mars - Finesse**

[![Finesse demo](https://img.youtube.com/vi/csBDM14ssts/0.jpg)](https://www.youtube.com/watch?v=csBDM14ssts)

*(https://www.youtube.com/watch?v=csBDM14ssts)*

The last chorus lags behind a bit, but for the most part sections and lines are
nicely aligned.

**Fun. - We Are Young**

[![We Are Young demo](https://img.youtube.com/vi/Z-yTGKd3ji8/0.jpg)](https://www.youtube.com/watch?v=Z-yTGKd3ji8)

*(https://www.youtube.com/watch?v=Z-yTGKd3ji8)*

The instrumental at the beginning throws off the first verse, but everything
catches up in by line 4.

## references
- de Jong, N. and T. Wempe. "[Praat script to detect syllable nuclei and measure speech rate automatically](https://link.springer.com/article/10.3758/BRM.41.2.385)." Behavior Research Methods 41(2), 2009, pp. 385–390.
- Dedina, M. J. and H. C. Nusbaum. "[PRONOUNCE: a program for pronunciation by analogy](https://www.sciencedirect.com/science/article/pii/088523089190017K)." Computer Speech & Language 5(1), 1991, pp. 55-64.
- Drossos, K., S. I. Mimilakis, D. Serdyuk, G. Schuller, T. Virtanen, Y. Bengio. "[MaD TwinNet: Masker-denoiser architecture with twin networks for monaural sound source separation](https://ieeexplore.ieee.org/document/8489565/)." IJCNN 2018.
- Lee, K. and M. Cremer. "[Segmentation-based lyrics-audio alignment using dynamic programming](https://www.semanticscholar.org/paper/Segmentation-Based-Lyrics-Audio-Alignment-using-Lee-Cremer/3a35971affde22fda14bc281ece66adf99474cd9)." ISMIR 2008.
- Marchand, Y. and R. I. Damper. "[A multistrategy approach to improving pronunciation by analogy](https://www.mitpressjournals.org/doi/10.1162/089120100561674)." Computational Linguistics 26(2), 2000, pp. 196-219.
- Marchand, Y. and R. I. Damper. "[Can syllabification improve pronunciation by analogy of English?](https://www.cambridge.org/core/journals/natural-language-engineering/article/can-syllabification-improve-pronunciation-by-analogy-of-english/669E90B388E3C5C591996C1A35F192FE)" Natural Language Engineering 13(1), 2007, pp. 1-24.
- Nieto, O. and J. P. Bello. "[Systematic exploration of computational music structure research](https://www.semanticscholar.org/paper/Systematic-Exploration-of-Computational-Music-Nieto-Bello/e3c130f2cd33036f0ff990b4f388a7709bfac1e2)." ISMIR 2016.
- Sejnowski, T. J. and C. R. Rosenberg. "[Parallel networks that learn to pronounce English text.](https://www.complex-systems.com/abstracts/v01_i01_a10/)" Complex Systems 1(1), 1987, pp. 145–168.
