********** Lead Instrument / Accompaniment Separation **********

The Python scripts provided in this archive implement the SIMM (Smoothed
Instantaneous Mixture Model), as published in:
    [Durrieu2011]
    J.-L. Durrieu, B. David and G. Richard
    "A musically motivated mid-level representation for pitch estimation
    and musical audio source separation", 
    IEEE Journal on Selected Topics on Signal Processing
    First submission on September 2010, 
    Publication scheduled on October 2011.
and:
    [Durrieu2010]
    J.-L. Durrieu,
    "Automatic Extraction of the Main Melody from Polyphonic Music Signals",
    EDITE
    Institut TELECOM, TELECOM ParisTech, CNRS LTCI
    May 2010

The scripts are:
    imageMatlab.py
        This is more or less a wrapper for Matplotlib imaging functions
        such that their behaviour is equivalent, in terms of colormap,
        aspect and so forth, to the expected behaviour of Matlab's 
        functions.
    separateLeadStereoParam.py
        This script can be used to execute the desired separation. See
        below for an example of use of this file.
    SIMM.py
        This script implements the actual algorithm for parameter 
        estimation. It is mainly used by separateLead.py.
    tracking.py
        The Viterbi decoding algorithm is implemented in this script.

Requirements:
    These scripts have been tested with Python 2.6, 
    
    The packages that are required to run the scripts are Numpy, 
    Scipy, Matplotlib. One can respectively find the latest versions
    at the following addresses:
        http://numpy.scipy.org/
        http://scipy.org/
        http://matplotlib.sourceforge.net/

    Notes:
        Prefer recent versions of the above packages, in order to avoid
        compatibility issues, notably for Matplotlib. Note that this 
        latter package is not necessary for the program to run, although
        you might want to watch a bit what is happening!
        Scipy sghould be version 0.8+, since we use its io.wavefile module
        to read the wavefiles. We once used the audiolab module, but it would
        seem that it is a bit more complicated to install (with the benefit
        that many more file formats are allowed).

Usage:
    The easy way to use these scripts is to run the following line:
        python separateLeadStereoParam.py path/to/audio/file.wav
    
    Only WAV file format is recognized for the moment.
    
    The default action is to process the given file, and write five
    result files: 
        path/to/audio/file_acc.wav
            Audio file of the estimated accompaniment part.
        path/to/audio/file_lead.wav
            Audio file of the estimated lead instrument part (the 
            so-called "solo").
        path/to/audio/file_acc_VUIMM.wav
            Audio file of the estimated accompaniment part, with
            unvoiced lead parts estimation.
        path/to/audio/file_lead_VUIMM.wav
            Audio file of the estimated lead instrument part (the 
            so-called "solo"), with unvoiced lead parts estimation.
        path/to/audio/file_pitches.txt
            TXT file containing, in the first line, the time labels
            of each analysis window (in s) and, in the second line,
            the sequence of pitches of the lead voice (in Hz).
    
    One can also add some options to the above command line, so as to 
    use different configurations of parameters. These options are given 
    by the following command:
        python separateLeadStereoParam.py -h

Disclaimer:
    Although these scripts implement the techniques and algorithms 
    proposed in some previous articles, one may expect some differences between
    results we obtained with our original Matlab implementation and results 
    obtained by the present scripts. These results depend on many 
    parameter "tunings", and, for some reason, the adopted numerical
    precision might also lead to differences in the results.
    
    I originally programmed these scripts as an example that it was possible
    to proceed to some advanced signal processing experiment using
    Python instead of Matlab. For the JSTSP article, all reported results
    were however obtained with the provided code.

Contact information:
    e-mail: jean -DOT- louis -DOT- durrieu -AT- gmail -DOT- com