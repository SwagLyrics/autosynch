import os
import parselmouth as pm

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
resourcesdir = os.path.join(parentdir, 'resources')

class SyllableExtractor(object):
    def __init__(self, silencedb=-25, mindip=2, minpause=0.3, showtext=2):
        self.silencedb = silencedb
        self.mindip = mindip
        self.minpause = minpause
        self.showtext = showtext
        self.supported = ['.wav', '.aiff', '.aifc', '.au', '.mp3']

    def run(self, fpath):
        fext = os.path.splitext(fpath)[-1]
        if fext not in self.supported:
            print('{} extension not supported'.format(fext))

        with open(os.path.join(resourcesdir,'praat-script.txt'), 'r') as f:
            script = f.read()

        script = script.format(self.silencedb, self.mindip, self.minpause,
                               self.showtext, fpath)
        pm.praat.run(script)
