import pyaudio
import wave
import sys
import yaml

CHUNK = 1024

if len(sys.argv) < 2:
    print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')

p = pyaudio.PyAudio()

fr = wf.getframerate()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=fr,
                output=True)

file = '/Users/Chris/autosynch/resources/align_tests/BrunoMars_Finesse.wav'
b_alg = 'olda'
l_alg = 'cnmf'
song = 'Finesse'
artist = 'Bruno Mars'

with open('/Users/Chris/autosynch/resources/outputs_2/BrunoMars_Finesse.yml', 'r') as f:
    align = yaml.safe_load(f)['align']
with open('/Users/Chris/autosynch/resources/tagged/BrunoMars_Finesse_tagged.yml', 'r') as f:
    tagged = yaml.safe_load(f)['align']

data = wf.readframes(CHUNK)

n_frames = 0

i_align = 0
j_align = 0

while data != '':
    n_frames += CHUNK
    sec = n_frames/fr
    if i_align >= len(align) or sec > align[i_align]['end']:
        i_align += 1
        print(end='\t\t')
    else:
        print(align[i_align]['label'], end='\t\t')
    if j_align >= len(tagged) or sec > tagged[j_align]['end']:
        j_align += 1
        print()
    else:
        print(align[j_align]['label'])

    stream.write(data)
    data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()
