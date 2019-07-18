import re
from num2words import num2words

regex = re.compile("[^a-zA-Z\s']+")

def format(word):
    if word == '[?]':
        return ''

    try: # word is numerical
        float(word)
        return num2words(word).replace('-', '')
    except:
        pass

    formatted_word = ''
    for ch in word: # digit/letter hybrids
        if ch.isdigit():
            formatted_word += num2words(ch)
        else:
            formatted_word += ch
    return regex.sub('', formatted_word).lower() # lowercase, no punctuation

def stripper(lyrics):
    formatted_lyrics = []
    segment = []

    lines = lyrics.splitlines()
    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            if segment:
                formatted_lyrics.append(segment[:])
                segment.clear()
        elif line:
            line = line.replace('-', ' ').replace('—', ' ').replace('/', ' ')
            segment.append([format(word) for word in line.split()])
    formatted_lyrics.append(segment)

    return formatted_lyrics
