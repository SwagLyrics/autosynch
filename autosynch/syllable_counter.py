import os
import re
from num2words import num2words
from collections import Counter, deque
from operator import itemgetter

from autosynch.config import cmudict_path, nettalk_path

class SyllableCounter(object):
    """ Helper class for counting number of syllables in lyrics.
    """

    def __init__(self, sba_lexicon_path=nettalk_path, cmudict_path=cmudict_path):
        # Regex for alphabetical characters, apostrophes, and whitespaces
        self.regex = re.compile(r"[^a-zA-Z\s']+")

        # Load lexicon, dictionary
        self.lexicon, self.counter = self._load_data(sba_lexicon_path, cmudict_path)

    def _load_data(self, sba_lexicon_path, cmudict_path):
        """
        Loads data for SbA lexicon and for counting dictionary.

        Lexicon file should follow format of NetTalk dataset, with '#' denoting
        comment lines and data lines following the format:

            [word] [any_text] [stress_pattern]

        with stress_pattern as specified in Sejnowski & Rosenberg (1987). CMU
        dict file should have comment lines begin with ';;;'.

        Returns None, None if SbA lexicon cannot be opened.

        :param sba_lexicon_path: Path to SbA lexicon file.
        :type sba_lexicon_path: file-like
        :param cmudict_path: Path to CMU dict file.
        :type cmudict_path: file-like
        :return lexicon, counter: List of hyphenated words for SbA processing\
                and dictionary with values denoting number of syllables in key.
        :rtype: list[str], dict{str: int}
        """

        lexicon = []
        counter = {}

        # Check lexicon file
        try:
            with open(sba_lexicon_path, 'r') as f:
                sba_lexicon = f.read().splitlines()
        except Exception as e:
            print(e)
            return None, None

        # Check CMUdict file
        try:
            with open(cmudict_path, 'r') as f:
                cmudict = f.read().splitlines()
        except Exception as e:
            print('Unable to read CMUdict')
            cmudict = []

        # Format and load lexicon data
        for line in sba_lexicon:
            if not line or line.startswith('#'):
                continue
            line = line.split()
            word = line[0]
            syll = line[2]
            count = 0

            hyphenated_word = ''
            for i, ch in enumerate(word):
                if i < len(word)-1 and syll[i] != '>' and syll[i+1] != '<':
                    hyphenated_word += ch + '-'
                    count += 1
                else:
                    hyphenated_word += ch + '*'

            lexicon.append('#{}#'.format(hyphenated_word[:-1]))
            counter[word] = count + 1

        # Format and save CMUdict data
        for line in cmudict:
            if not line or line.startswith(';;;'):
                continue
            word = line.split(None, 1)[0].lower()
            count = sum(ch.isdigit() for ch in line)

            if word.endswith(')'):
                continue

            if word not in counter or counter[word] < count:
                counter[word] = count

        return lexicon, counter

    def _naive(self, input):
        """
        Naive algorithm for counting syllables in a word based on simplified
        foundational syllabification rules.

        :param input: Word to count syllables.
        :type input: str
        :return n_vowels: Number of syllables.
        :rtype: int
        """

        vowels = 'aeiouy'

        n_vowels = 0
        prev_vowel = False

        for ch in input:
            is_vowel = False

            # Vowel counts as syllable
            if ch in vowels:
                # Check for diphthong
                if not prev_vowel:
                    n_vowels += 1
                is_vowel = True
                prev_vowel = True
            if not is_vowel:
                prev_vowel = False

        # Remove silent vowels
        if input.endswith('es') or input.endswith('e'):
            n_vowels -= 1

        return n_vowels

    def _sba(self, input):
        """
        Implementation of Marchand & Damper's syllabification by analogy
        algorithm (2006). Requires lexicon to be loaded prior to calling.

        Y. Marchand and R. I. Damper. "Can syllabification improve pronunciation
        by analogy of English?" Nat. Lang. Eng. 13(1), 2006, pp. 1-24.

        :param input: Word to count syllables.
        :type input: str
        :param verbose: Flag for printing syllabification failure warnings.
        :type verbose: bool
        :return n_syllables: Number of syllables.
        :rtype: int
        """

        # Node data class
        class data(object):
            def __init__(self):
                self.outputs = Counter() # Set of arcs going out
                self.sinputs = [] # Set of arcs coming in, filled by BFS
                self.distance = float('inf') # From origin, filled by BFS

        # Format input
        input = '#{}#'.format(input.replace('', '*')[1:-1])

        # Pronunciation lattice
        lattice = {('#', 0): data(), ('#', len(input)-1): data() }

        # Substring matcher and lattice builder
        substring = []
        for entry in self.lexicon:
            for offset in range(-len(input)+3, len(entry)-2):
                for i in range(max(0, -offset), min(len(input), len(entry)-offset)):
                    if input[i] == entry[i+offset] or (input[i] == '*' and entry[i+offset] == '-'):
                        substring.append((entry[i+offset], i))
                    else:
                        for i, node in enumerate(substring):
                            string = ''
                            if node not in lattice:
                                lattice[node] = data()
                            for j in range(i+1, len(substring)):
                                string += substring[j][0]
                                arc = (substring[j], string[:-1])
                                lattice[node].outputs[arc] += 1
                        substring.clear()

                for i, node in enumerate(substring):
                    string = ''
                    if node not in lattice:
                        lattice[node] = data()
                    for j in range(i+1, len(substring)):
                        string += substring[j][0]
                        arc = (substring[j], string[:-1])
                        lattice[node].outputs[arc] += 1

        # Decision function 1: get shortest path(s)
        queue = deque([('#', 0)])
        lattice[('#', 0)].distance = 0
        while queue:
            node = queue.popleft()
            for out_arc in lattice[node].outputs:
                adjacent = out_arc[0]
                in_arc = (node, out_arc[1], lattice[node].outputs[out_arc])
                if lattice[node].distance + 1 < lattice[adjacent].distance:
                    queue.append(adjacent)
                    lattice[adjacent].distance = lattice[node].distance + 1
                    lattice[adjacent].sinputs.append(in_arc)
                elif lattice[node].distance + 1 == lattice[adjacent].distance:
                    lattice[adjacent].sinputs.append(in_arc)

        # Decision function 2: score by strategy
        # PF = product, SDPS = standard deviation, WL = weak link
        # Calculate scores
        paths = []
        def dfs(node, path, arcs):
            if node == ('#', 0):
                pf, sdps, wl = 1, 0, float('inf')
                mean = sum(arcs)/len(arcs)
                for arc in arcs:
                    pf *= arc
                    sdps += (arc-mean)**2
                    wl = min(wl, arc)
                sdps /= len(arcs)

                paths.append((path, pf, sdps, wl))
                return
            if not lattice[node].sinputs:
                return

            path = node[0] + path
            for arc in lattice[node].sinputs:
                _arcs = arcs[:]
                _arcs.append(arc[2])
                dfs(arc[0], arc[1]+path, _arcs)

        dfs(('#', len(input)-1), '', [])
        if not paths:
            return None

        # Assign rankings and points
        scores = {path[0]: 0 for path in paths}
        for s in range(1, 4):
            ranking = sorted(paths, key=itemgetter(s), reverse=True)
            rank, cand, cval = len(paths), 0, ranking[0][s]

            for i, path in enumerate(ranking):
                if path[s] < cval:
                    points = rank - (cand-1)/2
                    for t in ranking[i-cand:i]:
                        scores[t[0]] += points
                    rank -= 1
                    cand, cval = 1, path[s]
                else:
                    cand += 1

            points = rank - (cand-1)/2
            for t in ranking[-cand:]:
                scores[t[0]] += points

        # Get shortest path by points
        shortest_path = max(scores.items(), key=itemgetter(1))[0]
        n_syllables = shortest_path.count('-') + 1

        return n_syllables

    def build_lyrics(self, lyrics):
        """
        Constructs segmented lyrics structure by song section, line, and word.

        Returns of list of tuples representing sections, each of which contains
        a list of lists representing lines of lyrics, each of which is a list of
        words in that line. The first element of the tuple is the section's
        category (i.e., chorus, verse, etc.). The second element is the section
        lyrics themselves.

        :param lyrics: Lyrics in format of Genius.com.
        :type lyrics: str
        :return formatted_lyrics: Lyrics in segmented format.
        :rtype: list[tuple(str, list[list[str]])]
        """

        formatted_lyrics = []
        section = []
        section_type = 'default'

        lines = lyrics.splitlines()
        for line in lines:
            # Check for section header
            if line.startswith('[') and line.endswith(']'):
                # Append section to lyrics
                if section:
                    formatted_lyrics.append((section_type, section[:]))
                    section.clear()

                # Get next section type
                if 'Chorus' in line:
                    section_type = 'chorus'
                elif 'Verse' in line:
                    section_type = 'verse'
                elif 'Produced' in line or 'Instrumental' in line:
                    continue
                elif 'Bridge' in line or 'Hook' in line:
                    section_type = 'bridge'
                else:
                    section_type = 'intro'
            elif line:
                # Convert -, —, and / compounds into two words
                line = line.replace('-', ' ').replace('—', ' ').replace('/', ' ')

                # Append line to section
                section.append([word for word in line.split()])
        formatted_lyrics.append((section_type, section))

        return formatted_lyrics

    def get_syllable_count_word(self, word):
        """
        Formats and retrieves syllable count for individual words, including
        numerals.

        Words are first checked for in `self.counter`. If not found, SbA is
        performed, and if no syllabification exists, then the naive algorithm is
        performed. The resulting syllable count is then added to `self.counter`.

        :param word: Word to count syllables.
        :type word: str | float
        :return n_syllables: Number of syllables.
        :rtype: int
        """

        # Check if word is numerical
        try:
            word = num2words(float(word)).replace('-', ' ')
            return sum([self.get_syllable_count_word(word) for word in word.split()])
        except ValueError:
            # Else, format word
            word = self.regex.sub('', word).lower()

        # Check if word has already been processed
        if word in self.counter:
            return self.counter[word]

        # Remove apostrophes, do SbA
        n_syllables = self._sba(word.replace("'", ''))

        # If SbA fails, do naive
        if n_syllables is None:
            n_syllables = self._naive(word)

        # Add word to already processed words
        self.counter[word] = n_syllables

        return n_syllables

    def get_syllable_count_lyrics(self, formatted_lyrics):
        """
        Formats and retrieves syllable counts for each word in lyrics.

        Returns of list of tuples representing sections, each of which contains
        a list of lists representing lines of lyrics, each of which is a list of
        syllable counts of words in that line. See build_lyrics() for more
        information.

        :param formatted_lyrics: Lyrics output from build_lyrics().
        :type formatted_lyrics: list[tuple(str, list[list[str]])]
        :return syl_lyrics: Syllable counts for words in segmented format.
        :rtype: list[tuple(str, list[list[int]])]
        """

        syl_lyrics = []
        syl_section = []
        for section in formatted_lyrics:
            for line in section[1]:
                syl_section.append([self.get_syllable_count_word(word) for word in line])
            syl_lyrics.append((section[0], syl_section[:]))
            syl_section.clear()

        return syl_lyrics

    def get_syllable_count_per_section(self, syl_lyrics):
        """
        Formats and retrieves syllable counts per section in lyrics.

        Sums syllable counts from each section in return value of
        get_syllable_count_lyrics().

        :param syl_lyrics: Lyrics output from get_syllable_count_lyrics().
        :type lyrics: list[tuple(str, list[list[int]])]
        :return: Syllable counts for each section in segmented format.
        :rtype: list[tuple(str, int)]
        """

        return [(section[0], sum(sum(line) for line in section[1])) for section in syl_lyrics]
