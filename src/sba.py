from collections import Counter, deque

def sba(input, marked_set):
    # Node data class
    class data(object):
        def __init__(self):
            self.outputs = Counter() # Set of arcs going out
            self.sinputs = [] # Set of arcs coming in, filled by BFS
            self.distance = float('inf') # From origin, filled by BFS

    # Initialize and format input and lexical dataset
    input = '#{}#'.format(input.replace('', '*')[1:-1])
    lex = []
    for word in marked_set:
        n = '#'
        for i, ch in enumerate(word):
            n += ch
            if i < len(word)-1 and word[i+1] != '-' and ch != '-':
                n += '*'
        n += '#'
        lex.append(n)

    # Pronunciation lattice
    lattice = {('#', 0): data(), ('#', len(input)-1): data() }

    # Substring matcher and lattice builder
    substring = []
    for entry in lex:
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

    # Decision function 2: get shortest arc length
    path = ''
    node = ('#', len(input)-1)
    while node != ('#', 0):
        path = node[0] + path
        min_freq = float('inf')

        if not lattice[node].sinputs:
            print('UserWarning: No syllabification found')
            return None
        for arc in lattice[node].sinputs:
            if arc[2] < min_freq:
                node, arc_string, min_freq = arc
        path = arc_string + path
    path = path.replace('*', '')[:-1]

    return path

if __name__ == '__main__':
    import os
    from config import resourcesdir

    db_path = os.path.join(resourcesdir, 'syllables.txt')

    marked_set = []
    with open(db_path, 'r') as f:
        for line in f.read().splitlines():
            marked_set.append(line.split()[1])

    input = 'oregano'
    output = sba(input, marked_set)

    print('Input: ' + input)
    print('SBA:   ' + output)
