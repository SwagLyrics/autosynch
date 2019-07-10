from collections import Counter, deque

input = ''
db_path = ''

# Initialize and format input and lexical database
input = '#{}#'.format(input.replace('', '*')[1:-1])
with open(db_path, 'r') as f:
    db = f.read().split()

# Data class
class data(object):
    def __init__(self):
        self.children = Counter()
        self.sparents = []
        self.distance = float('inf')

# Pronunciation lattice
lattice = {('#', 0): data(), ('#', len(input)-1): data() }

# Substring matcher and lattice builder
substring = []
for entry in db:
    entry = '#{}#'.format(entry)

    for offset in range(-len(input)+3, len(entry)-2):
        for i in range(max(0, -offset), min(len(input), len(entry)-offset)):
            if input[i] == entry[i+offset] or (input[i] == '*' and entry[i+offset] == '|'):
                substring.append((entry[i+offset], i))
            else:
                for i, node in enumerate(substring):
                    if node not in lattice:
                        lattice[node] = data()
                    for j in range(i+1, len(substring)):
                        lattice[node].children[substring[j]] += 1
                substring.clear()

        for i, node in enumerate(substring):
            if node not in lattice:
                lattice[node] = data()
            for j in range(i+1, len(substring)):
                lattice[node].children[substring[j]] += 1
        substring.clear()

# Decision function 1: get shortest path(s)
queue = deque([('#', 0)])
lattice[('#', 0)].distance = 0
while queue:
    node = queue.popleft()
    for adjacent in lattice[node].children:
        if lattice[node].distance + 1 < lattice[adjacent].distance:
            queue.append(adjacent)
            lattice[adjacent].distance = lattice[node].distance + 1
            lattice[adjacent].sparents.append(node)
        elif lattice[node].distance + 1 == lattice[adjacent].distance:
            lattice[adjacent].sparents.append(node)

# Decision function 2: get shortest arc length
path = []
node = ('#', len(input)-1)
while node != ('#', 0):
    path.append(node)
    temp = float('inf')
    for parent in lattice[node].sparents:
        if lattice[parent].distance < temp:
            node = parent
path.append(('#', 0))
path.reverse()
