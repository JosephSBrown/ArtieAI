import json
import numpy as np

import ArtieUtils

with open('ArtieIntents.json', 'r') as Artie:
    intents = json.load(Artie)

print(intents)

all = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = ArtieUtils.tokenise(pattern)
        all.extend(w)
        xy.append((w, tag))

ignore = ['?', '.', '!', ',']
all = [ArtieUtils.stem(w) for w in all if w not in ignore]
all = sorted(set(all))
tags = sorted(set(tags))
print(tags)

XTrain = []
YTrain = []
for (patternsentence, tag) in xy:
    bag = ArtieUtils.WordBag(patternsentence, all)
    XTrain.append(bag)

    label = tags.index(tag)
    YTrain.append(label)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)