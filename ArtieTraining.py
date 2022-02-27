import json
from pickletools import optimize
import numpy as np

import torch
import torch.nn as  nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import ArtieUtils
from model import NeuralNetwork

with open('ArtieIntents.json', 'r') as ArtieIntent:
    intents = json.load(ArtieIntent)

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

XTrain = []
YTrain = []
for (patternsentence, tag) in xy:
    bag = ArtieUtils.WordBag(patternsentence, all)
    XTrain.append(bag)

    label = tags.index(tag)
    YTrain.append(label)

XTrain = np.array(XTrain)
YTrain = np.array(YTrain)

class ChatDataset(Dataset):
    def __init__(self):
        self.nSamples = len(XTrain)
        self.XData = XTrain
        self.YData = YTrain

    def __ItemGet__(self, index):
        return self.XData[index], self.YData[index]

    def __len__(self):
        return self.nSamples

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(XTrain[0])
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
TrainLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)


device = torch.device('cpu')
model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in TrainLoader:
        words = words.to(device)
        labels = labels.to(device)
            
        output = model(words)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model.state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all,
    "tags": tags
} 

Artie = "ArtieData.pth"
torch.save(data, Artie)

print(f'training complete, file saved to {Artie}')
