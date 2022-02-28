import torch
import json
import random

from model import NeuralNetwork
from ArtieUtils import WordBag, tokenise

device = torch.device('cpu')

with open('ArtieIntents.json', 'r') as f:
    intents = json.load(f)

Artie = 'ArtieData.pth'
data = torch.load(Artie)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
AllWords = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Artie"


def get_response():
    print('Let\'s Chat! Type \'Quit\' To Exit')
    while True:
        sentence = input('You: ')
        if sentence == 'quit':
            break

        sentence = tokenise(sentence)
        X = WordBag(sentence, all)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy()

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    print(f'{bot_name}: {random.choice(intent["responses"])}')
        else:
            print(f'{bot_name}: I Do Not Understand...')
