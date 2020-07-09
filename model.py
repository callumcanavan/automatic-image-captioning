import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size= hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer for captions
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM RNN
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer for outputting vocab scores
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # Embed captions and permute to form suitable for lstm
        embedded_captions = self.embedding(captions)
        
        # Concatenate image features with all but last caption
        inputs = torch.cat(
            (features.view(-1, 1, self.embed_size), embedded_captions[:, :-1, :]),
            dim=1
        )
        
        # Get LSTM outputs and permute to form suitable for linear
        outputs, _ = self.lstm(inputs)
        
        # Get vocab scores
        outputs = self.hidden2vocab(outputs)
        return outputs
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # Index of end token
        end_idx = 1
        
        # Generate output words until end token or max_len is reached
        last_word = None
        output = []
        while (last_word != end_idx) and (len(output) <= max_len):
            Z, states = self.lstm(inputs, states)
            Z = self.hidden2vocab(Z)
            last_word = torch.argmax(Z, dim=2)
            output += [last_word.item()]
            inputs = self.embedding(last_word)
        
        return output             
                         