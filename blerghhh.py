import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Download the pre-trained ResNet model
resnet = models.resnet18(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer
resnet.eval()

# Download a pre-trained word embedding model
# In a real-world scenario, you would typically train this as part of the captioning model
# For simplicity, we'll use a pre-trained GloVe embedding in this example
glove_embedding = torch.load('glove_embedding.pt')

# LSTM-based captioning model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=False)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.fc(hiddens[0])
        return outputs

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Generate a caption for an image
def generate_caption(image_path, model, vocab, max_length=20):
    model.eval()
    image = preprocess_image(image_path)
    features = resnet(image).squeeze().detach()

    captions = [vocab['<start>']]
    for _ in range(max_length):
        caption_tensor = torch.tensor(captions).unsqueeze(0)
        lengths = torch.tensor([len(captions)])

        outputs = model(features, caption_tensor, lengths)
        predicted_word_index = outputs.argmax(2)[:, -1].item()
        captions.append(predicted_word_index)

        if predicted_word_index == vocab['<end>']:
            break

    caption = [vocab.idx2word[idx] for idx in captions]
    return ' '.join(caption[1:-1])  # Remove <start> and <end>

# Example usage
if __name__ == "__main__":
    # Download a sample image
    image_url = "https://example.com/sample_image.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Display the image
    image.show()

    # Load your vocabulary (make sure it has the '<start>' and '<end>' tokens)
    # You can create a vocabulary using your dataset and the torchtext library, for example
    vocab = torch.load('vocab.pt')

    # Load your trained captioning model
    model = CaptioningModel(embed_size=300, hidden_size=512, vocab_size=len(vocab))
    model.load_state_dict(torch.load('captioning_model.pth'))

    # Generate and display the caption
    caption = generate_caption( image_url, model, vocab)
    print(f"Generated Caption: {caption}")
