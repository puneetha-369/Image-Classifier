# Imports necessary tools
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import argparse
import json

# Brings in arguments from CLI
parser = argparse.ArgumentParser(description="Predict image class using a trained deep learning model.")

parser.add_argument('image_path', type=str, help='Path to the image to be predicted.')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names.')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')

args = parser.parse_args()

# Define a function to load the model from a checkpoint
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# Define a function to process the image
def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model, returns a tensor."""
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image)

# Define a function to make predictions
def predict(image_path, model, top_k=5, device='cpu'):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.to(device)
    model.eval()
    
    image = process_image(image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    probs, indices = F.softmax(output, dim=1).topk(top_k, dim=1)
    
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

# Main function
def main():
    # Load the model
    model = load_model(args.checkpoint)
    
    # Determine device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Predict the class
    probs, classes = predict(args.image_path, model, args.top_k, device)
    
    # Map classes to category names if JSON file is provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(cls, cls) for cls in classes]
    
    # Print the results
    print("Predictions:")
    for prob, cls in zip(probs, classes):
        print(f"Class: {cls}, Probability: {prob:.3f}")

if __name__ == '__main__':
    main()
