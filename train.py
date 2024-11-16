# Imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

# Argument Parsing
parser = argparse.ArgumentParser(description="Train a new network on a dataset and save the model as a checkpoint.")
parser.add_argument('data_dir', type=str, help="Path to dataset directory")
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help="Path to save the trained model checkpoint")
parser.add_argument('--arch', type=str, default='vgg16', help="Choose model architecture: vgg16 or densenet121")
parser.add_argument('--learning_rate', type=float, default=0.003, help="Learning rate for training")
parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units in classifier")
parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training if available")

args = parser.parse_args()

# Data Loading and Transformation
train_dir = f"{args.data_dir}/train"
valid_dir = f"{args.data_dir}/valid"

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

# Model Setup
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_features = 25088
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_features = 1024
else:
    raise ValueError("Unsupported architecture. Choose 'vgg16' or 'densenet121'.")

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(input_features, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)

# Training the Model
epochs = args.epochs
print_every = 40
steps = 0

for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

# Save the Checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {
    'arch': args.arch,
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'epochs': args.epochs,
    'learning_rate': args.learning_rate,
    'hidden_units': args.hidden_units
}

torch.save(checkpoint, args.save_dir)
print("Model checkpoint saved.")
