import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Define the model class (must be the same as during training)
class RipenessCNN(nn.Module):
    def __init__(self, num_classes):
        super(RipenessCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class_names = ['Bellpepper_fresh', 'Bellpepper_intermediate_fresh', 'Bellpepper_rotten', 'Carrot_fresh', 'Carrot_intermediate_fresh', 'Carrot_rotten', 'Cucumber_fresh', 'Cucumber_intermediate_fresh', 'Cucumber_rotten', 'Potato_fresh', 'Potato_intermediate_fresh', 'Potato_rotten', 'Tomato_fresh', 'Tomato_intermediate_fresh', 'Tomato_rotten', 'ripe_apple', 'ripe_banana', 'ripe_mango', 'ripe_oranges', 'ripe_strawberry', 'rotten_apple', 'rotten_banana', 'rotten_mango', 'rotten_oranges', 'rotten_strawberry', 'unripe_apple', 'unripe_banana', 'unripe_mango', 'unripe_oranges', 'unripe_strawberry']
num_classes = len(class_names)

# Initialize the model (use the same num_classes as during training)
# num_classes = 3  # Change this to match your actual number of classes
model = RipenessCNN(num_classes)

# Load the saved state dict
model.load_state_dict(torch.load('ripeness_cnn_model_aug.pth', map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_ripeness(image_path, model, transform, class_names):
    # Load and transform the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = F.softmax(output, dim=1)[0] * 100
    
    # Return results
    return {
        'class': class_names[predicted.item()],
        'confidence': probability[predicted.item()].item(),
        'all_probabilities': {class_names[i]: prob.item() for i, prob in enumerate(probability)}
    }

# Example usage
# class_names = ['unripe', 'semi-ripe', 'fully-ripe']  # Replace with your actual class names

result = predict_ripeness('apple.jpeg', model, transform, class_names)
print(f"Predicted class: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}%")
print("All probabilities:")
for cls, prob in result['all_probabilities'].items():
    print(f"{cls}: {prob:.2f}%")