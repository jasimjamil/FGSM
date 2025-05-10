import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

class FGSM:
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.model.eval()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
    def generate(self, x, y_true, epsilon=0.03):
        """
        Generate FGSM adversarial examples
        
        Args:
            x: Input image (batch)
            y_true: True labels
            epsilon: Attack strength parameter
            
        Returns:
            Adversarial examples
        """
        x.requires_grad = True
        
        # Forward pass
        outputs = self.model(x)
        
        # Calculate loss
        loss = self.loss_fn(outputs, y_true)
        
        # Get gradients
        loss.backward()
        
        # Create perturbation
        perturbation = epsilon * torch.sign(x.grad.data)
        
        # Generate adversarial example
        x_adv = x + perturbation
        
        # Clamp to valid image range [0,1]
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv.detach()

def load_image(image_path, size=(224, 224)):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def predict(model, image):
    """Make prediction for image"""
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

if __name__ == "__main__":
    # Example usage
    # Load a pretrained model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Create FGSM attack
    fgsm = FGSM(model)
    
    # Load example image (you would need to provide an image path)
    # image = load_image("example.jpg")
    # label = torch.tensor([232])  # Example label
    
    # Generate adversarial example
    # adv_image = fgsm.generate(image, label, epsilon=0.03)
    
    print("FGSM implementation ready for use") 