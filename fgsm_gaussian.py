import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

class FGSMGaussian:
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.model.eval()
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        
    def generate(self, x, y_true, epsilon=0.03, std=0.1):
        """
        Generate FGSM adversarial examples with Gaussian noise
        
        Args:
            x: Input image (batch)
            y_true: True labels
            epsilon: Attack strength parameter
            std: Standard deviation for Gaussian noise
            
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
        
        # Create Gaussian noise
        gaussian_noise = torch.normal(mean=0., std=std, size=x.shape).to(x.device)
        
        # Create perturbation with Gaussian noise
        perturbation = epsilon * torch.sign(x.grad.data) * gaussian_noise
        
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
    
    # Create FGSM attack with Gaussian noise
    fgsm_gaussian = FGSMGaussian(model)
    
    # Load example image (you would need to provide an image path)
    # image = load_image("example.jpg")
    # label = torch.tensor([232])  # Example label
    
    # Generate adversarial example
    # adv_image = fgsm_gaussian.generate(image, label, epsilon=0.03, std=0.1)
    
    print("FGSM with Gaussian noise implementation ready for use") 