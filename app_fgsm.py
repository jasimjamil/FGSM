import torch
import torchvision.models as models
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from fgsm import FGSM, load_image
import numpy as np
import base64
from typing import Optional
from torchvision import transforms

app = FastAPI(title="FGSM Attack API")

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Initialize FGSM attack
fgsm = FGSM(model)

def image_to_base64(image_tensor):
    """Convert tensor to base64 string"""
    # Convert tensor to numpy array and then to PIL Image
    image_np = (image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    
    # Convert PIL Image to base64
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.post("/attack")
async def create_adversarial_example(
    file: UploadFile = File(...),
    label: int = Form(...),
    epsilon: Optional[float] = Form(0.03)
):
    """
    Create an adversarial example using FGSM
    
    Args:
        file: Input image file
        label: True label for the image
        epsilon: Attack strength parameter (default: 0.03)
        
    Returns:
        JSON response with attack results
    """
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image to tensor
        transform = torch.nn.Sequential(
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        )
        image_tensor = transform(image).unsqueeze(0)
        
        # Create label tensor
        label_tensor = torch.tensor([label])
        
        # Generate adversarial example
        adv_image = fgsm.generate(image_tensor, label_tensor, epsilon=epsilon)
        
        # Get predictions
        original_pred = predict(model, image_tensor)
        adversarial_pred = predict(model, adv_image)
        
        # Convert images to base64 for response
        original_b64 = image_to_base64(image_tensor)
        adversarial_b64 = image_to_base64(adv_image)
        
        return JSONResponse({
            "status": "success",
            "original_prediction": int(original_pred),
            "adversarial_prediction": int(adversarial_pred),
            "original_image": original_b64,
            "adversarial_image": adversarial_b64,
            "epsilon": epsilon
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/")
def read_root():
    return {"message": "FGSM Attack API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 