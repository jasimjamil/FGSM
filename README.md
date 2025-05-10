# FGSM Adversarial Attack Implementation

This project implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples, along with a variant using Gaussian noise. It includes a FastAPI-based REST API for serving the attack.

## Project Structure
.
├── README.md
├── requirements.txt
├── fgsm.py # Standard FGSM implementation
├── fgsm_gaussian.py # FGSM with Gaussian noise
└── app_fgsm.py # FastAPI implementation

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Components

### 1. FGSM Implementation (fgsm.py)
- Standard implementation of the Fast Gradient Sign Method
- Uses PyTorch for gradient computation
- Includes utility functions for image loading and prediction

### 2. FGSM with Gaussian Noise (fgsm_gaussian.py)
- Modified version of FGSM using Gaussian noise
- Adds randomness to the perturbation
- Maintains the same interface as standard FGSM

### 3. FastAPI Implementation (app_fgsm.py)
- RESTful API for the FGSM attack
- Endpoints:
  - GET /: Health check
  - POST /attack: Generate adversarial examples
- Accepts image uploads and returns both original and adversarial images

## Running the API

Start the FastAPI server:
```bash
uvicorn app_fgsm:app --reload
```

The API will be available at `http://localhost:8000`

## API Usage

### Generate Adversarial Example
```bash
curl -X POST "http://localhost:8000/attack" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_image.jpg" \
     -F "label=232" \
     -F "epsilon=0.03"
```

Response format:
```json
{
    "status": "success",
    "original_prediction": 232,
    "adversarial_prediction": 147,
    "original_image": "base64_encoded_string",
    "adversarial_image": "base64_encoded_string",
    "epsilon": 0.03
}
```

## Testing

To test the implementations:

1. Standard FGSM:
```python
python fgsm.py
```

2. FGSM with Gaussian noise:
```python
python fgsm_gaussian.py
```

3. API:
```bash
curl http://localhost:8000/
```

## Notes

- The project uses ResNet18 as the target model
- Images are automatically resized to 224x224 pixels
- Epsilon (attack strength) can be adjusted via the API
- All images are processed in RGB format
