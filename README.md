# FPGA-based-Mb2DKDNet-for-Field-Disease-and-Pest-Diagnosis_1
Testing Instructions for Reviewers
1. Model Weights
The trained model weights are provided in “Mb2DKDNet.pth”.
2. Dependencies
To run the model, ensure the following environment is installed:
Python ≥ 3.8.20
PyTorch == [2.4.0]
Other required packages: numpy, opencv-python, etc.
3. Load the Model
Provide a minimal code snippet to load the weights:
import torch
from your_model_architecture import YourModel  
model = YourModel(*args, **kwargs)  
model.load_state_dict(torch.load("Mb2DKDNet.pth"))
model.eval()
4. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

5. Run Inference
Example code for testing on new data:
input_data = ...  
with torch.no_grad():
    output = model(torch.from_numpy(input_data).unsqueeze(0))  
print("Prediction:", output)
6. Expected Output
Format of the output (class probabilities).
Example:
predicted_class = torch.argmax(output).item()
