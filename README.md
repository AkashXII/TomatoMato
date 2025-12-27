TomatoMato xd
Note: This repository represents a learning and experimentation project.
It was built to understand machine learning workflows, dataset limitations, and deployment, and is not intended for real-world agricultural use.

Overview:
TomatoMan is a small machine learning project created to explore end-to-end ML system design using a mix of image-based and tabular models.
The project focuses on understanding:
1)how CNNs behave on real datasets
2)how different models can be wired together
3)how dataset choice affects what a system can meaningfully claim

What This Project Does:

1)Image-based disease classification--
CNN trained on the PlantVillage dataset
Supports tomato, potato, and pepper leaf images
Experiments with both custom CNNs and pretrained architectures

2)Tabular model (exploratory)--
Random Forest trained on publicly available soil and climate data
Used as an auxiliary signal to experiment with tabular ML
Included to explore multi-model integration, not as a definitive predictor

3)Fusion logic (experimental)--
Simple logic that combines outputs from the CNN and the Random Forest
Implemented to understand decision-level model fusion
Results are illustrative and part of the learning process

4)Streamlit application--
Allows uploading a leaf image and entering environment parameters
Displays model predictions and confidence scores
Demonstrates ML inference and deployment workflow

plot.png
