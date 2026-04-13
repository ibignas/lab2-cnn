# Vehicle Classification CNN

This project implements convolutional neural networks for vehicle image classification. It uses transfer learning to train and evaluate four pre-trained models: VGG16, ResNet50, MobileNetV2, and EfficientNetB0.

The pipeline involves training these models on a categorized vehicle dataset, applying data augmentation, and fine-tuning the final layers. A Streamlit web application (`app.py`) is provided to run local CPU inference on new images, visualize the augmentation process, and compare the performance metrics (confusion matrices, learning curves) of each model. 

Ultimately, the models are evaluated based on their test accuracy, and the web interface automatically parses these results to select the highest-performing model for real-time predictions.

## How to run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the web interface:
   ```bash
   streamlit run app.py
   ```