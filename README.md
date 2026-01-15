# Age Estimation and Gender Classification using CNNs
This project focuses on building deep learning models to estimate a person’s age and predict gender from facial images. Two convolutional neural network (CNN) approaches are implemented and compared:
- Model A: A custom CNN trained from scratch
- Model B: A fine-tuned pre-trained VGG16 model

The project demonstrates end-to-end deep learning workflow, including data exploration, preprocessing, augmentation, multi-task learning, model design, training, and evaluation.

## Dataset
- Source: UTKFace dataset (subset)
- Size: 5,000 face images
- Image resolution: 128 × 128 × 3 (RGB)
- Labels:
  - Age: integer value (regression task)
  - Gender: binary label
    - 0 → Male
    - 1 → Female

Labels are extracted directly from image filenames.

## Tasks and Metrics
| Task | Type | Metric |
|------|------|--------|
| Age estimation | Regression | Mean Absolute Error (MAE) |
| Gender classification | Binary classification | Accuracy |

Both tasks are learned simultaneously using a shared CNN backbone with task-specific output heads.

## Environment Setup
- Platform: Google Colab
- Hardware: GPU enabled
- Frameworks & Libraries:
  - TensorFlow / Keras
  - NumPy
  - OpenCV
  - Matplotlib
  - PIL

Google Drive is mounted to load the dataset and save trained models.

## Data Exploration
Before training, random samples from the dataset are visualised to:
- Verify correct label parsing
- Inspect age distribution and gender balance
- Identify potential noise or bias in the data

Each image is displayed with its corresponding age and gender label.

## Data Preprocessing
- Dataset split:
  - 80% training (4,000 images)
  - 20% validation (1,000 images)
- Pixel values normalised to the range [0, 1]
- Labels parsed directly from filenames
- Images loaded without resizing to preserve the required input shape

## Data Augmentation
To improve generalisation and reduce overfitting, on-the-fly data augmentation is applied during training:
- Random horizontal flip
- Small random rotations
- Random zoom
- Contrast and hue adjustments

Augmentation is implemented using Keras preprocessing layers and applied only during training.

## Model A: Custom CNN (From Scratch)

**Architecture Overview**
- Input: 128 × 128 × 3
- Stacked convolutional blocks with increasing depth
- Max pooling for spatial downsampling
- Dropout and L2 regularisation to reduce overfitting
- Fully connected layers shared by both tasks
- Two output heads:
  - Age: single neuron with ReLU activation
  - Gender: single neuron with sigmoid activation

**Key Features**
- Multi-task learning with shared representations
- Careful regularisation and dropout
- Designed to satisfy constraints on feature map size

## Model B: Fine-Tuned VGG16

**Architecture Overview**
- Base model: VGG16 pre-trained on ImageNet
- Early convolutional layers frozen
- Global Average Pooling to reduce parameters
- Separate dense branches for age and gender prediction
- Regularisation and dropout applied to task-specific heads

**Motivation**

Using a pre-trained backbone enables:
- Faster convergence
- Improved feature extraction
- Better performance on limited data

## Training Configuration
**Model A**
- Optimiser: Adam
- Loss functions:
  - Age: MAE
  - Gender: Binary Crossentropy
- Batch size: 64
- Epochs: up to 170
- Callbacks:
  - Early stopping
  - Learning rate reduction on plateau

**Model B**
- Optimiser: Adam (lower learning rate)
- Batch size: 32
- Epochs: up to 60
- Callbacks:
  - Early stopping

## Evaluation and Results
For both models, training and validation curves are plotted for:
- Gender classification loss
- Gender classification accuracy
- Age estimation loss
- Age estimation MAE

These learning curves are used to:
- Monitor convergence
- Detect overfitting
- Compare stability and performance between models

Overall, the pre-trained VGG16 model demonstrates faster convergence and improved generalisation, while the custom CNN provides insight into end-to-end model design.

## Saved Models
Trained models are saved to Google Drive for reuse and evaluation:
- age_gender_A.keras — Custom CNN
- age_gender_B.keras — Fine-tuned VGG16

## Key Learning Outcomes
- Built and trained multi-output CNN models
- Applied data augmentation and regularisation techniques
- Compared training-from-scratch vs transfer learning
- Analysed learning curves to diagnose model behaviour
- Gained practical experience with regression and classification in computer vision

## Future Improvements
- Class imbalance handling for age distribution
- Age group classification as an auxiliary task
- Hyperparameter optimisation using bayesian optimisation
- Model explainability using Grad-CAM (Gradient-weighted Class Activation Mapping) - a technique that visualises what parts of an image CNN is using to make a prediction
- Evaluation on unseen external datasets
