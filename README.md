# Assignment-Two-CNN-Model.ipynb
# CNN-based Rotation Angle Estimation

## Overview
This project implements a Convolutional Neural Network (CNN) to predict the rotation angles of MNIST images. The images are augmented by rotating them randomly, with the rotation angle as the target variable for this regression task. For comparison, a Feed-Forward Artificial Neural Network (ANN) is also trained on the same dataset. The performance of both models is evaluated using **Mean Absolute Error (MAE)** and **Loss**, with the goal of identifying which model is more effective for rotation angle estimation.

## Training Details

- **Optimizer**: Adam optimizer, a popular optimization algorithm that adapts the learning rate during training.
- **Loss Function**: Mean Squared Error (MSE) for regression tasks, which is used to measure the difference between predicted and actual rotation angles.
- **Metric**: Mean Absolute Error (MAE) is tracked during training to evaluate the average prediction error of the models.
- **Epochs**: The models are trained for 20 epochs with a batch size of 64.
- **Validation**: 20% of the data is used as validation data to evaluate model performance during training.

## Results

### Training History

- **Model Loss**: The loss decreases and stabilizes over time, indicating that the model is learning effectively.
- **Mean Absolute Error (MAE)**: The MAE values show that the model’s predictions are very close to the actual rotation angles.

### Observations

- The **CNN model** achieves significantly lower test loss and MAE compared to the **ANN model**, demonstrating the advantage of convolutional layers in handling image-based regression tasks.

### Next Steps

- **Increase Epochs**: Further training might improve the model’s performance if validation loss and MAE are still decreasing.
- **Model Tuning**: Adjusting model complexity, such as adding more layers or changing activation functions, could improve results if overfitting or underfitting occurs.

## Model Comparison

| Model | Test Loss | Test MAE |
|-------|-----------|----------|
| CNN   | 8.75      | 2.00     |
| ANN   | 34.21     | 5.78     |

## Files

- **train_images.npy**: Training dataset of MNIST images (augmented with random rotations).
- **train_angles.npy**: Labels corresponding to the rotation angles for the training set.
- **test_images.npy**: Test dataset of MNIST images (augmented with random rotations).
- **test_angles.npy**: Labels corresponding to the rotation angles for the test set.
- **cnn_model.h5**: The trained CNN model saved in H5 format.
- **ann_model.h5**: The trained ANN model saved in H5 format.

## Challenges and Solutions

- **Overfitting**: The CNN model exhibited signs of overfitting. This was addressed by adding **dropout layers** within the fully connected layers to regularize the model and improve its generalization.
  
- **Model Performance**: The **ANN model** underperformed compared to the CNN model due to the lack of convolutional layers. This experiment highlights the importance of convolutional layers for tasks involving spatial relationships in images, such as rotation angle estimation.

## Conclusion

The CNN model outperforms the ANN model in predicting rotation angles with a significantly lower error. This demonstrates that CNNs are more effective for image-based regression tasks, particularly those that involve spatial relationships such as rotation angle estimation.

