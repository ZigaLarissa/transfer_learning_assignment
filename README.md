
# Transfer Learning with CIFAR-10 Dataset

## Project Overview

This project explores transfer learning using the CIFAR-10 dataset to fine-tune three pre-trained models from TensorFlow Hub: **MobileNetV2**, **ResNet50**, and **InceptionV3**. Transfer learning allows us to leverage models pre-trained on large datasets, such as ImageNet, and adapt them to solve new tasks more efficiently.

In this project, the models were fine-tuned using the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to assess the performance of these models on this new task and compare their strengths and weaknesses.

## Problem Statement

The primary task is to classify images from the CIFAR-10 dataset into 10 categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. This classification task provides a good foundation for evaluating the effectiveness of transfer learning in improving model performance on a relatively small and simple dataset.

## Dataset Overview

- **Dataset**: CIFAR-10
- **Number of Classes**: 10
- **Image Size**: 32x32 pixels
- **Pre-processing**: Images were resized and normalized to match the requirements of the pre-trained models used (e.g., ImageNet input size).

PS: We had to use less percentage of the dataset, since we were having trouble loading the whole of it.

## Models Used

The following pre-trained models were fine-tuned:
1. **MobileNetV2** (originally trained on ImageNet)
2. **ResNet50** (originally trained on ImageNet)
3. **InceptionV3** (originally trained on ImageNet)

Each model was retrained for fewer epochs (due to time constraints) with careful resizing, normalization, and other pre-processing steps to ensure compatibility with the new dataset.

## Evaluation Metrics

The models were evaluated using several metrics to assess their performance:

- **Accuracy**: Measures the proportion of correctly classified images.
- **Loss**: Indicates how well or poorly the model is performing (lower is better).
- **Precision**: The proportion of true positives among all predicted positives.
- **Recall**: The proportion of true positives among all actual positives.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

## Experiment Findings

### MobileNetV2
- This model initially performed well on ImageNet data, confirming its robustness. However, when retrained on CIFAR-10, it showed moderate performance with an accuracy of 75.3%. The trade-off between performance and efficiency makes this model suitable for scenarios where resources are limited.

### ResNet50
- ResNet50 showed better overall performance compared to MobileNetV2. It achieved a higher accuracy of 77.9%, indicating its deeper architecture allowed it to adapt better to the new dataset.

### InceptionV3
- InceptionV3 performed similarly to ResNet50, achieving the same accuracy and evaluation scores, highlighting its effectiveness in handling complex data and features.

## Evaluation Results

| Model          | Accuracy | Loss  | Precision | Recall | F1 Score |
|----------------|----------|-------|-----------|--------|----------|
| MobileNetV2    | 0.7530   | 0.7853| 0.79      | 0.75   | 0.76     |
| ResNet50       | 0.7790   | 0.6790| 0.78      | 0.78   | 0.78     |
| InceptionV3    | 0.7790   | 0.6790| 0.78      | 0.78   | 0.78     |

## Discussion

From the experiments, we observed that:
- **ResNet50 and InceptionV3** performed better than **MobileNetV2**, with comparable metrics.
- **MobileNetV2**, while efficient, showed lower performance, which can be attributed to its design for mobile and resource-constrained environments.
- Transfer learning was effective for all models, showcasing the power of pre-trained models in adapting to new tasks.
- However, **MobileNetV2**'s lower accuracy suggests that models with deeper architectures like ResNet50 and InceptionV3 handle complex datasets like CIFAR-10 better, even when retrained with fewer epochs.

## Conclusion

This project highlights the benefits and limitations of transfer learning. While all models performed reasonably well, deeper architectures like ResNet50 and InceptionV3 demonstrated stronger results on the CIFAR-10 dataset. The trade-off between performance and computational efficiency is evident, making the choice of model dependent on the specific use case.

