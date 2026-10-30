# Applied AI Midterm — Image Super-Resolution and Classification Using SRGAN

## Project Overview

This project explores whether image super-resolution can improve binary image classification. I implemented and compared two MobileNetV2-based classifiers:

- **Model A:** Trained on original 128 × 128 cat-and-dog images.
- **Model B:** Trained on 128 × 128 images reconstructed by SRGAN from 32 × 32 inputs.

The goal was to evaluate whether reconstructing low-resolution images using SRGAN could preserve enough useful information for accurate classification.

## Dataset and Preprocessing

The project uses the Cats vs. Dogs dataset, containing 25,000 images.

The dataset was divided into:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 15,750 | 63% |
| Validation | 1,750 | 7% |
| Testing | 7,500 | 30% |

Together, training and validation account for 70% of the dataset.

For classification, images were resized to 128 × 128 pixels and normalized to the range [-1, 1]. Random horizontal flipping, rotation, and zooming were applied during training.

For SRGAN, the 128 × 128 images were downsampled to 32 × 32 pixels before being passed to the generator.

## Model A: Original Image Classification

Model A uses MobileNetV2 pretrained on ImageNet.

The original classification layer was replaced with global average pooling, dropout (0.3), and a sigmoid output layer.

Training was performed in three stages:

1. Training the classification head with the backbone frozen.
2. Continuing classification-head training.
3. Fine-tuning part of the backbone with a reduced learning rate.

The model used binary cross-entropy loss and the Adam optimizer.

## SRGAN Architecture and Training

The SRGAN consists of a generator and discriminator.

The generator uses eight residual blocks and two PixelShuffle upsampling blocks to reconstruct 128 × 128 images from 32 × 32 inputs.

The discriminator uses convolutional layers to distinguish real high-resolution images from generated images.

Training was performed in two stages:

- Generator pretraining: 5 epochs using MSE loss.
- Adversarial training: 150 epochs using pixel, perceptual, and adversarial losses.

The final recorded validation reconstruction metrics were:

| Metric | Value |
|---|---:|
| PSNR | 25.31 dB |
| SSIM | 0.7033 |

## Model B: SRGAN-Based Classification

Model B uses the same MobileNetV2 architecture as Model A but is trained on SRGAN-generated images.

It starts from fresh ImageNet-pretrained weights rather than reusing Model A's trained weights.

The same three-stage training approach was used to make the comparison consistent.

## Results

Both models were evaluated on the same 7,500-image test set.

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 97.84% | 95.11% |
| Precision | 97.29% | 95.48% |
| Recall | 98.43% | 94.69% |
| F1-score | 97.85% | 95.09% |
| ROC-AUC | 99.82% | 99.23% |

Model A made 162 incorrect predictions, while Model B made 367.

## Discussion

The results showed that Model A performed better than Model B across all evaluation metrics.

Although SRGAN reconstructed the images to 128 × 128 pixels, the generated images were not as clear or detailed as the originals. Downsampling to 32 × 32 pixels removes information that cannot always be recovered during reconstruction.

This experiment demonstrates that increasing image resolution does not necessarily restore the original image quality or improve classification performance.

## Project Structure

```text
applied-ai-midterm/
├── notebooks/
│   └── midterm-appl-ai.ipynb
├── src/
│   ├── model_a.py
│   ├── srgan_models.py
│   ├── train_model_a.py
│   ├── train_model_b.py
│   ├── pretrain_srgan.py
│   ├── train_srgan.py
│   ├── generate_model_b_data.py
│   ├── continue_model_a.py
│   ├── continue_model_b.py
│   ├── finetune_model_a.py
│   ├── finetune_model_b.py
│   └── evaluate_models.py
├── results/
│   ├── model_a/
│   ├── model_b/
│   ├── srgan/
│   └── evaluation/
├── data/
│   └── splits/
└── README.md
```

The image dataset and large model checkpoints are not included in the GitHub repository.

## Training and Evaluation

Run the following commands from the project root after setting up the Python environment and preparing the dataset:

```bash
python src/pretrain_srgan.py
python src/train_srgan.py --epochs 150
python src/generate_model_b_data.py

python src/train_model_a.py
python src/continue_model_a.py
python src/finetune_model_a.py

python src/train_model_b.py
python src/continue_model_b.py
python src/finetune_model_b.py

python src/evaluate_models.py
```

The Jupyter notebook contains image comparisons, training curves, evaluation metrics, confusion matrices, and discussion of the results.

## Conclusion

This project successfully implemented an SRGAN-based image reconstruction pipeline and compared its effect on downstream classification.

Although the SRGAN-generated images did not improve classification accuracy, the experiment highlighted the difference between reconstructing visually plausible images and preserving information useful for classification.