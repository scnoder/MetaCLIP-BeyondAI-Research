# MetaCLIP for Diabetic Retinopathy Detection

This repository contains two notebooks that apply **MetaCLIP** to the task of classifying diabetic retinopathy (DR) severity from retinal fundus images using the **EyePACS** dataset.

---

## What is MetaCLIP?

MetaCLIP is an improved approach to training image-text models like CLIP. Standard CLIP training suffers from noisy or inaccurate web captions. MetaCLIP fixes this by using a better filtering and matching process to create higher-quality image-text pairings, resulting in stronger and more reliable visual representations.

---

## Dataset

**EyePACS** — Diabetic Retinopathy Detection  
Emma Dugas, Jared, Jorge, and Will Cukierski. *Diabetic Retinopathy Detection.* Kaggle, 2015.  
https://kaggle.com/competitions/diabetic-retinopathy-detection

Available on HuggingFace: https://huggingface.co/datasets/bumbledeep/eyepacs

### Classes

| ID | Label |
|----|-------|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Retinopathy |
| 2 | Moderate Retinopathy |
| 3 | Severe Retinopathy |
| 4 | Proliferative Retinopathy |

### Dataset Notes
- Images were collected from different camera models, so visual appearance may vary
- Images are resized to 1024×1024
- The dataset is heavily imbalanced — approximately 73.3% of images are Grade 0 (No DR)

### License
MIT License — see dataset page for full terms.

---

## Notebooks

### 1. `MetaCLIP_ZeroShot.ipynb` — Zero-Shot Classification

Uses MetaCLIP out of the box with no training. The model receives detailed text descriptions of each DR grade and classifies images by matching them to the closest description.

**Approach:** Each class is described with a clinical text prompt (e.g. *"a retina with extensive intraretinal hemorrhages, venous beading, severe intraretinal microvascular abnormalities (IRMA), but without new vessel formation"*) and the model predicts by finding the best image-text match.

**Key packages:**
- `autodistill` — image labeling and model export
- `autodistill-metaclip` — MetaCLIP zero-shot classifier plugin
- `scikit-learn` — precision, recall, and F1 evaluation
- `matplotlib` — visualization

**Note:** A runtime patch is applied to fix a bug in `autodistill-metaclip` v0.1.3 where the model tag `metaclip_400m` was not being recognized correctly.

**Results:** Zero-shot MetaCLIP showed class imbalance issues, frequently predicting majority classes, and required approximately 8 hours of training time to fine-tune. Evaluation is done using per-class precision, recall, and F1 scores visualized as bar charts.

---

### 2. `MetaCLIP_Finetuning.ipynb` — Fine-Tuning

Fine-tunes `facebook/metaclip-2-worldwide-s16` on EyePACS for image classification using the HuggingFace `Trainer` API.

**Approach:**
- Loads EyePACS and extracts the `label_code` column (integer grades 0–4)
- Fixes class imbalance using `RandomOverSampler` — duplicates minority grades until all 5 classes have equal representation
- Splits data 70% train / 30% test with stratification
- Applies data augmentation to training images (random rotation, sharpness adjustment)
- Fine-tunes for 4 epochs on a T4 GPU

**Key packages:**
- `transformers` — `AutoModelForImageClassification`, `Trainer`, `TrainingArguments`
- `datasets` — HuggingFace dataset loading and processing
- `imbalanced-learn` — `RandomOverSampler` for class balancing
- `torchvision` — image transforms
- `evaluate` — accuracy metric

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Model | `facebook/metaclip-2-worldwide-s16` |
| Epochs | 4 |
| Batch size (train) | 32 |
| Batch size (eval) | 8 |
| Learning rate | 2e-5 |
| Weight decay | 0.02 |
| Warmup steps | 50 |
| Hardware | T4 GPU |

**Results:**

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|--------------|-----------------|----------|
| 1 | 0.8264 | 0.7128 | 76.83% |
| 2 | 0.6736 | 0.6245 | 79.28% |
| 3 | 0.6321 | 0.5959 | 80.41% |
| 4 | 0.5818 | 0.5755 | 81.14% |

Final test accuracy: **81.1%** | Test loss: **0.575** | Evaluation speed: **50.8 samples/second**

---

## Getting Started

### Prerequisites
- Python 3.10+
- Google Colab (recommended) with T4 GPU enabled
- HuggingFace account and access token

### HuggingFace Token Setup
1. Go to huggingface.co and sign in
2. Click your profile picture → Settings
3. Click **Access Tokens** in the left sidebar
4. Click **New Token**, name it, set role to **Read**
5. Copy the token (starts with `hf_...`)
6. In the notebook, run:
```python
from huggingface_hub import notebook_login
notebook_login()
```

### GPU Check
```python
import torch
print(torch.cuda.is_available())  # Should print True
```
In Colab: **Runtime → Change runtime type → T4 GPU**

---

## Repository Structure

```
├── MetaCLIP_ZeroShot.ipynb      # Zero-shot classification with MetaCLIP
├── MetaCLIP_Finetuning.ipynb    # Fine-tuning MetaCLIP on EyePACS
└── README.md
```

---

## Citation
1. Emma Dugas, Jared, Jorge, and Will Cukierski. Diabetic Retinopathy Detection.
https://kaggle.com/competitions/diabetic-retinopathy-detection, 2015. Kaggle.
2.  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE. [IEEE Xplore](https://doi.org/10.1109/cvpr.2016.90)
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. [Internet Archive](https://doi.org/10.48550/ARXIV.1905.11946)
4. Chuang, Y.-S., Li, Y., Wang, D., Yeh, C.-F., Lyu, K., Raghavendra, R., Glass, J., Huang, L., Weston, J., Zettlemoyer, L., Chen, X., Liu, Z., Xie, S., Yih, W., Li, S.-W., & Xu, H. (2025). Meta CLIP 2: A Worldwide Scaling Recipe (Version 3). [Internet Archive](https://doi.org/10.48550/ARXIV.2507.22062)
5. Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. Jama, 316(22), 2402-2410. [Jama Network](https://jamanetwork.com/journals/jama/fullarticle/2588763)
6. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). [NIPS](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

