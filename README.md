# Paying Attention Where it Counts

## Getting Started & Execution
> [!IMPORTANT]
> - **Accelerator Required**: A hardware accelerator (GPU/CUDA-enabled device) is required to train the U-Net and Attention U-Net models.
> - **Modular Pipeline**: To build and execute the pipeline step-by-step, run the modular notebook [notebooks/pipeline.ipynb](file:///c:/Users/rohet/OneDrive/Documents/CS_WORK/Machine_Learning/Attention_Vs,_U-NET/notebooks/pipeline.ipynb).
> - **Original Reference Notebook**: To see the original Google Colab notebook, check [Main_Notebook.ipynb](file:///c:/Users/rohet/OneDrive/Documents/CS_WORK/Machine_Learning/Attention_Vs,_U-NET/Main_Notebook.ipynb). **Please DO NOT run this notebook locally**, as it is for reference only and contains Colab-specific paths and commands that will not execute in local environments.


## 2. Goal
This project implements, evaluates, and compares an **Attention U-Net** architecture against a standard **baseline U-Net** on the Oxford-IIIT Pet dataset. The target task is binary image segmentation, separating the pet body (foreground) from the surrounding background.

## 3. Motivation / Research Question
The original Attention U-Net paper was designed for segmenting small, hard-to-spot anatomical targets in visually noisy medical scan environments. The Oxford-IIIT Pet dataset presents the opposite scenario—the target regions (pet bodies) are large, prominent, and visually obvious. This raises the research question: **Does the attention mechanism still provide a performance benefit over a standard U-Net on an easy-to-localize target, or does it add compute and memory costs with little to no performance gain?**

## 4. Dataset
We evaluate our models on the **Oxford-IIIT Pet Dataset**, which contains images of 37 pet breeds with corresponding pixel-level trimap masks. 
- **Resolution**: All images and masks are resized to $256 \times 256$ pixels.
- **Split**: We use a fixed 80/20 train/validation split.
- **Boundary Handling**: Ambiguous boundary pixels (labeled as `255`) are masked out and ignored during loss calculation and metrics evaluation. Training on boundary zones introduces noise at transition boundaries; ignoring them encourages the model to learn clean background vs. foreground signals.

## 5. Data Augmentation
To improve generalization and prevent overfitting, we apply the following spatial and color-level augmentations on-the-fly to the training set:
- **Random Horizontal Flip**: Exploits the left-right reflection symmetry of animals to expand the training footprint.
- **Random Rotation ($\pm 10^\circ$)**: Prepares the network to handle diverse tilts and pet alignments.
- **Brightness & Contrast Jitter ($\pm 10\%$)**: Simulated using linear scaling to accommodate varying lighting conditions and exposure levels.
- **Normalization**: Standardizes images across channels using precomputed dataset statistics (RGB channel mean and standard deviation) to stabilize network gradients.

---

## 6. Architectures

### 6a. Baseline U-Net
Our standard contractive-expansive network uses padded convolutions to retain spatial dimensions. All convolutional modules follow a `3×3 Conv → Batch Normalization → ReLU` pattern, paired with `2×2 MaxPool` in the encoder and `2×2 UpConv` in the decoder.

| Stage | Input Dimension | Output Dimension | Output Channels |
| :--- | :--- | :--- | :--- |
| **Encoder Block 1** | $(256, 256, 3)$ | $(256, 256, 64)$ | 64 |
| **Encoder Block 2** | $(128, 128, 64)$ | $(128, 128, 128)$ | 128 |
| **Encoder Block 3** | $(64, 64, 128)$ | $(64, 64, 256)$ | 256 |
| **Encoder Block 4** | $(32, 32, 256)$ | $(32, 32, 512)$ | 512 |
| **Bottleneck** | $(16, 16, 512)$ | $(16, 16, 1024)$ | 1024 |
| **Decoder Block 1** | $(32, 32, 512 + 512)$ | $(32, 32, 512)$ | 512 |
| **Decoder Block 2** | $(64, 64, 256 + 256)$ | $(64, 64, 256)$ | 256 |
| **Decoder Block 3** | $(128, 128, 128 + 128)$ | $(128, 128, 128)$ | 128 |
| **Decoder Block 4** | $(256, 256, 64 + 64)$ | $(256, 256, 64)$ | 64 |
| **Final Layer** | $(256, 256, 64)$ | $(256, 256, 1)$ | 1 (Logits) |

*Note: $+$ represents concatenation of the skip connection.*

### 6b. Attention U-Net
The Attention U-Net shares the contractive backbone of the baseline model, but introduces an **Attention Gate** on the skip connection features before concatenation at each decoder block.

- **Gate Mechanism**: The gate accepts the skip feature map ($x$) and the gating signal ($g$) from the lower decoder layer.
- **Coefficient Computation**: It projects both signals, sums them, applies a ReLU activation followed by a 1x1 convolution, and runs a Sigmoid function to generate spatial attention coefficients ($\alpha \in [0, 1]$).
- **Filtering**: The skip connection is multiplied by $\alpha$ to suppress background noise and focus features on the object of interest. This adds an extra learned filtering step per skip connection, which is the only structural difference from the baseline.

---

## 7. Loss Function
The models are trained using a hybrid loss consisting of a 1:1 addition of:
$$\text{Loss} = \text{BCEWithLogitsLoss} + (1 - \text{DiceScore})$$

- **Binary Cross Entropy (BCE)**: Optimizes pixel-level class classification.
- **Dice Loss**: Directly optimizes the intersection-over-union metric of the pet segmentation boundary.
- **Ignore Masking**: Boundary pixels (label `255`) are masked out and completely excluded from both calculations.

---

## 8. Training Protocol

| Parameter | Value |
| :--- | :--- |
| **Resolution** | $256 \times 256$ |
| **Split** | 80/20 Fixed Train/Val |
| **Batch Size** | 8 |
| **Optimizer** | Adam |
| **Initial Learning Rate** | $1 \times 10^{-3}$ |
| **LR Scheduler** | `ReduceLROnPlateau` (factor=0.5, patience=5) |
| **Epochs** | 20 |
| **Random Seed** | 42 (and 123 for baseline comparison) |
| **Ignore Label** | 255 |

---

## 9. Metrics & Where Computed

| Metric | Where Computed / Logged |
| :--- | :--- |
| **Train Loss / Train Dice** | Inside the training epoch loop |
| **Val Loss / Val Dice** | Inside the validation epoch loop |
| **Best Val Dice** | Inside the main training driver |
| **Epoch Time** | Inside the main training driver |
| **Convergence Speed** | Evaluated post-training |
| **FLOPs (MACs)** | Profiled before training |
| **Param Count (Total / Block)** | Profiled before training |
| **Inference Time (Latency)** | Profiled before training (synchronized runs) |

---

## 10. Model Comparison (Benchmarks)
All measurements were computed on a CUDA-enabled GPU device.

| Benchmark | Baseline U-Net | Attention U-Net |
| :--- | :--- | :--- |
| **Total Parameters** | 31,043,521 | 31,395,045 (+1.13%) |
| **GFLOPs (MACs)** | 54.74 GFLOPs | 55.35 GFLOPs (+1.11%) |
| **Avg. Inference Latency** | 20.16 ms | 21.57 ms (+6.99%) |
| **Validation Dice (Seed 42)** | 0.9397 | **0.9460** (+0.63% absolute) |

### Layer-Wise Parameter Breakdown

| Module | Baseline U-Net | Attention U-Net | Difference (Attention Overhead) |
| :--- | :--- | :--- | :--- |
| **enc1** | 38,976 | 38,976 | +0 |
| **enc2** | 221,952 | 221,952 | +0 |
| **enc3** | 886,272 | 886,272 | +0 |
| **enc4** | 3,542,016 | 3,542,016 | +0 |
| **bottleneck** | 14,161,920 | 14,161,920 | +0 |
| **dec1** | 9,178,624 | 9,442,561 | +263,937 |
| **dec2** | 2,295,552 | 2,361,985 | +66,433 |
| **dec3** | 574,336 | 591,169 | +16,833 |
| **dec4** | 143,808 | 148,129 | +4,321 |
| **final** | 65 | 65 | +0 |

---

## 11. Results & Interpretation
- **Results**: The baseline U-Net achieved a Best Validation Dice score of **0.9397** on Seed 42. Under the same configuration, the Attention U-Net achieved a Best Validation Dice score of **0.9460**, showing a **+0.63% absolute improvement**.
- **Interpretation**: The attention mechanism *did* show a slight positive improvement in segmentation accuracy on the Oxford-IIIT Pet dataset. This gain is achieved with minimal computational overhead: parameter count increased by only **1.13%** and inference latency rose by only **1.41 ms** (+6.99%). Therefore, even when target objects are large and visually prominent, attention gates provide a small, cost-effective benefit in boundary delineation without substantially compromising inference speed.
