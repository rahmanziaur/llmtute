# ML-Tute

# README: Assessing Machine Learning Research Experiments

## 1. Purpose

This guide provides a systematic framework for assessing the **experimental quality, validation methodology, and reported performance** of a Machine Learning (ML) research study.

A common mistake in ML research evaluation is to focus primarily on **accuracy**. Accuracy alone does not establish that an ML experiment is valid, reliable, statistically meaningful, or generalizable.

A rigorous assessment should consider:

> **Dataset → Experimental Design → Validation → Metrics → Baselines → Statistical Evidence → Ablation → Generalization → Reproducibility**

---

## 2. Accuracy Is Not the Same as Validation

### Accuracy

Accuracy measures how many predictions are correct:

[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
]

It is useful, but it can be misleading, particularly for **imbalanced datasets**.

For example, suppose a dataset contains:

* 950 samples of Class A
* 50 samples of Class B

A model that predicts every sample as Class A achieves:

[
Accuracy = \frac{950}{1000}=95%
]

However, the model completely fails to identify Class B.

Therefore:

> **A high accuracy does not necessarily mean that the model is good.**

---

# 3. Experimental Assessment Framework

## 3.1 Dataset Assessment

First evaluate the quality of the dataset.

### Questions to ask

* What is the total number of samples?
* How many classes are present?
* Is the dataset balanced?
* Where did the data come from?
* Is the dataset representative of the intended application?
* Are there duplicate samples?
* Are there missing or noisy samples?
* How were labels generated?
* Is the dataset sufficiently large?
* Are the data sources clearly described?

### Important concern

A model trained and tested on a small or highly homogeneous dataset may report excellent performance but have poor real-world generalization.

---

# 4. Train, Validation, and Test Sets

A fundamental ML experiment should clearly distinguish:

```text
                    Complete Dataset
                          |
             +------------+------------+
             |                         |
          Training                  Test Set
             |                         |
      +------+------+
      |             |
   Training      Validation
     Data           Data
```

### Training Set

Used to learn model parameters.

### Validation Set

Used for:

* Hyperparameter selection
* Model selection
* Threshold selection
* Early stopping
* Architecture decisions

### Test Set

Used for **final unbiased evaluation**.

The test set should ideally remain completely unseen until the final evaluation.

---

# 5. Data Leakage

Data leakage is one of the most serious threats to ML experimental validity.

### Example

Suppose a dataset contains images from 100 patients.

If the researchers randomly split individual images:

```text
Patient 1 → Training
Patient 1 → Test
```

the model may effectively learn patient-specific characteristics.

A better approach is:

```text
Patients 1–70  → Training
Patients 71–85 → Validation
Patients 86–100 → Test
```

This is particularly important for:

* Medical datasets
* Video datasets
* IoT sensor data
* Human activity recognition
* Speech datasets
* Financial/time-series data
* Repeated measurements

### Reviewer question

> **Are the units used for splitting the data independent?**

---

# 6. Validation Strategies

## 6.1 Holdout Validation

Example:

```text
70% Training
15% Validation
15% Test
```

Simple and commonly used.

However, results can depend significantly on the particular random split.

---

## 6.2 K-Fold Cross-Validation

Example:

```text
Dataset
   |
   +-- Fold 1 → Test
   +-- Fold 2 → Test
   +-- Fold 3 → Test
   +-- Fold 4 → Test
   +-- Fold 5 → Test
```

The model is trained and evaluated multiple times.

A common choice is:

[
k=5 \quad \text{or} \quad k=10
]

Results can be reported as:

[
Mean \pm Standard\ Deviation
]

For example:

> F1-score = 94.72 ± 0.83%

---

# 7. Nested Cross-Validation

When hyperparameter tuning is substantial, nested cross-validation can provide a more reliable estimate.

```text
             Outer CV
        +-------+-------+
        |               |
   Training          Test
      |
   Inner CV
      |
Hyperparameter
   Selection
```

This prevents information from the evaluation folds from influencing hyperparameter selection.

---

# 8. Appropriate Evaluation Metrics

Accuracy should be selected based on the research problem.

## Classification

Important metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Sensitivity
* Balanced Accuracy
* MCC
* ROC-AUC
* PR-AUC

### Precision

[
Precision=\frac{TP}{TP+FP}
]

Measures how many predicted positives are actually positive.

### Recall

[
Recall=\frac{TP}{TP+FN}
]

Measures how many actual positives are successfully detected.

### F1-score

[
F1=2\frac{Precision\times Recall}{Precision+Recall}
]

Useful when both precision and recall matter.

---

# 9. Metrics by Research Problem

| Problem                   | Recommended Metrics                      |
| ------------------------- | ---------------------------------------- |
| Balanced classification   | Accuracy, Precision, Recall, F1          |
| Imbalanced classification | Macro-F1, Balanced Accuracy, MCC, PR-AUC |
| Medical diagnosis         | Sensitivity, Specificity, F1, ROC-AUC    |
| Multiclass classification | Macro-F1, Weighted-F1, Per-class Recall  |
| Regression                | MAE, MSE, RMSE, R²                       |
| Object Detection          | Precision, Recall, mAP                   |
| Image Segmentation        | IoU, Dice, Precision, Recall             |
| Ranking                   | MRR, NDCG, MAP                           |
| Time-series forecasting   | MAE, RMSE, MAPE                          |

The authors should justify why the selected metrics are appropriate.

---

# 10. Confusion Matrix

For classification experiments, a confusion matrix is highly informative.

Example:

| Actual / Predicted |  A |  B |  C |
| ------------------ | -: | -: | -: |
| A                  | 95 |  3 |  2 |
| B                  |  4 | 92 |  4 |
| C                  |  1 |  5 | 94 |

It allows researchers to identify:

* Which classes are confused
* Minority-class failures
* False positives
* False negatives
* Class-specific performance

A paper reporting only overall accuracy provides much less information.

---

# 11. Per-Class Performance

Researchers should ideally report:

| Class   | Precision | Recall |   F1 |
| ------- | --------: | -----: | ---: |
| Class A |      97.2 |   98.1 | 97.6 |
| Class B |      95.1 |   93.4 | 94.2 |
| Class C |      97.0 |   96.0 | 96.5 |

This is especially important when classes are imbalanced.

---

# 12. Baseline Comparison

A proposed method should be compared with appropriate alternatives.

At minimum, consider:

```text
Simple Baseline
      ↓
Traditional ML Method
      ↓
Established Deep Learning Method
      ↓
Recent State-of-the-Art Method
      ↓
Proposed Method
```

Examples of baselines may include:

* Logistic Regression
* Random Forest
* SVM
* Decision Tree
* CNN
* ResNet
* Transformer
* Recent state-of-the-art models

The selected baselines should be **relevant to the research problem**.

---

# 13. Fair Comparison

A comparison is meaningful only when experimental conditions are reasonably comparable.

Check:

* Same dataset
* Same test set
* Same evaluation metrics
* Comparable preprocessing
* Comparable data augmentation
* Appropriate hyperparameter tuning
* Same or clearly reported evaluation protocol

Be cautious when a paper compares its model with published results obtained using a **different dataset split or evaluation protocol**.

---

# 14. Ablation Study

An ablation study determines whether the proposed components actually contribute to performance.

Suppose a proposed model contains:

```text
CNN
 +
Attention
 +
Feature Fusion
 +
Proposed Module
```

An appropriate ablation could be:

| Model                | Accuracy |       F1 |
| -------------------- | -------: | -------: |
| Baseline CNN         |     91.8 |     91.2 |
| CNN + Attention      |     93.1 |     92.7 |
| CNN + Feature Fusion |     94.2 |     93.8 |
| Full Model           | **95.6** | **95.1** |

This provides evidence that the proposed components contribute to the improvement.

---

# 15. Statistical Reliability

A single experimental result is often insufficient.

Instead of:

> Accuracy = 98.5%

a stronger report might be:

> Accuracy = 98.5 ± 0.4% over five independent runs.

Researchers should consider reporting:

* Mean
* Standard deviation
* Confidence interval
* Multiple independent runs
* Statistical significance where appropriate

---

# 16. Statistical Significance

Suppose:

```text
Proposed Model = 98.2%
Baseline       = 97.9%
```

The difference is:

[
0.3%
]

The question is:

> Is the 0.3% improvement statistically or practically meaningful?

The authors should provide appropriate evidence rather than assuming that a higher number automatically means a better model.

---

# 17. Generalization

A strong ML model should not merely memorize the experimental dataset.

Assess whether the authors investigate:

* External datasets
* Unseen subjects
* Unseen environments
* Different devices
* Different acquisition conditions
* Cross-dataset testing
* Temporal generalization
* Robustness to noise

A particularly strong experiment is:

```text
Dataset A
   ↓
Training
   ↓
Model
   ↓
Dataset B
   ↓
External Evaluation
```

---

# 18. Robustness Analysis

Researchers can test performance under challenging conditions.

For example:

```text
Clean Data       → 96.2%
Low Noise        → 95.4%
Medium Noise     → 92.7%
High Noise       → 88.1%
```

This provides more information than a single accuracy value.

---

# 19. Error Analysis

A good research paper should investigate **where and why the model fails**.

Questions include:

* Which classes are frequently confused?
* What types of samples are difficult?
* Are there systematic errors?
* Are failures caused by noise?
* Are minority classes poorly detected?
* Does the model fail under particular environmental conditions?

Error analysis helps establish whether the proposed approach is genuinely useful.

---

# 20. Computational Evaluation

Performance is not always the only consideration.

For practical systems, evaluate:

* Number of parameters
* FLOPs
* Training time
* Inference time
* Memory requirements
* Model size
* GPU/CPU requirements
* Energy consumption where relevant

For example:

| Model   |   F1 | Parameters | Inference |
| ------- | ---: | ---------: | --------: |
| Model A | 95.2 |        25M |     18 ms |
| Model B | 95.8 |       150M |    125 ms |

Model B has better F1, but Model A may be preferable for edge deployment.

---

# 21. Reproducibility

A research experiment should provide enough information for another researcher to reproduce it.

Check whether the paper reports:

* Dataset source
* Dataset preprocessing
* Train/validation/test split
* Random seed
* Model architecture
* Hyperparameters
* Optimizer
* Learning rate
* Batch size
* Number of epochs
* Hardware
* Software/framework versions
* Evaluation procedure
* Code availability

---

# 22. Common Experimental Red Flags

Be cautious when you see:

### Red Flag 1

> "Our model achieved 99.8% accuracy."

but no validation methodology is described.

### Red Flag 2

Only accuracy is reported for a highly imbalanced dataset.

### Red Flag 3

The same dataset is repeatedly used for training, tuning, and testing.

### Red Flag 4

No confusion matrix or per-class results.

### Red Flag 5

No comparison with strong baselines.

### Red Flag 6

No ablation study despite multiple proposed components.

### Red Flag 7

Results from only one random experiment.

### Red Flag 8

Extremely high performance on a very small dataset.

### Red Flag 9

Random sample splitting is used when samples are grouped by subject/patient/device.

### Red Flag 10

The test set appears to have been used for hyperparameter tuning.

---

# 23. Recommended ML Experiment Assessment

A reviewer can use the following structure:

| Assessment Area  | Key Question                                                |
| ---------------- | ----------------------------------------------------------- |
| Dataset          | Is the dataset appropriate and sufficiently representative? |
| Data Quality     | Are labels and samples reliable?                            |
| Data Split       | Are train/validation/test sets properly separated?          |
| Leakage          | Is information leakage prevented?                           |
| Validation       | Is the validation methodology appropriate?                  |
| Metrics          | Are appropriate metrics reported?                           |
| Class Balance    | Is imbalance properly addressed?                            |
| Confusion Matrix | Are class-specific errors examined?                         |
| Baselines        | Are relevant methods compared?                              |
| Fairness         | Are comparisons conducted fairly?                           |
| Ablation         | Is the contribution of each component demonstrated?         |
| Statistics       | Are results statistically reliable?                         |
| Robustness       | Is model robustness evaluated?                              |
| Generalization   | Is performance tested on unseen conditions/data?            |
| Error Analysis   | Are failure cases investigated?                             |
| Efficiency       | Are computational costs reported?                           |
| Reproducibility  | Can the experiment be reproduced?                           |

---

# 24. A Practical Scoring Scheme

You can score each category from **0–5**:

| Score | Interpretation |
| ----: | -------------- |
|     0 | Not reported   |
|     1 | Very poor      |
|     2 | Weak           |
|     3 | Acceptable     |
|     4 | Good           |
|     5 | Excellent      |

For example:

| Category             |   Score |
| -------------------- | ------: |
| Dataset              |      /5 |
| Data Splitting       |      /5 |
| Leakage Prevention   |      /5 |
| Validation           |      /5 |
| Metrics              |      /5 |
| Baselines            |      /5 |
| Statistical Analysis |      /5 |
| Ablation             |      /5 |
| Generalization       |      /5 |
| Robustness           |      /5 |
| Error Analysis       |      /5 |
| Reproducibility      |      /5 |
| **Total**            | **/60** |

### Interpretation

* **51–60:** Excellent experimental methodology
* **41–50:** Strong
* **31–40:** Acceptable but needs improvement
* **21–30:** Weak
* **0–20:** Serious methodological concerns

The numerical score should **support—not replace—expert judgment**.

---

# 25. Recommended Reviewer Principle

When assessing an ML research experiment, do not ask only:

> **"How high is the accuracy?"**

Instead ask:

> **"How convincingly do the experiments demonstrate that the reported performance is valid, reliable, statistically meaningful, and generalizable?"**

A strong ML experiment therefore has:

```text
                ML EXPERIMENT
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
    Dataset       Validation      Metrics
       │              │              │
       ↓              ↓              ↓
  Data Quality   Leakage Control   F1/AUC/etc.
       │              │              │
       └──────────────┼──────────────┘
                      ↓
               Baseline Comparison
                      ↓
                 Ablation Study
                      ↓
              Statistical Analysis
                      ↓
              Robustness Testing
                      ↓
             Generalization Testing
                      ↓
                Error Analysis
                      ↓
               Reproducibility
```

## Bottom Line

**Accuracy is a performance metric; validation is an experimental methodology.**

For ML research, **high accuracy without rigorous validation is not strong evidence**. A convincing experiment should demonstrate that the model performs well on genuinely unseen data, using appropriate metrics and fair baselines, with leakage controlled and results shown to be sufficiently reliable and reproducible.


# Helpful: https://medium.com/@alfred.weirich/transfer-learning-with-keras-tensorflow-an-introduction-51d2766c30ca?sk=0ef6f25197e4ce6147cfd80c6f6642dc
