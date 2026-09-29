# Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain

[](http://kdd.org/) [](https://pytorch.org/) [](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation for our KDD 2026 paper: **"Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain"**.

In this work, we propose **DAMPS**, a novel framework that calibrates multimodal representations in the spectral domain to enhance recommendation performance.

## 🏗️ Model Architecture

![DAMPS Framework](Figure/DAMPS.png)

> **Figure 1:** The overall architecture of the proposed DAMPS framework.

-----

## ⚙️ Prerequisites

To reproduce the results, please ensure you have the following environment set up.

**Core Dependencies:**

  * **Python:** 3.8.20
  * **PyTorch:** 2.4.1
  * **Torchvision:** 0.19.1
  * **NumPy:** 1.22.4
  * **SpaCy:** 3.7.2
  * **Scikit-learn:** 1.3.2

**Quick Install:**

```bash
pip install -r requirements.txt
# OR manually:
pip install numpy==1.22.4 spacy==3.7.2 scikit-learn==1.3.2 torch==2.4.1 torchvision==0.19.1
```

-----

## 📂 Datasets

We provide pre-processed datasets including text and image features extracted via Sentence-Transformers and CNNs.

### 1\. Amazon Datasets (Baby, Sports, Clothing, Elec)

Download the datasets from Google Drive:
👉 **[Download Link (Baby/Sports/Elec)](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)**

### 2\. MicroLens (Short-video Recommendation)

An alternative dataset for short-video scenarios:
👉 **[Download Link (MicroLens)](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link)**

### Directory Setup

After downloading, please organize your directory as follows:

```text
.
├── data/
│   ├── baby/
│   ├── sports/
│   └── ...
├── src/
│   ├── main.py
│   ├── MGCN.py
│   ├── DAMPS.py
│   └── ...
├── Figure/
│   └── DAMPS.png
└── README.md
```

-----

## 🚀 Usage

### Training the Model

Navigate to the source directory and run `main.py` using the following arguments:

```bash
cd src/
python main.py -m <model_name> -d <dataset_name> -g <gpu_id>
```

**Arguments:**

  * `-m`: Model name (e.g., `SMORE`, `MGCN`).
  * `-d`: Dataset name (e.g., `baby`, `sports`, `clothing`, `elec`).
  * `-g`: GPU ID (e.g., `0`).

**Example:**

```bash
python main.py -m MGCN -d baby -g 0
```

### Switching Between DAMPS and Vanilla MGCN

By default, the **DAMPS** framework is integrated into the `MGCN` model to demonstrate our performance improvements.

  * **To run DAMPS (Ours):** Simply run the command above. The code for the core logic is located in `DAMPS.py`.
  * **To run Vanilla MGCN (Baseline):** If you wish to compare results with the original MGCN model, please **comment out** lines **105** and **153** in `src/MGCN.py`.

-----

## 📧 Contact

If you have any questions regarding the code or the paper, please feel free to contact:

**Email:** wmh18872323043@163.com

-----

## 📝 Citation

If you find this repository or our paper useful, please cite:

```bibtex
@inproceedings{damsp2026,
  title={Enhancing Multimodal Recommendation via Multimodal Representation Calibration in Spectral Domain},
  author={Minghui Wang, Tingting Zhang, Yu Li, and Yi Chang},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
