# 💊 Fine-Tuning an LLM on AWS SageMaker

An end-to-end example of **fine-tuning a small open-source language model** on domain-specific data and **deploying it as a live endpoint** on AWS.

The model is **TinyLlama 1.1B**, fine-tuned with **LoRA** on a pharmaceutical question-and-answer dataset (for example, *"Explain the mechanism of action of Metformin"*). The whole pipeline runs on **Amazon SageMaker**: the data lives in S3, training runs on a GPU instance, and the trained model is served from a real-time endpoint.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2)
![SageMaker](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws&logoColor=white)

---

## ✨ What This Project Covers

- **Instruction dataset:** pharma Q&A pairs in `instruction / input / output` format
- **Parameter-efficient fine-tuning:** LoRA adapters trained on the attention layers, so only a small fraction of the weights change
- **Managed training:** a SageMaker Hugging Face training job on an `ml.g5.xlarge` GPU instance
- **Deployment:** the trained model is packaged from S3 and deployed to a SageMaker real-time endpoint
- **Experimentation:** a local notebook for loading the data and testing the LoRA setup before running a paid cloud job

---

## 🏗️ Pipeline

```
pharma_instruction_data.csv
          │  upload
          ▼
   Amazon S3 (datasets/)
          │
          ▼
SageMaker training job (Hugging Face container, ml.g5.xlarge)
   TinyLlama-1.1B + LoRA (r=8, alpha=16, q_proj & v_proj)
          │  model.tar.gz
          ▼
   Amazon S3 (models/)
          │
          ▼
SageMaker real-time endpoint ──► send a question, get an answer
```

---

## ⚙️ Training Configuration

| Setting | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| Method | LoRA (PEFT) |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 2 |
| Batch size (per device) | 2 |
| Learning rate | 2e-5 |
| Training instance | `ml.g5.xlarge` |
| Inference instance | `ml.m5.xlarge` |

---

## 🧰 Tech Stack

Python · Hugging Face Transformers · Datasets · PEFT (LoRA) · PyTorch · Amazon SageMaker · Amazon S3 · boto3

---

## 📁 Project Structure

```
Fine-tuning-onAWS/
├── experiments.ipynb            # Load the dataset and set up the LoRA model locally
├── estimator_launcher.ipynb     # Launch the SageMaker training job and deploy the endpoint
├── pharma_instruction_data.csv  # Sample instruction-tuning dataset
├── scripts/                     # Training entry point (train.py) used by SageMaker, not yet committed
├── main.py
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 How to Run

### Prerequisites
- An AWS account with SageMaker access and an execution role
- Two S3 buckets: one for the dataset and one for model artifacts
- GPU quota for `ml.g5.xlarge` training instances

### Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt sagemaker boto3
   ```
2. **Upload the dataset to S3**
   ```bash
   aws s3 cp pharma_instruction_data.csv s3://<your-dataset-bucket>/datasets/
   ```
3. **Try it locally (optional):** open `experiments.ipynb` to load the data and check the LoRA setup.
4. **Train on SageMaker:** open `estimator_launcher.ipynb` in SageMaker Studio, update the S3 bucket names, and run the cells to start the training job.
5. **Deploy:** run the deployment cells to create a real-time endpoint from the trained model.

> 💡 **Remember to delete the endpoint when you're done.** SageMaker endpoints are billed for every hour they run.

---

## 📚 What I Learned

- How LoRA makes fine-tuning affordable by training small adapter layers instead of the full model
- Packaging a training script for SageMaker's Hugging Face containers
- Managing the full model lifecycle on AWS: data in S3, training jobs, model artifacts, and endpoints

---

## 👤 Author

**Madhu** · [GitHub](https://github.com/madhugadi)
