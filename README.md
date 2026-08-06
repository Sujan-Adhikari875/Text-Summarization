# Text Summarizer using T5, Hugging Face & FastAPI

## Project Overview

This project is an AI-powered **Text Summarization** application that automatically converts long conversations into short, meaningful summaries. The application is built by fine-tuning the **T5-Small (Text-to-Text Transfer Transformer)** model on the **SAMSum** dialogue summarization dataset using **PyTorch** and **Hugging Face Transformers**.

After training, the model is deployed as a **FastAPI** web application with a simple HTML interface where users can paste text and instantly receive a generated summary.

---

# Objectives

The primary goals of this project are:

* Build an end-to-end NLP application.
* Fine-tune a pretrained Transformer model.
* Generate high-quality summaries from dialogues.
* Deploy the trained model using FastAPI.
* Provide an easy-to-use web interface for inference.

---

# Features

* Fine-tuned **T5-Small** model for dialogue summarization.
* Automatic text cleaning and preprocessing.
* Transformer-based sequence-to-sequence text generation.
* Beam Search decoding for better summary quality.
* REST API developed using FastAPI.
* Interactive frontend using HTML, CSS, and JavaScript.
* Automatic device detection (CPU, CUDA GPU, or Apple MPS).
* Save and reload the trained model for future inference.

---

# Technologies Used

| Technology                | Purpose                           |
| ------------------------- | --------------------------------- |
| Python                    | Programming Language              |
| PyTorch                   | Deep Learning Framework           |
| Hugging Face Transformers | Pretrained T5 Model and Tokenizer |
| FastAPI                   | REST API Development              |
| Pandas                    | Dataset Processing                |
| HTML                      | User Interface                    |
| CSS                       | Styling                           |
| JavaScript                | API Communication                 |
| Jinja2                    | HTML Template Rendering           |

---

# Dataset

The project uses the **SAMSum Dataset**, a benchmark dataset specifically designed for dialogue summarization.

Each record contains:

* **Dialogue** – A conversation between two or more people.
* **Summary** – A human-written summary describing the conversation.

Example:

**Dialogue**

```text
John: Are you coming to the meeting?
Sarah: Yes, I'll arrive in 15 minutes.
John: Great. Bring the project report.
Sarah: Sure.
```

**Summary**

```text
Sarah will attend the meeting in 15 minutes and bring the project report.
```

---

# Project Workflow

The application follows the complete NLP pipeline:

### 1. Data Loading

Training and validation datasets are loaded from CSV files using Pandas.

### 2. Data Preprocessing

The text is cleaned by:

* Removing line breaks
* Removing extra spaces
* Converting text to lowercase
* Trimming unnecessary whitespace

This improves consistency during training.

### 3. Tokenization

The T5 tokenizer converts text into numerical token IDs that can be processed by the Transformer model.

### 4. Fine-Tuning

Instead of training a model from scratch, the pretrained **T5-Small** model is fine-tuned using the SAMSum dataset. During training, the model learns how to map dialogues to their corresponding summaries.

### 5. Model Saving

After training, both the model and tokenizer are saved locally in the `saved_summary` directory, allowing them to be reused without retraining.

### 6. Inference

When a user submits a dialogue:

* The text is cleaned.
* The tokenizer converts it into tokens.
* The trained model generates a summary using Beam Search.
* The generated tokens are decoded back into readable text.

### 7. Deployment

FastAPI exposes the summarization model as a REST API, while the frontend communicates with the API to display summaries in real time.

---

# Model Configuration

| Parameter              | Value    |
| ---------------------- | -------- |
| Model                  | T5-Small |
| Epochs                 | 6        |
| Batch Size             | 8        |
| Warmup Steps           | 500      |
| Weight Decay           | 0.005    |
| Maximum Input Length   | 512      |
| Maximum Summary Length | 150      |
| Beam Size              | 4        |
| Early Stopping         | Enabled  |

---

# Project Structure

```text
project/
│
├── saved_summary/          # Trained model and tokenizer
├── results/                # Training checkpoints
├── samsum-train.csv
├── samsum-validation.csv
├── app.py                  # FastAPI backend
├── index.html              # Frontend interface
├── README.md
└── requirements.txt
```

---

# API Endpoint

### POST `/summarize/`

### Request

```json
{
    "dialogue": "Your conversation here..."
}
```

### Response

```json
{
    "summary": "Generated summary"
}
```

---

# Installation

Install the required libraries:

```bash
pip install torch transformers fastapi uvicorn pandas sentencepiece jinja2
```

Run the application:

```bash
uvicorn app:app --reload
```

Open the application:

```text
http://127.0.0.1:8000
```

---

# Important Concepts

### Transformer

A Transformer is a deep learning architecture that uses self-attention mechanisms to understand relationships between words in a sequence. It processes text in parallel, making it faster and more effective than traditional recurrent neural networks.

### T5 (Text-to-Text Transfer Transformer)

T5 treats every NLP task as a text generation problem. Whether the task is translation, summarization, or question answering, both the input and output are represented as text. This unified framework makes T5 highly flexible and effective.

### Tokenization

Tokenization converts text into smaller units called tokens and maps them to numerical IDs. These IDs serve as input to the Transformer model during both training and inference.

### Fine-Tuning

Fine-tuning starts with a pretrained language model and trains it further on a task-specific dataset. This approach reduces training time while improving performance on the target task.

### Beam Search

Beam Search is a decoding algorithm that explores multiple candidate summaries at each generation step and selects the sequence with the highest overall probability. It generally produces more fluent and accurate summaries than greedy decoding.

### FastAPI

FastAPI is a modern Python web framework for building APIs. It offers high performance, automatic request validation using Pydantic, interactive API documentation, and seamless integration with machine learning models.

### Device Selection

The application automatically checks for available hardware acceleration:

* CUDA for NVIDIA GPUs
* Apple MPS for Apple Silicon
* CPU when no GPU is available

This ensures the application runs efficiently on different systems.

---

# Future Improvements

* Fine-tune larger T5 models (T5-Base or T5-Large).
* Evaluate performance using ROUGE metrics.
* Support PDF, Word, and TXT document summarization.
* Add multilingual summarization.
* Containerize the application using Docker.
* Deploy to cloud platforms such as AWS, Azure, or Hugging Face Spaces.

---

# Author

**Adhikari Sujan**
