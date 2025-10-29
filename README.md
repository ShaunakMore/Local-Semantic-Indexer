# 🧠 Local Semantic Image Indexer

A lightweight, privacy-friendly tool that indexes and searches images by meaning, not just pixels.
Powered by Ollama for local AI embeddings — no cloud, no data leaks.

## 🚀 Features

- Runs fully offline using Ollama

- Generates semantic embeddings for images

- Fast similarity search for finding related images

- Simple and easy to extend

## ⚙️ How It Works

1. Each image is processed with an Ollama Vision 3.2 model to generate a semantic description of   the image.

2. Embbeddings for the description are generated using Ollama Embeddings model.

3. Embeddings are stored in a local index (custom vector database and a image map).

4. You can query the system with text to find semantically similar results.