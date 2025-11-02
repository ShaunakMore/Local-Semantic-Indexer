# 🖼️ Semantic Image Indexer

A privacy-first, AI-powered image search tool that lets you find photos using natural language queries - completely offline and local to your machine.

## ✨ Features

- 🔍 **Natural Language Search** - Find images using descriptions like "sunset over mountains" or "cat sleeping"
- 🔒 **100% Private & Local** - Everything runs on your machine, your images never leave your computer
- 🎯 **Semantic Understanding** - Finds images by meaning, not just filenames or tags
- 💾 **Smart Caching** - Indexes once, search instantly forever
- 🔄 **Auto-Update** - Automatically detects and indexes new images
- ⚙️ **Dual Methods** - Choose between CLIP (fast & lightweight) or Ollama Vision (best quality)

## 🛠️ Two Approaches

### Method 1: CLIP (Recommended)
- **Model Size**: 150-900MB
- **RAM Usage**: 2-4GB
- **Speed**: Fast (⚡⚡⚡)
- **Quality**: Very Good
- **Best For**: Most users, quick searches, limited hardware

### Method 2: Ollama Vision
- **Model Size**: 7.5GB
- **RAM Usage**: 12-16GB
- **Speed**: Slower (⚡)
- **Quality**: Excellent
- **Best For**: Power users, best semantic understanding, high-end hardware

Both methods are **completely private** and run locally on your machine.

---

## 📖 How It Works

### CLIP Method (`indexer_clip.py`)

```
┌─────────────────────────────────────────────────────────┐
│              Your Image Folder (*.jpg, *.png)            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  CLIP Model (OpenAI)                     │
│         Directly converts images to embeddings           │
│              (No caption generation needed)              │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              ShoreDB Vector Database (C++)               │
│        Stores 512D vectors + cosine similarity           │
│              (.index_cache_clip/index.bin)               │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            Text Query → CLIP Embedding                   │
│      Find k-nearest neighbors in vector space            │
│              Return most similar images                  │
└─────────────────────────────────────────────────────────┘
```

**How CLIP Works:**
- CLIP was trained on 400M image-text pairs
- It learns to match images with their descriptions
- Images and text live in the **same semantic space**
- When you search "red car", CLIP finds images that are "close" to that concept
- No need to generate captions - direct image-to-vector conversion

---

### Ollama Method (`indexer_ollama.py`)

```
┌─────────────────────────────────────────────────────────┐
│              Your Image Folder (*.jpg, *.png)            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           LLaMA 3.2 Vision Model (via Ollama)            │
│     Generates detailed caption for each image            │
│  "A golden retriever playing with a ball in a park"      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Nomic Embed Text Model                      │
│          Converts caption text to embedding              │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              ShoreDB Vector Database (C++)               │
│        Stores embeddings + cosine similarity             │
│              (.index_cache/index.bin)                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│      Text Query → Embed → Find similar captions          │
│              Return most similar images                  │
└─────────────────────────────────────────────────────────┘
```

**How Ollama Method Works:**
- Vision LLM generates human-readable descriptions
- Each image gets a detailed caption stored implicitly
- Search matches your query against these rich descriptions
- Better understanding of context, actions, and complex scenes
- Slower but more accurate for nuanced searches

---

## 🗄️ Why ShoreDB?

**ShoreDB** is a custom-built, lightweight vector database specifically designed for this project.

### Advantages:
- ✅ **Simple & Lightweight** - No heavy dependencies like FAISS or ChromaDB
- ✅ **C++ Performance** - Fast search with Python bindings
- ✅ **Cosine Distance** - Perfect for normalized embeddings
- ✅ **No Setup Hassle** - Works out of the box
- ✅ **Privacy-Focused** - Simple binary storage, no external services
- ✅ **Custom Built** - Optimized for small-to-medium datasets (1K-100K images)

### How It Works:
```python
# Add vectors
db.add_vector(id="img_001", vector=[0.123, -0.456, ...])

# Search (k-nearest neighbors)
results = db.k_nearest_neighbours(query_vector, k=5)
# Returns: [(id, distance), (id, distance), ...]

# Persistence
db.save_to_disk("index.bin")  # Save index
db.load_from_file("index.bin")  # Load index
```

ShoreDB is open-source and available at: [github.com/ShaunakMore/Vector-DB-from-scratch]

---

## 🔐 Privacy & Security

**Your images and data are 100% private. Here's our guarantee:**

### What Stays Local:
- ✅ **All images** - Never uploaded or transmitted anywhere
- ✅ **All processing** - CLIP/Ollama run entirely on your device
- ✅ **All embeddings** - Stored in local hidden folder (`.index_cache`)
- ✅ **All searches** - Computed locally, no external queries
- ✅ **All metadata** - Filenames and mappings stay on your disk

### How We Ensure Privacy:

#### CLIP Method:
```python
# Model downloaded once from Hugging Face
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# After download, runs 100% offline
# No internet connection required
# No telemetry, no tracking, no data collection
```

#### Ollama Method:
```python
# Models downloaded via Ollama (one-time)
ollama pull llama3.2-vision:11b
ollama pull nomic-embed-text

# Ollama runs locally on your machine
# No API keys, no external services
# Completely self-hosted
```

### What Data is Stored:
```
.index_cache_clip/  (or .index_cache/)
├── index.bin           # Vector embeddings (mathematical representations)
└── image_map.pkl       # ID-to-filename mapping

NO actual image data is stored in the index
NO cloud synchronization
NO external database connections
```

### Open Source Verification:
- 📖 All code is open source - audit it yourself
- 🔍 No hidden network calls
- 🔒 No analytics or tracking
- 🛡️ No third-party services

**Privacy Philosophy:** Your photos are personal. They should stay on your device, under your control, always.

---

## 🗺️ Roadmap

### Planned Features

#### 🎨 UI/UX (High Priority)
- [ ] **Web Interface** - React frontend with Flask backend
- [ ] **Desktop App** - Tauri/Electron for native experience
- [ ] **Better Results Display** - Grid view with thumbnails and lightbox
- [ ] **Search History** - Quick access to previous searches
- [ ] **Progress Indicators** - Show indexing progress in real-time

#### 🔍 Search Features (High Priority)
- [ ] **Image-to-Image Search** - Find similar images using a reference photo
- [ ] **Hybrid Search** - Combine text query + reference image
- [ ] **Advanced Filters** - Date range, file type, size, dimensions
- [ ] **Negative Search** - "cats but NOT black cats"
- [ ] **Search Within Results** - Refine your search progressively

#### 📁 Organization (Medium Priority)
- [ ] **Collections & Tags** - Organize search results into albums
- [ ] **Duplicate Detection** - Find and remove similar/duplicate images
- [ ] **Smart Auto-Collections** - Auto-group by themes (beach, food, etc.)
- [ ] **Bulk Operations** - Copy, move, delete multiple images at once

#### 🎯 Advanced Features (Medium Priority)
- [ ] **Color Search** - Find images by dominant color
- [ ] **EXIF Metadata Extraction** - Search by date, location, camera
- [ ] **Multi-Folder Support** - Index and search across multiple folders
- [ ] **Image Clustering** - Visualize image relationships in 2D/3D
- [ ] **OCR Text Detection** - Search for text within images

#### ⚙️ Technical Improvements (Low Priority)
- [ ] **Model Selection UI** - Switch between CLIP models easily
- [ ] **Background Indexing** - Watch folders and auto-index new images
- [ ] **Incremental Updates** - Faster re-indexing of changed folders
- [ ] **Export/Import** - Backup and restore indexes
- [ ] **Performance Dashboard** - Show index stats and memory usage

---

## 🎯 Choosing Your Method

### Use CLIP if:
- ✅ You have 8GB+ RAM
- ✅ You want fast results (< 1 second per query)
- ✅ You're searching a personal photo library (< 50K images)
- ✅ You want easy setup with minimal downloads
- ✅ "Very good" quality is good enough

### Use Ollama Vision if:
- ✅ You have 16GB+ RAM
- ✅ You can wait a bit for better quality
- ✅ You want the absolute best semantic understanding
- ✅ You search for complex scenes or activities
- ✅ You need human-readable captions for each image

### Why Not Both?
You can keep both methods! Index with both and compare results to see which works better for your image collection.

---

**Made with ❤️ for privacy-conscious photographers and digital hoarders**

*Keep your memories searchable, secure, and private.*