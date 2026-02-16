# AI Learning Assistant (RAG-based)


This project builds an AI-powered learning assistant for the Machine Learning course using Retrieval-Augmented Generation (RAG). It answers user questions by referencing actual course content (video subtitles), guiding users to relevant videos and timestamps for deeper learning.

## Architecture

The system follows a 6-step RAG workflow:

1. **Videos to Text**: Convert course videos to text using OpenAI Whisper
2. **Chunking**: Split transcripts into manageable chunks with metadata
3. **Text to Vectors**: Convert text chunks to vector embeddings
4. **Queries to Vectors**: Convert user queries to vector embeddings
5. **RAG Setup**: Combine vectorized content for retrieval
6. **LLM Response**: Generate answers using a Large Language Model

## Features

- Converts course videos to MP3 audio files
- Transcribes audio to text using Whisper with timestamp chunking
- Embeds subtitle chunks for semantic search using local embedding models
- Answers user questions by retrieving relevant content and generating human-like responses
- Guides users to specific videos and timestamps for deeper learning
- Handles unrelated questions politely

## Workflow

### 1. Video to Audio Conversion (`video_to_mp3.py`)
Converts all videos in the `videos/` folder to MP3 files in `audios/` using FFmpeg.

### 2. Audio Transcription & Chunking (`mp3_to_jsons.py`)
- Uses OpenAI Whisper to transcribe each MP3 file
- Splits transcripts into chunks with metadata:
  ```json
  {
    "timestamp": "00:23:30",
    "duration": 34,
    "text": "The cat is very good",
    "course": "machine learning course"
  }
  ```
- Saves each transcript as a JSON file in `jsons/`

### 3. Embedding Generation (`preprocess_json.py`)
- Reads all JSON subtitle chunks
- Sends chunk texts to local embedding API (bge-m3 model via Ollama)
- Stores embeddings and metadata in a DataFrame, saved as `embeddings.joblib`

### 4. Semantic Search & Question Answering (`process_incoming.py`)
- Loads embeddings and metadata
- Accepts user questions and creates query embeddings
- Finds most similar subtitle chunks using cosine similarity
- Constructs prompts with relevant chunks and sends to local LLM (llama3.2 via Ollama)
- Generates human-like answers referencing specific videos and timestamps
- Writes prompts and responses to `prompt.txt` and `response.txt`

## File Structure

```
ai-rag-based-project/
├── videos/                 # Original course videos (MP4, AVI, etc.)
├── audios/                 # Converted MP3 audio files
├── jsons/                  # Transcribed subtitle chunks (JSON format)
├── video_to_mp3.py         # Convert videos to MP3 audio
├── mp3_to_jsons.py         # Transcribe audio and create chunks
├── preprocess_json.py      # Generate embeddings from chunks
├── process_incoming.py     # Main Q&A interface
├── embeddings.joblib       # Vector embeddings storage
├── prompt.txt              # Last generated prompt (debug)
├── response.txt            # Last LLM response (debug)
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

## Requirements

### System Requirements
- Python 3.8+
- FFmpeg (for video/audio conversion)
- Ollama (for local LLM and embedding models)

### Python Packages
```bash
pip install openai-whisper pandas numpy scikit-learn joblib requests
```

### Ollama Models
```bash
ollama pull bge-m3      # For embeddings
ollama pull llama3.2    # For text generation
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-rag-based-project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg**
   - Windows: Download from [FFmpeg official site](https://ffmpeg.org/download.html)
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

4. **Setup Ollama**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull required models
   ollama pull bge-m3
   ollama pull llama3.2
   ```

## Usage

### Initial Setup

1. **Place course videos** in the `videos/` directory

2. **Convert videos to MP3**
   ```bash
   python video_to_mp3.py
   ```

3. **Transcribe and chunk audio**
   ```bash
   python mp3_to_jsons.py
   ```

4. **Generate embeddings**
   ```bash
   python preprocess_json.py
   ```

### Question Answering

Run the interactive question answering system:
```bash
python process_incoming.py
```

Example questions:
- "How does linear regression work?"
- "What is the difference between supervised and unsupervised learning?"
- "Explain the k-nearest neighbors algorithm"

## How It Works

1. **Content Processing**: Course videos are converted to audio, then transcribed and chunked with timestamps
2. **Vector Embeddings**: Text chunks are converted to vector representations for semantic search
3. **Query Processing**: User questions are embedded and matched against course content
4. **Context Retrieval**: Most relevant content chunks are retrieved based on similarity
5. **Response Generation**: LLM generates answers using retrieved context, citing specific videos and timestamps
6. **Quality Control**: Unrelated questions are politely declined

## Extending the System

### Adding New Content
1. Add new videos to the `videos/` directory
2. Re-run the processing pipeline:
   ```bash
   python video_to_mp3.py
   python mp3_to_jsons.py
   python preprocess_json.py
   ```

### Model Upgrades
- Swap embedding models via Ollama API (update `preprocess_json.py`)
- Upgrade LLM models (update `process_incoming.py`)
- Popular alternatives: `mxbai-embed-large`, `qwen2.5`, `mistral`

### Performance Optimization
- Increase chunk size for better context
- Adjust similarity thresholds
- Implement caching for frequent queries
- Add parallel processing for large video collections

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is in your system PATH
   - Verify installation with `ffmpeg -version`

2. **Ollama connection errors**
   - Check if Ollama is running: `ollama list`
   - Restart Ollama service if needed

3. **Memory issues with large datasets**
   - Reduce chunk size in `mp3_to_jsons.py`
   - Use batch processing in `preprocess_json.py`

4. **Poor response quality**
   - Try different LLM models
   - Adjust similarity thresholds
   - Improve prompt engineering in `process_incoming.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for audio transcription
- Ollama for local LLM and embedding models
- FFmpeg for media processing
- The Machine Learning course content providers
