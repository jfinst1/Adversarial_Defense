# Adversarial Defense System Installation & Integration Guide

## Step 1: Prerequisites

Ensure your system meets these requirements:

- **Python Version**: 3.8+ (due to dependency compatibility, e.g., TensorFlow, PennyLane).
- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows).
- **Hardware**: Minimum 8GB RAM (more for heavy training); GPU optional but beneficial for TensorFlow.
- **Internet**: Required for ChatGPT API calls (`v8_gpt`) or optional if using Ollama locally (`v8`).

## Step 2: Installation

Follow these steps to set up the environment and install dependencies.

### 1. Clone or Save the Code

Save `adversarial_defense_system_v8_gpt.py` in a directory, e.g., `adversarial_defense`.

Or

Save `adversarial_defense_system_v8.py` in a directory, e.g., `adversarial_defense`.

If using Git:

```bash
git init adversarial_defense
cd adversarial_defense
# Copy the file here
```

### 2. Set Up a Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install numpy nltk scikit-learn tensorflow tensorflow-privacy pennylane adversarial-robustness-toolbox openai asyncio
```

**Notes:**
- Use `tensorflow-gpu` if you have a compatible GPU.
- PennyLane uses the `default.qubit` simulator; no quantum hardware required.
- Skip `openai` if using Ollama locally (`v8`).

### 4. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')
```

### 5. Configure ChatGPT (`v8_gpt`) or Ollama (`v8`)

**For ChatGPT (`v8_gpt`):**

- Get an API key from OpenAI and set it:

```bash
export OPENAI_API_KEY='your-api-key-here'  # Linux/macOS
set OPENAI_API_KEY=your-api-key-here      # Windows
```

- Optionally set the GPT model:

```bash
export GPT_MODEL_NAME='gpt-4'  # Default is gpt-3.5-turbo
```

**For Ollama (`v8`):**

- Install Ollama locally ([Installation Guide](https://ollama.com/install)).
- Start the Ollama server:

```bash
ollama serve
```

Ensure it runs at `http://localhost:11434`.

## Step 3: Running Standalone

### 1. Verify Setup

Run directly:

```bash
python adversarial_defense_system_v8_gpt.py
```

**Expected Output:**

```
INFO:__main__:Testing: VGVzdCBpcyBnb29kIQ==...
INFO:__main__:Sanitized: test good
INFO:__main__:Heuristic: Clean
INFO:__main__:ChatGPT: [Analysis from ChatGPT]
INFO:__main__:Quantum defense prediction: Clean
```

### 2. Troubleshooting

- **No API Key**: Set `OPENAI_API_KEY` or use Ollama.
- **Module Errors**: Re-run `pip install`.
- **Memory Issues**: Reduce epochs or batch size.

## Step 4: Integration Options

### Option 1: Command-Line Interface (CLI)

Modify `main()` to accept CLI inputs:

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Defense System")
    parser.add_argument("--text", type=str, default="VGVzdCBpcyBnb29kIQ==")
    parser.add_argument("--lightweight", action="store_true")
    args = parser.parse_args()

    async def main():
        defense_system = AdversarialDefenseSystem()
        input_vector = np.random.rand(10)
        await defense_system.sanitize_and_detect(args.text, input_vector, lightweight=args.lightweight)

    asyncio.run(main())
```

Run:

```bash
python adversarial_defense_system_v8_gpt.py --text "Hello world" --lightweight
```

### Option 2: Web API (FastAPI)

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()
defense_system = AdversarialDefenseSystem()

@app.post("/analyze")
async def analyze_input(text: str, lightweight: bool = False):
    input_vector = np.random.rand(10)
    await defense_system.sanitize_and_detect(text, input_vector, lightweight)
    return {"status": "Processed", "logs": defense_system.detector.adversarial_log}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Install and run:

```bash
pip install fastapi uvicorn
python adversarial_defense_system_v8_gpt.py
```

Test:

```bash
curl -X POST "http://localhost:8000/analyze" -H "Content-Type: application/json" -d '{"text": "VGVzdCBpcyBnb29kIQ==", "lightweight": true}'
```

### Option 3: Library Module

Example:

```python
from adversarial_defense_system_v8_gpt import AdversarialDefenseSystem
import asyncio

async def run_defense():
    defense = AdversarialDefenseSystem()
    text = "Test input"
    vector = np.random.rand(10)
    await defense.sanitize_and_detect(text, vector, lightweight=True)

if __name__ == "__main__":
    asyncio.run(run_defense())
```

### Option 4: Real-Time Pipeline

Process file input:

```python
async def process_file(file_path: str):
    defense_system = AdversarialDefenseSystem()
    with open(file_path, 'r') as f:
        for line in f:
            input_vector = np.random.rand(10)
            await defense_system.sanitize_and_detect(line.strip(), input_vector, lightweight=True)

if __name__ == "__main__":
    asyncio.run(process_file("input.txt"))
```

## Step 5: Ensuring Proper Operation

- Replace dummy input vectors (`np.random.rand(10)`) with real vectors (e.g., TF-IDF, embeddings).
- Train on real datasets labeled clean/adversarial.
- Optimize (lightweight mode, batch processing, quantum tuning).

**Logging**:

```python
logging.basicConfig(filename='defense.log', level=logging.INFO)
```

## Docker Deployment

**Dockerfile:**

```Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY adversarial_defense_system_v8_gpt.py .
CMD ["python", "adversarial_defense_system_v8_gpt.py"]
```

**requirements.txt:**

```
numpy
nltk
scikit-learn
tensorflow
tensorflow-privacy
pennylane
adversarial-robustness-toolbox
openai
asyncio
```

Build & Run:

```bash
docker build -t adversarial-defense .
docker run -e OPENAI_API_KEY=your-api-key -it adversarial-defense
```

## Final Checklist

- [ ] Dependencies installed
- [ ] ChatGPT/Ollama configured
- [ ] NLTK data downloaded
- [ ] Standalone script runs
- [ ] Integrated into chosen setup
- [ ] Real input vectors used
- [ ] Logs verified

Your environment is now fully configured!

