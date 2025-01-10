# PromptEngineeringChatbot

## Project Description
This project implements a pseudo "chain-of-thought" or prompt engineering approach in a smaller language model (TinyLlama 1.1B). The chatbot breaks down its reasoning process into explicit steps when answering questions, similar to how larger models like ChatGPT approach problems. This enhances debugging  and specific use case scenarios.

comparision_demo.py is a comparision bot that outputs the same question in both prompt engineered style and without it to show difference.

### Features
- Step-by-step reasoning process
- Multiple thinking patterns (general, analysis, problem-solving)
- GPU acceleration support
- Response time tracking
- Customizable prompt templates
- Memory-efficient implementation suitable for consumer GPUs(ensure to account for CUDA version integration)

## Requirements
- Python 3.10 or newer
- CUDA-capable GPU (tested on RTX 3070)
- 16GB+ GPU VRAM
- Windows/Linux/MacOS

## Installation

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m virtualenv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the demo script:
```bash
python PromptEngineering.py
```

The script will:
1. Load the TinyLlama model
2. Run through test questions(you can edit these to add your own questions)
3. Display thinking process and generation time for each response

### Example Output:
    Present in alice_analysis..txt and comparision_..txt files in this folder.

## Customization

You can modify the thinking patterns in the `generate_thinking_prompt` method of the `ThinkingLLM` class. Current patterns include:
- General reasoning
- Systematic analysis
- Problem-solving

## Technical Details

The project uses:
- TinyLlama 1.1B Chat model
- PyTorch with CUDA support
- Hugging Face Transformers library
- Half-precision (FP16) for efficient memory usage

## Files
- `PromptEngineering.py`: Main implementation
- `comparision_demo.py` : Comparision Feature
- `requirements.txt`: Required Python packages
- `README.md`: Project documentation
- `alice_analysis_...txt` : Example use case with Alice in Wonderland pdf
- `comparision_20...txt` : Example cases with answers in prompt engineered and non-engineered responses.

## Memory Usage
- Base model: ~2.2GB VRAM
- Runtime usage: ~4-6GB VRAM
- Recommended: 8GB+ VRAM for comfortable operation

## Known Limitations
- Response quality limited by model size (1.1B parameters)
- May require prompt engineering for best results
- Generation times vary based on input complexity
- To ensure a significantly faster experience, needs specific cuda and torch versions respective to the user's Nvidia graphics card and drivers.