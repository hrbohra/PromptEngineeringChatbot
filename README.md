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
- Python 3.12
- CUDA-capable GPU (tested on RTX 3070) CUDA 11.8
- 16GB+ GPU VRAM
- Windows

## Installation

1. Create and activate a virtual environment via powershell terminal:
```bash
# Create virtual environment
python -m virtualenv venv

# Activate virtual environment
.\venv\Scripts\activate

```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```
Note- if using 11.8 CUDA the above will work. For your version of CUDA, find the respective Pytorch command at their official website- https://pytorch.org/get-started/locally/

To check you cuda, start up the Python interpreter in the terminal(by typing "python" and enter.) then type:

```
import torch
print(torch.version.cuda)  

```
The above will print the CUDA version PyTorch is using.

It is crucial to install three torch libraries with specific cuda version tags to ensure your gpu is utlized, otherwise this project all though utilizing the lower end of LLMs, will still be infeasable on consumer grade CPUs. A GPU with correct drivers and compatible python libraries are crucial for this project.

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
    Present in example_test_case for PromptEngineering.py,alice_analysis..txt and comparision_..txt files in this folder.

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
- `alice_analysis_...txt` : Example test case 
- `comparision_20...txt` : Example cases with answers in prompt engineered and non-engineered responses.
- `example_test_case.txt` : a test run of PromptEngineering.py

## Memory Usage
- Base model: ~2.2GB VRAM
- Runtime usage: ~4-6GB VRAM
- Recommended: 8GB+ VRAM for comfortable operation

## Known Limitations
- Response quality limited by model size (1.1B parameters)
- May require prompt engineering for best results
- Generation times vary based on input complexity
- To ensure a significantly faster experience, needs specific cuda and torch versions.