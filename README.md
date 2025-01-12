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
    Initializing ThinkingLLM (this may take a moment)...        

    Test Type: general
    Question: What makes a good leader?

    - Understanding the question: What makes a good leader? - Breaking down key
    components: Leadership, character, skills, and experience - Clarifying the
    question: What makes a good leader who can inspire and motivate people? -
    Providing a clear answer: A leader who has a strong character, a proven track
    record of success, and a deep understanding of their team's needs.  Example: A
    company is looking for a new CEO. The CEO is asked, "What makes a good leader?"
    Breaking down key components: Leadership, character, skills, and experience  1.
    Leadership: The CEO needs to have the ability to inspire and motivate their
    team. They need to have a strong personality and be able to connect with their
    team members on a personal level.  2. Character: The CEO needs to be a role
    model for their team. They need to have strong values and a commitment to doing
    what is right for their team.  3. Skills: The CEO needs to be able to lead their
    team through complex challenges. They need to have a deep understanding of their
    team's needs and be able to bring together the right resources to overcome
    obstacles.  4. Experience: The CEO needs to have a proven track record of
    success. They need to have experience in leading a successful team and in
    developing and executing strategies.  Clarifying the question: A leader who has
    a strong character, a proven track record of success, and a deep understanding
    of their team's needs.  My answer: A leader who has a strong character, a proven
    track record of success, and a deep understanding of their team's needs. This
    includes having a strong personality, commitment to doing what is right for
    their team, ability to lead their team through complex challenges, and proven
    track record of success.

    Generation time: 12.16 seconds
    ==========================================================

    Press Enter for next test...


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