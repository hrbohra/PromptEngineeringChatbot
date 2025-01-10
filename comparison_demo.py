import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import textwrap
import time
from datetime import datetime

class ComparisonLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize model with both thinking and direct response capabilities"""
        print("Loading model (this may take a moment)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate_thinking_prompt(self, question):
        """Create a prompt that encourages step-by-step thinking"""
        return f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
Let me think about this step by step:
1) First, I need to understand what's being asked
2) Then, I'll break down the key components
3) Finally, I'll provide a clear answer

My thinking process:<|im_end|>"""

    def generate_direct_prompt(self, question):
        """Create a prompt for direct response"""
        return f"""<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

    def get_response(self, question, use_thinking=True, max_length=512):
        """Generate a response with or without thinking steps"""
        prompt = self.generate_thinking_prompt(question) if use_thinking else self.generate_direct_prompt(question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        ).to(self.model.device)
        
        start_time = time.time()
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            no_repeat_ngram_size=3
        )
        end_time = time.time()
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]
        return textwrap.fill(response, width=80), end_time - start_time

    def compare_responses(self, question):
        """Get and compare both types of responses"""
        print("\n" + "="*80)
        print(f"Question: {question}")
        print("="*80)
        
        # Get thinking response
        print("\nThinking Process Response:")
        print("-"*80)
        thinking_response, thinking_time = self.get_response(question, use_thinking=True)
        print(thinking_response)
        print(f"\nGeneration time: {thinking_time:.2f} seconds")
        
        print("\nDirect Response:")
        print("-"*80)
        direct_response, direct_time = self.get_response(question, use_thinking=False)
        print(direct_response)
        print(f"\nGeneration time: {direct_time:.2f} seconds")
        
        # Save responses to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{timestamp}.txt"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"Question: {question}\n\n")
            f.write("Thinking Process Response:\n")
            f.write(f"{thinking_response}\n")
            f.write(f"Generation time: {thinking_time:.2f} seconds\n\n")
            f.write("Direct Response:\n")
            f.write(f"{direct_response}\n")
            f.write(f"Generation time: {direct_time:.2f} seconds\n")
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nResponses saved to {filename}")

def main():
    # Test questions that highlight different thinking patterns
    test_questions = [
        "What are the implications of artificial intelligence on job markets?",
        "How can we solve the problem of plastic pollution in oceans?",
        "What makes a good book memorable?",
        "Should humans colonize Mars? Why or why not?",
        "How does social media influence modern relationships?"
    ]
    
    llm = ComparisonLLM()
    
    for question in test_questions:
        llm.compare_responses(question)
        input("\nPress Enter for next question...")

if __name__ == "__main__":
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    main()