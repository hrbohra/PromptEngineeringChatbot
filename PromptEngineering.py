import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import textwrap
import time

class ThinkingLLM:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize with TinyLlama which can run on 16GB VRAM
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token to be different from EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Remove manual device assignment and let accelerate manage it
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision
            device_map="auto"  # Automatically handle device placement (CPU or GPU)
        )
        
    def generate_thinking_prompt(self, question, prompt_type="general"):
        """Create a prompt that encourages step-by-step thinking"""
        prompts = {
            "general": f"""Question: {question}
Let me think about this step by step:
1) First, I need to understand what's being asked
2) Then, I'll break down the key components
3) Finally, I'll provide a clear answer

My thinking process:""",
            
            "analysis": f"""Question: {question}
Let me analyze this systematically:
1) Key facts and context
2) Important relationships and patterns
3) Possible implications
4) Conclusion based on analysis

My analysis:""",
            
            "problem_solving": f"""Problem: {question}
Let me solve this methodically:
1) What are the given elements?
2) What approach should I take?
3) What steps will lead to a solution?
4) Let me verify my answer

My solution process:"""
        }
        return prompts.get(prompt_type, prompts["general"])

    def get_response(self, question, prompt_type="general", max_length=512):
        """Generate a response with visible thinking steps"""
        prompt = self.generate_thinking_prompt(question, prompt_type)
        
        # Tokenize with proper padding and attention mask
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        ).to(self.model.device)  # Let the tokenizer know about the model's device
        
        start_time = time.time()
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        end_time = time.time()
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):]
        return textwrap.fill(response, width=80), end_time - start_time

def run_tests():
    print("Initializing ThinkingLLM (this may take a moment)...")
    llm = ThinkingLLM()
    
    test_cases = [
        # General reasoning tests
        ("general", "What makes a good leader?"),
        ("general", "How does weather affect people's mood?"),
        
        # Analysis tests
        ("analysis", "Compare electric cars vs traditional gasoline vehicles"),
        ("analysis", "What are the impacts of social media on society?"),
        
        # Problem-solving tests
        ("problem_solving", "How would you organize a small community library?"),
        ("problem_solving", "Design a simple recycling system for a small office")
    ]
    
    total_time = 0
    for prompt_type, question in test_cases:
        print("\n" + "="*80)
        print(f"Test Type: {prompt_type}")
        print(f"Question: {question}")
        print("-"*80)
        
        response, generation_time = llm.get_response(question, prompt_type)
        total_time += generation_time
        
        print(response)
        print(f"\nGeneration time: {generation_time:.2f} seconds")
        print("="*80)
        input("\nPress Enter for next test...")
    
    print(f"\nAverage generation time: {total_time/len(test_cases):.2f} seconds")

if __name__ == "__main__":
    run_tests()
