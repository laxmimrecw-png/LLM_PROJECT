import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "tinyllama_adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model)

base = AutoModelForCausalLM.from_pretrained(base_model)

model = PeftModel.from_pretrained(base, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


def answer(question):
    prompt = f"""### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("### Answer:")[-1].strip()


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a real estate question..."
    ),
    outputs=gr.Textbox(lines=6),
    title="🏠 Real Estate Chatbot",
    description="Ask any real estate question."
)

demo.launch()