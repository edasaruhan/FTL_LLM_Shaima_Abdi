import gradio as gr
from transformers import pipeline

# Load the text generation model using Bloom
model_name = "bigscience/bloom-560m"  # Using a smaller version of Bloom for demonstration
text_generator = pipeline("text-generation", model=model_name)

# Define a function to generate text based on a user-provided question
def generate_response(question):
    prompt = f"{question}"
    response = text_generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"]

# Create a Gradio interface with a text input box for the question
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your question related to SDG 13: Climate Action"),
    outputs=gr.Textbox(label="Generated response"),
    title="SDG 13 Text Generation",
    description="Type any question related to Climate Action (SDG 13), and the model will generate a response.",
)

# Launch the Gradio app
demo.launch()
