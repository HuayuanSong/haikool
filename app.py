import os
import gradio as gr
from huggingface_hub import InferenceClient

class HaikuGenerator:
    def __init__(self):
        HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
        self.text_client = InferenceClient(token=HUGGINGFACE_API_TOKEN, model="HuggingFaceH4/zephyr-7b-beta")
        self.image_client = InferenceClient(token=HUGGINGFACE_API_TOKEN, model="stabilityai/stable-diffusion-xl-base-1.0")

    def generate_haiku(self, prompt):
        system_message = "You are a Haiku generator."
        response = ""
        
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}]
        
        # Get the chat completion response
        message = self.text_client.chat_completion(
            messages,
            max_tokens=30,
            stream=False,
            temperature=0.7,
            top_p=0.95,
        )
        
        # Extract the generated content
        response = message['choices'][0]['message']['content']
        
        return response.strip()

    def text_to_image(self, prompt, style):
        # Modify the prompt based on the selected style
        if style == "Japanese":
            prompt += ", in Japanese art style"
        elif style == "oil painting":
            prompt += ", in the style of an oil painting"
        
        image = self.image_client.text_to_image(prompt)
        return image

    def gradio_interface(self):
        # Custom CSS to apply a Japanese-style font
        custom_css = """
            body {
                font-family: 'Sawarabi Mincho', serif !important;
            }
            h1, h6 {
                font-family: 'Sawarabi Mincho', serif !important;
            }
            .button {
                font-family: 'Sawarabi Mincho', serif !important;
            }
        """

        with gr.Blocks(theme='earneleh/paris', css=custom_css) as demo:
            gr.HTML("""
                <center><h1 style="color:#C73E3A">HaiKool - Haiku Poem and Image Generator</h1></center>""")
            gr.HTML("""
                <center><h6 style="color:#C73E3A">Generate a Haiku poem and an image based on it - Please note that loading time can be up to 1 minute.</h6></center>""")
            
            with gr.Column(elem_id="col-container"):
                haiku_output = gr.Textbox(label="Generated Haiku", interactive=False)
                image_output = gr.Image()
            
            with gr.Row(elem_id="col-container"):
                with gr.Column():
                    prompt = gr.Textbox(show_label=False, placeholder="Enter a prompt for the Haiku")
                with gr.Column():
                    style = gr.Dropdown(label="Select Image Style", choices=["default", "Japanese", "oil painting"], value="default")
                with gr.Column():
                    generate_button = gr.Button("Generate Haiku and Image", elem_classes="button")

            # Define the function that integrates both steps: Haiku generation and image creation
            def generate_haiku_and_image(prompt, style):
                haiku = self.generate_haiku(prompt)
                image = self.text_to_image(haiku, style)
                return haiku, image

            generate_button.click(generate_haiku_and_image, inputs=[prompt, style], outputs=[haiku_output, image_output])
            prompt.submit(generate_haiku_and_image, inputs=[prompt, style], outputs=[haiku_output, image_output])

        demo.launch(debug=True)

if __name__ == "__main__":
    haiku_generator = HaikuGenerator()
    haiku_generator.gradio_interface()
