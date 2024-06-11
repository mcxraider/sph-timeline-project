import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Define your Gradio interface
def my_function(input_text):
    return "Output: " + input_text

iface = gr.Interface(fn=my_function, inputs="text", outputs="text")

# Create a FastAPI app
app = FastAPI()

# Serve the Gradio interface at the root path
@app.get("/")
async def main():
    return HTMLResponse(iface.launch(inline=True, share=False, prevent_thread_lock=True))

# Alternatively, create an API endpoint that launches the Gradio app
@app.get("/gradio")
async def gradio_app():
    return HTMLResponse(iface.launch(inline=True, share=False, prevent_thread_lock=True))

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
