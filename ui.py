import gradio as gr
from rag import build_vector_store


def chat(message, history):
    from rag import conversation_chain

    # Here you would normally call your model to get a response
    # For this example, we'll just echo the message back
    if conversation_chain is None:
        gr.Warning("Please upload a file first.")
        return "Please upload a file first."
    response = conversation_chain.invoke({"question": message})
    return response["answer"]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            file = gr.File(
                file_count="directory",
                label="Upload a file",
            )
            file.upload(build_vector_store, inputs=[file])
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(chat)

demo.launch(share=True)
