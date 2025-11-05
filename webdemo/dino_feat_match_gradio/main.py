import gradio as gr
import numpy as np

with gr.Blocks() as demo:
    table = gr.Dataframe([[1, 2, 3], [4, 5, 6]])
    gallery = gr.Gallery([("cat.jpg", "Cat"), ("dog.jpg", "Dog")])
    textbox = gr.Textbox("Hello World!")
    statement = gr.Textbox()

    def on_select(value, evt: gr.EventData):
        return f"The {evt.target} component was selected, and its value was {value}."

    table.select(on_select, table, statement)
    gallery.select(on_select, gallery, statement)
    textbox.select(on_select, textbox, statement)

demo.launch(show_error=True)
