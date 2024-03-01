import gradio as gr

def choose_color(color):
    return f"You chose {color}!"

color_radio = gr.Radio(["Red", "Green", "Blue"])

iface = gr.Interface(fn=choose_color, inputs=color_radio, outputs="text")
iface.launch()
