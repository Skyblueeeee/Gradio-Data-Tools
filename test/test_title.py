import gradio as gr


css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}
"""


name_list = ["（ฅᵔ·͈༝·͈ᵔฅ）","数据处理平台"]

def update_title(value):

    return gr.HTML.update(f"""<h1 align="center">{value}</h1>""")

def add_name(name):
    if name !="":
        name_list.append(name)

    return gr.Dropdown.update(choices=name_list)

with gr.Blocks() as demo:
    title = gr.HTML("""<h1 align="center">（ฅᵔ·͈༝·͈ᵔฅ）</h1>""")
    
    with gr.Tab("给它起个名"):
        with gr.Row():
            with gr.Column():
                name_output = gr.Dropdown(choices=name_list,label="标题选择")
            with gr.Column():
                name_input = gr.Textbox(label="输入奇思妙想", lines=1)
                pp_ok_btn = gr.Button("确认添加", variant="primary")

            box1 = gr.Textbox(value="Good Job", elem_classes="feedback")
            box2 = gr.Textbox(value="Failure", elem_id="warning", elem_classes="feedback")
    pp_ok_btn.click(add_name, inputs=[name_input], outputs=name_output)
    name_output.change(fn=update_title,inputs=[name_output],outputs=title)

demo.queue(150).launch(server_port=8003, share=False, show_api=False)
