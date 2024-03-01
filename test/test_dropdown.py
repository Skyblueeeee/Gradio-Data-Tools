import gradio as gr
import os


def upload_ori(test_folder_dropdown):
    test_folder_path = r"D:\shy_code\AIGC\Gradio_Tools\scripts"
    return gr.Dropdown.update(choices=os.listdir(os.path.join(test_folder_path)))
 
def upload_specific(test_file_dropdown):
    print(test_file_dropdown)

def add_drop(a):
    print(f"Adding: {a}")
    list1.insert(0,a)
    print(f"Add hou:{list1}")
    return gr.Dropdown.update(choices=list1)

with gr.Blocks() as demo:
    list1 = ["ch", "en"]
    with gr.Tab("PSAA"):
        with gr.Column():
            with gr.Row():
                pp_lang = gr.inputs.Dropdown(choices=list1, label="类型")        
                pp_lang_input = gr.Textbox(label="输入类型", lines=1)
                pp_ok_btn = gr.Button("确认", variant="primary")
                pp_out = gr.Textbox(label="输出")
    pp_ok_btn.click(add_drop, inputs=[pp_lang_input],outputs=pp_lang)
    pp_lang.change(fn=None,inputs=[pp_lang],outputs=[pp_lang])

    # 双dropdown联动
    with gr.Tab("sss"):
        with gr.Row():
            test_folder_dropdown = gr.inputs.Dropdown(choices=["aaa"], label="Choose a folder")
            test_file_dropdown = gr.inputs.Dropdown(choices=["请先选择文件夹"],label="you can Choose a file or write a file name")
        test_folder_dropdown.change(fn=upload_ori,inputs=[test_folder_dropdown],outputs=[ test_file_dropdown]) 
        test_file_dropdown.change(fn=None,inputs=[test_file_dropdown],outputs=[test_file_dropdown])


demo.queue(150).launch(server_port=8003, share=False, show_api=False)
