import gradio as gr
import time,os
from PIL import Image, ImageDraw, ImageFont
from scripts.datatools_calimgs.coco_label import print_statistics

def txttoimgs(out_dir,delimiter="&msg", image_width=400, image_height=400):
    txt_file_path = "code/questions.txt"
    with open(txt_file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # 分割文本内容
    parts = content.split(delimiter)
    font = "code/YaHei.ttf"
    font_fill = "black"

    # 遍历每个部分并生成图片
    for i, part in enumerate(parts):
        if part not in ["","\n"]:
            delimiter1 = "&nms"
            delimiter2 = "&tit"
            delimiter3 = "&bdy"
            name = part.split(delimiter1)[1]
            title = part.split(delimiter2)[1]
            body = part.split(delimiter3)[1]
            if len(body)>27:
                body = body[0:27]+"\n"+body[27:]
            # 创建空白图片
            image = Image.new("RGB", (image_width, image_height), color="white")
            draw = ImageDraw.Draw(image)
            
            # 在图片上绘制文本内容
            y_offset = 0
            draw.text((10, y_offset), name, fill=font_fill, font=ImageFont.truetype(font, 20))
            draw.text((10, y_offset+30), title, fill=font_fill, font=ImageFont.truetype(font, 15))
            draw.text((10, y_offset+50), body, fill=font_fill, font=ImageFont.truetype(font, 10))
            
            # 保存图片
            output_image_path = f"{out_dir}/image_{i}.png"
            image.save(output_image_path)

def image_path():
    from collections import deque
    image_dir = "code/imgs"
    txttoimgs(image_dir)
    images = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith((".jpg", ".jpeg", ".png"))]

    # dq = deque(images)
    # for i in range(len(images)):
    #     dq.rotate(-1)
    #     return list(dq)
    
    return images

def collect_question(name,title, body):
    if not name == "" and title == "" and body == "":
        # 将标题和内容组合成一个字符串
        question = f"&msg&nms{name}&nms&tit{title}&tit&bdy{body}&bdy&msg\n"
        # 将问题追加到本地文件中
        with open("code/questions.txt", "a",encoding="utf-8") as file:
            file.write(question)
        # 返回成功的消息
        return "提交成功: " + title

def read_questions():
    # 读取本地文件中的所有问题
    with open("code/questions.txt", "r",encoding="utf-8") as file:
        questions = file.readlines()
    # 返回问题列表
    return "".join(questions)

with open("更新日志.md", 'r', encoding='utf-8') as file:
        md_text = file.read()

with gr.Blocks() as demo:
    gr.Gallery(image_path,label="留言画廊", every=1,preview=True)
    with gr.Tab("建议"):
        # output_saved_text = gr.Textbox(value=read_questions,label="留言显示", type="text",lines=5,every=5)
        with gr.Row():
            with gr.Column():
                input_name = gr.Textbox(lines=1, label="名字")
                input_title = gr.Textbox(lines=1, label="标题")
            output_text = gr.Textbox(label="提交提示",lines=5)
        input_body = gr.Textbox(lines=5, label="内容")
        btn_ok = gr.Button("确认")
        btn_ok.click(collect_question,inputs=[input_name,input_title,input_body],outputs=[output_text])
    with gr.Tab("测试"):
        label_radio = gr.Radio([True,False],label="检测项名称")
        input_text = gr.Textbox(label="First")
        model_select = gr.Accordion("模型选择",open=False,visible=False)
        with model_select:
            llm = gr.Dropdown([True,False],label="a")
            embedding = gr.Dropdown([True,False],label="b")
        def update_box(a):
            if not a:
                return gr.Accordion.update(open=True,visible=True)
            else:
                return gr.Accordion.update(open=False,visible=False)

        label_radio.change(fn=update_box,inputs=label_radio,outputs=model_select)


if __name__ == "__main__":
    demo.queue(150).launch()
