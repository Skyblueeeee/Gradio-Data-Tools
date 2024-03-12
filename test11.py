import os
import numpy as np
import gradio as gr

from scripts.datatools_calimgs.coco_label import print_statistics
from scripts.datatools_augimgs.COCO_ImageAugWithBox import image_aug,image_dir_aug,check_dir

from scripts.datatools_viajson.via2coco import run_viatococo
from scripts.datatools_viajson.coco2via import main_coco2via
from scripts.datatools_viajson.coco2yolo import coco_to_txt
from scripts.datatools_viajson.yolo_to_via_project import run_yolo2via
from scripts.datatools_viajson.via_get_labels import run_quqian

from scripts.datatools_fiftyone.constructflow import GRConstuctFlow
from scripts.datatools_fiftyone.constructflow import db_tags,send_image
from scripts.datatools_fiftyone.fiftyone_app import fiftyone_start
from scripts.datatools_fiftyone.fiftyone_dir.construct_flow import ConstructFlow

css = "ui/style.css"

class GradioDataTools():
    def __init__(self) -> None:
        self.name_list = ["Gradio-Data-Tools","（ฅᵔ·͈༝·͈ᵔฅ）"]
        self.super = ["fitow","default"]
        self.label_path = "\\\\10.10.1.125\\ai01\\label\\"
        self.refresh_icon_path = "ui/imgs/刷新按钮.png"
        self.md = "README.md"
        self.log_path = "log/gdt_log.log"

        # 加载51
        fiftyone_start()

    def update_title(self,value):
        return gr.HTML.update(f"""<h1 align="center">{value}</h1>""")

    def add_name(self,name):
        if name !="":
            if name not in self.name_list:
                self.name_list.append(name)
        return gr.Dropdown.update(choices=self.name_list)

    def log_md(self):
        with open(self.md, 'r', encoding='utf-8') as file:
            md_text = file.read()

        return md_text

    def open_folder(self,folder_path):
        if os.path.exists(folder_path):
            return gr.Button.update(link=folder_path)

    def image_info(self,display):
        return display

    def _ff_build(self,ff_imagedir_input,ff_input_imgdir_rule,ff_dbname_input,ff_json_input10,ff_json_input1,
                ff_json_input2,ff_json_input3,ff_json_input4,ff_json_input5,
                ff_json_input6,ff_json_input7,ff_json_input8,ff_json_input9,ff_json_input_uniqe):
        if os.path.exists(ff_imagedir_input):
            cf = GRConstuctFlow(ff_dbname_input, back_up=ff_json_input10)
            cf.run( origin_dir= ff_imagedir_input,imgdir_rule=ff_input_imgdir_rule,file_uniqe=ff_json_input_uniqe,
                    init_dateset = ff_json_input1,merge_origin = ff_json_input2,  
                    merge_label  = ff_json_input3,label_attr = ff_json_input4,via_coco_mode=ff_json_input5,
                    to_sql = ff_json_input6, to_sql_dir = ff_json_input9,
                    del_db = ff_json_input7, del_mode = ff_json_input8
            )
            return "Fiftyone数据库任务已完成!"
        else:
            return "路径不存在，请检查！暂只支持数据中心(125/39/184服务器),不支持个人本机数据。"

    def add_super_caty(self,defau):
        if defau not in self.super and defau != "":
            self.super.append(defau)
        return gr.Dropdown.update(choices=self.super)

    def add_warmup(self,num,epoch,bs):
        w1 = int(num) * int(epoch) // int(bs) * 0.05
        w2 = int(num) * int(epoch) // int(bs) * 0.1

        return str(w1)+"---"+str(w2)
        
    def plot_forecast(self,final_year, companies, noise, show_legend, point_style="circle"):
        import matplotlib.pyplot as plt
        start_year = 2020
        x = np.arange(start_year, final_year + 1)
        year_count = x.shape[0]
        plt_format = ({"cross": "X", "line": "-", "circle": "o--"})[point_style]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, company in enumerate(companies):
            series = np.arange(0, year_count, dtype=float)
            series = series**2 * (i + 1)
            series += np.random.rand(year_count) * noise
            ax.plot(x, series, plt_format)
        if show_legend:
            plt.legend(companies)
        return fig

    def update_box(self,a,b):
        if a == False and b == True:
            return gr.Accordion.update(open=True,visible=True)
        else:
            return gr.Accordion.update(open=False,visible=False)
    
    def update_accordion(self,a):
        if a == True:
            return gr.Accordion.update(open=True,visible=True)
        else:
            return gr.Accordion.update(open=False,visible=False)
    
    def update_log(self,a):
        if a:
            with open(self.log_path, 'r') as f:
                return f.read()

    def gr_ui(self):
        with gr.Blocks(theme=gr.themes.Soft(),title="Gradio-Data-Tools",css=css) as demo:
            title = gr.HTML("""<h1 align="center">Gradio-Data-Tools</h1>""")

            with gr.Tab("首页"):
                gr.Markdown(self.log_md())
                with gr.Tab("标题"):
                    with gr.Row():
                        name_output = gr.Dropdown(choices=self.name_list,label="标题选择")
                        name_input = gr.Textbox(label="输入网页名称", lines=1)
                        pp_ok_btn = gr.Button(value='',icon=self.refresh_icon_path,elem_classes='refreshbutton')
                        pp_ok_btn.click(self.add_name, inputs=[name_input], outputs=name_output)
                        name_output.change(fn=self.update_title,inputs=[name_output],outputs=title)
                
            with gr.Tab("数据管理"):
                gr.Markdown("""
                            1.文件夹内图片名称尽量保证唯一性，否则导入失败，需要打开强制去重功能，会出现框与图不符情况!

                            2.由于部署原因，暂只支持数据中心(125/39/184服务器),不支持个人本机数据。
                            """)
                with gr.Column():
                    with gr.Row():
                        ff_imagedir_input = gr.Textbox(self.label_path,label = "输入地址",info="数据与JSON文件所在地址,支持多个文件夹同时录入",scale=5)
                        ff_input_imgdir_rule = gr.Text(label="检索关键词",info="目录下关键词所有数据")
                        ff_build_button = gr.Button("构建数据")
                        gr.Button("51客户端",link='http://10.10.1.129:8003/')

                    with gr.Tab("数据库录入"):
                        with gr.Row():
                            ff_dbname_input = gr.Textbox(label = "数据库名称",info="不可使用中文",scale=1)
                            ff_json_input1 = gr.Radio([True,False],value=False,label = "初始化数据库",info="是否保留过往数据信息",scale=1)
                            ff_json_input10 = gr.Radio([True,False],value=False,label = "备份数据库", info="默认为否，有需要可自行打开",scale=1)
                            ff_json_input2 = gr.Radio([True,False],value=False,label = "图片数据入库",info="只录入图片，无其他信息",scale=1)
                            ff_json_input_uniqe = gr.Radio([True,False],value=False,label = "图片是否去重",info="不建议，录不进去再开",scale=1)
                            ff_json_input5 = gr.Radio(["VIA","COCO"],value="VIA",label = "入库模式",info="选择JSOn文件",scale=1)
                            ff_json_input4 = gr.Radio([True,False],value=False,label = "标注有效性入库",info="区分是否为空图",scale=1)
                            ff_json_input3 = gr.Radio([True,False],value=False,label = "标注框入库",info="是否导入标注信息",scale=1)
                    with gr.Tab("数据库管理"):
                        with gr.Column():
                            with gr.Row():
                                ff_json_sqlname = gr.Dropdown(label="数据库名称",scale=1)
                                ff_json_sqlnamef5 = gr.Button("获取",icon=self.refresh_icon_path,elem_classes='refreshbutton',scale=1)
                                ff_json_input9 = gr.Textbox(self.label_path,label="导出地址",scale=7)
                            with gr.Row():
                                ff_json_input6 = gr.Radio([True,False],value=False,label = "导出数据库",info="导出.db文件本地查看,不填则默认地址",scale=2)
                                ff_btn_db = gr.Button("导出",scale=1)
                                ff_json_input7 = gr.Radio([True,False],value=False,label = "删除数据库",info="可以选择初始化数据库覆盖，或者选择删除",scale=2)
                                ff_json_input8 = gr.Radio(["equal","in","all"],value="equal",label = "删除模式",info="equal删除全称 in删除包含 all删除全部",scale=2)
                                ff_btn_del = gr.Button("删除",scale=1)
                        ff_json_sqlnamef5.click(ConstructFlow.search_datasets,inputs=ff_json_input8,outputs=ff_json_sqlname)
                        ff_json_sqlname.change(None,inputs=[ff_json_sqlname],outputs=[ff_json_sqlname])
                    with gr.Tab("数据库发图"):
                        with gr.Column():
                            with gr.Row():
                                ff_json_sqlname1 = gr.Dropdown(label="数据库名称",scale=1)
                                ff_json_sqlnamef51 = gr.Button("获取",icon=self.refresh_icon_path,elem_classes='refreshbutton',scale=1)
                                ff_json_input15 = gr.Dropdown(label="数据库标签",scale=3)
                                ff_json_input13 = gr.Textbox("",label="发图日期",scale=2)
                                ff_json_input_options = gr.Radio([True,False],value=False,label="添加模板",scale=1)
                            ff_json_input_options_Acc = gr.Accordion("VIA文件标注模板(中文可不填)",open=False,visible=False)
                            with ff_json_input_options_Acc:
                                ff_json_via_df = gr.Dataframe(headers=["算法标签", "中文名称"],datatype=["str", "str"],row_count=5,col_count=(2, "fixed"))
                            with gr.Row():
                                ff_json_input12 = gr.Textbox(self.label_path,label="发图地址",scale=6)
                                ff_json_input14 = gr.Button("发图",scale=1)

                ff_info_output = gr.Textbox(label = "输出信息",lines=10,max_lines=10)  

                ff_json_sqlnamef51.click(ConstructFlow.search_datasets,inputs=ff_json_input8,outputs=ff_json_sqlname1)
                ff_json_sqlname1.change(fn = db_tags,inputs=[ff_json_sqlname1],outputs=[ff_json_input15,ff_json_input13])
                ff_json_input15.change(None,inputs=[ff_json_input15],outputs=[ff_json_input15])

                ff_json_input14.click(send_image,inputs=[ff_json_sqlname1,ff_json_input15,ff_json_input12,ff_json_input13,ff_json_via_df],outputs=ff_info_output)
                ff_build_button.click(self._ff_build,inputs=[ff_imagedir_input,ff_input_imgdir_rule,ff_dbname_input,ff_json_input10,ff_json_input1,
                                                            ff_json_input2,ff_json_input3,ff_json_input4,ff_json_input5,
                                                            ff_json_input6,ff_json_input7,ff_json_input8,ff_json_input9,ff_json_input_uniqe],outputs=ff_info_output)
                ff_btn_del.click(self._ff_build,inputs=[ff_imagedir_input,ff_input_imgdir_rule,ff_json_sqlname,ff_json_input10,ff_json_input1,
                                                            ff_json_input2,ff_json_input3,ff_json_input4,ff_json_input5,
                                                            ff_json_input6,ff_json_input7,ff_json_input8,ff_json_input9,ff_json_input_uniqe],outputs=ff_info_output)
                ff_btn_db.click(self._ff_build,inputs=[ff_imagedir_input,ff_input_imgdir_rule,ff_json_sqlname,ff_json_input10,ff_json_input1,
                                                            ff_json_input2,ff_json_input3,ff_json_input4,ff_json_input5,
                                                            ff_json_input6,ff_json_input7,ff_json_input8,ff_json_input9,ff_json_input_uniqe],outputs=ff_info_output)
                ff_json_input_options.change(self.update_accordion,inputs=ff_json_input_options,outputs=ff_json_input_options_Acc)

            with gr.Tab("数据分析"):
                gr.Markdown("""
                            1.注意避免文件夹内多个coco文件,筛选条件为含有'coco'字符,结尾为'json'。
                            
                            2.由于部署原因，暂只支持数据中心(125/39/184服务器),不支持个人本机数据。
                            """)
                with gr.Tab("COCO标签"): 
                    with gr.Column():
                        with gr.Row():
                            cal_input1 = gr.Textbox(self.label_path,label="数据地址",info="需coco文件",scale=5)
                            cal_input_imgdir_rule = gr.Text(label="检索关键词",info="目录下关键词所有数据")
                            cal_ok_btm = gr.Button("确认")
                        # cal_info = gr.Textbox("",label="输出信息",lines=20,max_lines=20)
                        cal_info_df = gr.Dataframe(headers=["算法标签", "标签数量", "图片数量","平均占比"],datatype=["str", "str", "str","str"],row_count=10,col_count=(4, "fixed"))
                    with gr.Tab("标签条形图"):
                        gr_bar_plot = gr.BarPlot(title="标签条形图",x_title="Label",y_title="Number",height=400,width=1400)
                    with gr.Tab("标签散点图"):
                        gr_scatter_plot = gr.ScatterPlot(title="标签散点图",x_title="Label",y_title="Number",height=400,width=1400)
                    with gr.Tab("标签折线图"):
                        gr_line_plot = gr.LinePlot(title="标签折线图",x_title="Label",y_title="Number",height=400,width=1400)

                with gr.Tab("参数进阶"): 
                        with gr.Row():
                            imgs_num = gr.Textbox("",label="imgs_num",scale=1)
                            epoch = gr.Textbox("30",label="epoch",scale=1)
                            bs = gr.Textbox("1",label="batch_size",scale=1)
                            warmup = gr.Textbox("",label="warmup",scale=2)
                            ok_btn = gr.Button("确认",scale=1)

                cal_ok_btm.click(print_statistics,inputs=[cal_input1,cal_input_imgdir_rule],outputs=[cal_info_df,imgs_num,gr_scatter_plot,gr_line_plot,gr_bar_plot])
                ok_btn.click(self.add_warmup,inputs=[imgs_num,epoch,bs],outputs=warmup)
            
            with gr.Tab("数据增强"):
                gr.Markdown("""
                            1.暂时只支持XY轴偏移与明亮度调整。

                            2.由于部署原因，暂只支持数据中心(125/39/184服务器),不支持个人本机数据。
                            """)
                with gr.Column():
                    with gr.Row():
                        x1 = gr.Slider(minimum=-1, maximum=1, value=-0.05, interactive=True, label="偏移X1")
                        x2 = gr.Slider(minimum=-1, maximum=1, value=0.05, interactive=True,label="偏移X2")
                    with gr.Row():
                        y1 = gr.Slider(minimum=-1, maximum=1, value=-0.05, interactive=True,label="偏移Y1")
                        y2 = gr.Slider(minimum=-1, maximum=1, value=0.05, interactive=True,label="偏移Y2")
                    with gr.Row():
                        ml1 = gr.Slider(minimum=-1, maximum=1, value=0.8, interactive=True,label="明亮度1")
                        ml2 = gr.Slider(minimum=0, maximum=2, value=1.2, interactive=True,label="明亮度2")

                    flip = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, interactive=True,label="翻转")
                    count = gr.Slider(minimum=1, maximum=50, value=1, step=1, interactive=True,label="增强次数")
                with gr.Tab("单张数据增强"):
                    with gr.Row():
                        image_input1 = gr.Image(label="原图")
                        image_output1 = gr.Gallery(label="增强后的数据",columns=4)
                    image_button1 = gr.Button("开始增强",variant="primary")
                    image_button1.click(image_aug, inputs=[image_input1,x1,x2,y1,y2,ml1,ml2,count,flip], outputs=image_output1)

                with gr.Tab("多张数据增强"):
                    with gr.Column():
                        with gr.Row():
                            image_input2 = gr.Textbox(self.label_path,label="待增强数据文件夹路径",info="目录要求:'/文件A/文件B/coco.py'  输入'/文件A'即可")
                            info_output = gr.Textbox("",label="提示信息",lines=2,max_lines=2)
                        image_radio = gr.Radio(["否","是"],value="否",label="预览增强图片",info="默认选择'否',选择'是'非常耗时,不建议打开")
                        image_output2 = gr.Gallery(label="增强后的数据",columns=4)
                        with gr.Row():
                            image_button3 = gr.Button("校验增强目录",variant="primary")
                            image_button4 = gr.Button("还没想好",variant="primary",link=image_input2.value)
                        image_button2 = gr.Button("开始增强",variant="primary")

                with gr.Accordion("注意事项"):
                    gr.Markdown("需要目录中有coco.json文件")

                image_button2.click(image_dir_aug, inputs=[image_input2,x1,x2,y1,y2,ml1,ml2,count,image_radio,flip], outputs=image_output2)
                image_button3.click(check_dir, inputs=image_input2,outputs=info_output)
                # image_button4.click(None)
            
            with gr.Tab("提取标签"):
                gr.Markdown('''
                            1.默认使用via_project.json取标签。

                            2.由于部署原因，暂只支持数据中心(125/39/184服务器),不支持个人本机数据。
                            ''')
                with gr.Column():
                    with gr.Row():
                        get_imgs_input1 = gr.Textbox(self.label_path,label="数据地址",scale=5)
                        get_imgs_input7 = gr.Textbox("via_project",label="JSON名称",scale=1)
                    with gr.Row():
                        get_imgs_input2 = gr.Textbox(self.label_path,label="保存地址",scale=5)
                        get_imgs_input6 = gr.Dropdown(choices=self.super,value=self.super[0],label = "模板名称")
                        super_text = gr.Textbox(label="添加模板")
                        super_btn = gr.Button("添加",icon=self.refresh_icon_path,elem_classes='refreshbutton')
                        super_btn.click(self.add_super_caty,inputs=[super_text],outputs=[get_imgs_input6])
                        get_imgs_input6.change(fn=None,inputs=[get_imgs_input6],outputs=[get_imgs_input6])
                    with gr.Row():
                        get_imgs_input3 = gr.Textbox("",label = "标签名称",info="多标签可使用空格分隔开",scale=6,lines=7,max_lines=7)
                        with gr.Column(min_width=190):
                            get_imgs_input4 = gr.Radio([True,False],value = True, label = "其他标签保留",scale=1)
                            get_imgs_input8 = gr.Radio([True,False],value = True, label = "原来模板保留",scale=1)

                        with gr.Column(min_width=190):
                            get_imgs_input9 = gr.Radio([True,False],value = True, label = "图片是否复制",scale=1)
                            get_imgs_input5 = gr.Radio([True,False],value = True, label = "转换COCO",scale=1)
                        img_info = gr.Accordion("图片宽高",open=False,visible=False)
                        with img_info:
                            get_imgs_input10 = gr.Textbox("",label="width",scale=1)
                            get_imgs_input11 = gr.Textbox("",label="height",scale=1)
                    get_imgs_info = gr.Textbox("",label="输出信息",lines=10,max_lines=10)
                    get_imgs_btm = gr.Button("开始取图",variant="primary")

                get_imgs_input9.change(fn=self.update_box,inputs=[get_imgs_input9,get_imgs_input5],outputs=img_info)
                get_imgs_input5.change(fn=self.update_box,inputs=[get_imgs_input9,get_imgs_input5],outputs=img_info)
                get_imgs_btm.click(run_quqian,inputs=[get_imgs_input1,get_imgs_input7,get_imgs_input6,get_imgs_input2,get_imgs_input3,get_imgs_input4,get_imgs_input5,get_imgs_input8,get_imgs_input9,get_imgs_input10,get_imgs_input11],outputs=get_imgs_info)

            with gr.Tab("JSON互转"):
                gr.Markdown("""
                            1.请先确认json文件是否存在，注意名称不需要填写‘.json’。

                            2.由于部署原因，暂只支持数据中心(125/39/184服务器),不支持个人本机数据。
                            """)
                with gr.Tab("via2coco"):
                    with gr.Column():
                        with gr.Row():
                            json_dir_input = gr.Textbox(self.label_path,label = "输入地址",info="生成的coco格式可支持mmdet训练",max_lines=7,lines=7)
                            with gr.Column():
                                with gr.Row():
                                    json_vianame_input1 = gr.Textbox("via_project",label = "project名称",info="不需要输入'.json")
                                    json_coconame_input1 = gr.Textbox("coco",label = "COCO名称",info="不需要输入'.json'")
                                with gr.Row():
                                    json_supercategory_input1 = gr.Dropdown(choices=self.super,value=self.super[0],label = "模板名称")
                                    super_text = gr.Textbox(label="添加模板")
                                    super_btn = gr.Button("添加",icon=self.refresh_icon_path)
                            
                        json_info_output = gr.Textbox(label = "输出信息",lines=15,max_lines=15)
                        json_vc_ok_button = gr.Button("确认",variant="primary")
                    super_btn.click(self.add_super_caty,inputs=[super_text],outputs=[json_supercategory_input1])
                    json_supercategory_input1.change(fn=None,inputs=[json_supercategory_input1],outputs=[json_supercategory_input1])
                json_vc_ok_button.click(run_viatococo,inputs=[json_dir_input,json_vianame_input1,json_coconame_input1,json_supercategory_input1],outputs=json_info_output)

                with gr.Tab("coco2via"):
                    with gr.Column():
                        with gr.Row():
                            json_dir_input = gr.Textbox(self.label_path,label = "输入地址",info="输入图片地址，内有json文件，生成后的via文件可通过html打开",scale=5)
                            json_coconame_input2 = gr.Textbox("coco",label = "COCO名称",info="不需要输入'.json'",scale=1)
                            json_vianame_input2 = gr.Textbox("via_project",label = "via名称",info="不需要输入'.json'",scale=1)
                            json_supercategory_input2 = gr.Dropdown(["fitow","default"],label = "模板名称",info="不输入,默认为fitow",scale=1)
                        json_info_output = gr.Textbox(label = "输出信息",lines=15,max_lines=15)
                        json_cv_ok_button = gr.Button("确认",variant="primary")
                json_cv_ok_button.click(main_coco2via,inputs=[json_dir_input,json_coconame_input2,json_vianame_input2,json_supercategory_input2],outputs=json_info_output)

                with gr.Tab("coco2yolo"):
                    with gr.Column():
                        with gr.Row():
                            json_dir_input = gr.Textbox(self.label_path,label = "输入地址",info="输入图片地址，内有coco文件",scale=5)
                            json_coconame_input2 = gr.Textbox("via_export_coco",label = "COCO名称",info="不需要输入'.json'",scale=1)
                        json_info_output = gr.Textbox(label = "输出信息",lines=15,max_lines=15)
                        json_cy_ok_button = gr.Button("确认",variant="primary")
                json_cy_ok_button.click(coco_to_txt,inputs=[json_dir_input,json_coconame_input2],outputs=json_info_output)
               
                with gr.Tab("yolo2via"):
                    with gr.Row():
                        json_yc_input_text = gr.Text(self.label_path,label="数据地址",scale=5)
                        json_yc_input_vianame = gr.Text("via_project",label="VIA名称",scale=1)
                        json_yc_input_super = gr.Dropdown(choices=self.super,value=self.super[0],label = "模板名称",scale=1)
                        super_yc_text = gr.Textbox(label="添加模板",scale=1)
                        superyc__btn = gr.Button("添加",icon=self.refresh_icon_path,scale=1)
                    json_yc_input_df = gr.DataFrame(label="标注模板",headers=["算法标签", "中文名称"],datatype=["str", "str"],row_count=5,col_count=(2, "fixed"))
                    json_yc_info_output = gr.Textbox(label = "输出信息",lines=5,max_lines=5)
                    json_yc_ok_button = gr.Button("确认",variant="primary")
                json_yc_ok_button.click(run_yolo2via,inputs=[json_yc_input_text,json_yc_input_vianame,json_yc_input_super,json_yc_input_df],outputs=json_yc_info_output)

            with gr.Tab("其他"):
                log_radio = gr.Radio([True,False],value=False,label="日志查看")
                log_txt_acc = gr.Accordion("",open=False,visible=False)
                with log_txt_acc:
                    log_txt = gr.Text("",label="终端显示",max_lines=30,lines=30)

                log_radio.change(fn=self.update_accordion,inputs=log_radio,outputs=log_txt_acc)
                log_radio.change(fn=self.update_log,inputs=log_radio,outputs=log_txt,every=1)
        return demo

if __name__ == "__main__":
    gt = GradioDataTools()
    demo = gt.gr_ui()
    demo.queue(150).launch(server_name='10.10.1.129',server_port=8000,share=False,show_api=False)