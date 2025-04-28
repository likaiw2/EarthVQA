import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
# from earthvqa.model import EarthVQAModel  # 修改为你模型的实际路径
# from utils import run_segmentation, run_vqa  # 修改为你实际的推理函数路径
import tempfile
import os
from pred_seg import predict_seg
from pred_soba import predict_soba
import argparse
from PIL import ImageDraw, ImageFont
import sys

COLOR_MAP = {
    0: (0, 0, 0),              # nothing
    1: (255, 255, 255),        # Background
    2: (255, 0, 0),            # Building
    3: (255, 255, 0),          # Road
    4: (0, 0, 255),            # Water
    5: (159, 129, 183),        # Barren
    6: (0, 255, 0),            # Forest
    7: (255, 195, 128),        # Agricultural
    8: (165, 0, 165),          # Playground
    9: (0, 185, 246),          # Pond
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', 
                    # default='./log/deeplabv3p.pth',
                    default='/home/liw324/code/Segment/EarthVQA/weights/sfpnr50.pth'
                    )
parser.add_argument('--config_path',  type=str,
                    help='config path', 
                    # default='sfpnr50',
                    # default='/home/liw324/code/Segment/EarthVQA/configs/sfpnr50.py',
                    # default='configs/sfpnr50.py',
                    )
parser.add_argument('--save_dir',  type=str,
                    help='save dir', 
                    default='/home/liw324/code/Segment/EarthVQA/#EarthVQA_demo/out'
                    )
parser.add_argument('--image_path', 
                    type=str, 
                    help='Path to input image',
                    default='/home/liw324/code/Segment/EarthVQA/dataset/EarthVQA/Test/4193.png'
                    )

args = parser.parse_args()

def create_legend(color_map, box_size=(30, 30), font_size=20):
    """生成图例图像"""
    num_classes = len(color_map)
    img_width = 300
    img_height = box_size[1] * num_classes

    legend_img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(legend_img)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    label_names = {
        0: "Background",
        1: "Background",
        2: "Building",
        3: "Road",
        4: "Water",
        5: "Barren",
        6: "Forest",
        7: "Agricultural",
        8: "Playground",
        9: "Pond",
    }

    for idx, (cls_id, color) in enumerate(color_map.items()):
        y = idx * box_size[1]
        # 画色块
        draw.rectangle([5, y+5, 5+box_size[0], y+5+box_size[1]-10], fill=color)
        # 写标签
        text = label_names.get(cls_id, f"Class {cls_id}")
        draw.text((box_size[0] + 15, y + 5), text, fill=(0, 0, 0), font=font)

    return legend_img

st.set_page_config(page_title="EarthVQA Demo", layout="wide")

col1, col2, col3 = st.columns([3, 1, 3])  # 中间一个窄列
with col1:
    st.title("🌍 EarthVQA Demo")
    st.caption("Upload a remote sensing image to start segmentation and Q&A")
with col3:
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # 保存上传图像到临时文件
    with tempfile.NamedTemporaryFile(delete=True, suffix=".png", dir=args.save_dir) as temp_img:
        tmp_image_path = temp_img
        image.save(temp_img.name)
        
        # 执行语义分割
        with st.spinner("Segmenting..."):
            seg_result = predict_seg(args.ckpt_path, 
                                 'configs/sfpnr50.py', 
                                 args.save_dir,
                                 image_path=temp_img.name)
    
    # 横向并排显示原图和分割图
    col1, col2, col3 = st.columns([2, 2, 1])  # 左中右三栏，第三栏稍窄
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(seg_result, caption="Segmentation Output", use_container_width=True)
    with col3:
        legend_img = create_legend(COLOR_MAP)
        st.image(legend_img, caption="Legend", use_container_width=False)
    
    # 对话框模块放到下方
    st.subheader("💬 Answer")
    question = st.text_input("Your Question")
    hdf_path = tmp_image_path.name.replace('.png', '.hdf5')
    if question:
        with st.spinner("Thinking..."):
            answer = predict_soba(hdf_path, question)
            st.success(f"The output of EarthVQA: {answer}")