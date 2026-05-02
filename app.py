import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # 第二层
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # 第三层
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.3)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv_layers(x)
        x = x.view(-1, 256)
        x = self.fc_layers(x)
        return x

model = CNN()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

def process_sketchpad_data(sketch_data):
    if sketch_data is None:
        return None, "sketch_data is None"
    
    img_array = None
    
    if isinstance(sketch_data, np.ndarray):
        img_array = sketch_data
        
    elif isinstance(sketch_data, dict):
        if 'composite' in sketch_data and sketch_data['composite'] is not None:
            img_array = sketch_data['composite']
        elif 'image' in sketch_data and sketch_data['image'] is not None:
            img_array = sketch_data['image']
        elif 'mask' in sketch_data and sketch_data['mask'] is not None:
            img_array = sketch_data['mask']
        elif 'canvas' in sketch_data and sketch_data['canvas'] is not None:
            img_array = sketch_data['canvas']
        else:
            return None, "No valid data in dict"
            
    elif isinstance(sketch_data, tuple) and len(sketch_data) == 2:
        canvas, mask = sketch_data
        if mask is not None and isinstance(mask, np.ndarray):
            img_array = mask
        elif canvas is not None and isinstance(canvas, np.ndarray):
            img_array = canvas
        else:
            return None, "Invalid canvas/mask data"
            
    elif isinstance(sketch_data, Image.Image):
        img_array = np.array(sketch_data)
        
    else:
        return None, f"Unknown data type: {type(sketch_data)}"
    
    if img_array is None:
        return None, "img_array is None"
    
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.mean(axis=2)
    
    img_array = (img_array).astype(np.uint8)
    pil_image = Image.fromarray(img_array)
    pil_image = pil_image.convert('L')
    pil_image = pil_image.resize((28, 28), Image.LANCZOS)
    
    img_np = np.array(pil_image)
    
    if img_np.max() == img_np.min():
        return None, "Empty sketch (all pixels same)"
        
    img_np = img_np.astype(np.float32) / 255.0
    img_np = 1.0 - img_np
    
    img_tensor = torch.from_numpy(img_np).float()
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, "Success"

def process_uploaded_image(image):
    if image is None:
        return None, "image is None"
        
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(np.uint8(image))
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        return None, f"Unknown type: {type(image)}"
    
    pil_image = pil_image.convert('L')
    pil_image = pil_image.resize((28, 28), Image.LANCZOS)
    img_np = np.array(pil_image)
    img_np = 255 - img_np
    img_np = img_np.astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(img_np).float()
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, "Success"

def predict_from_tensor(img_tensor):
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0)
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted].item() * 100

        top3_probs, top3_indices = torch.topk(probabilities, 3)
        top3_results = [(int(top3_indices[0][i].item()), float(top3_probs[0][i].item()) * 100) for i in range(3)]

    return predicted, confidence, probabilities[0].cpu().numpy(), top3_results

def plot_probability(probabilities):
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(range(10), probabilities, color='#1f77b4')
    ax.set_xticks(range(10))
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.set_ylabel('Probability')
    ax.set_title('Digit Probability Distribution')
    ax.set_ylim(0, 1)

    max_idx = np.argmax(probabilities)
    bars[max_idx].set_color('#ff7f0e')

    plt.tight_layout()
    return fig

history = []

def predict_from_sketch(sketch_data):
    global history
    
    if sketch_data is None:
        return None, None, None, None, "请在画板上书写数字", history
    
    img_tensor, status = process_sketchpad_data(sketch_data)
    
    if img_tensor is None:
        return "预处理失败", "N/A", None, "N/A", f"错误: {status}", history
    
    predicted, confidence, probabilities, top3 = predict_from_tensor(img_tensor)

    history.insert(0, {'数字': str(predicted), '置信度': f'{confidence:.2f}%'})
    if len(history) > 5:
        history.pop()

    prob_plot = plot_probability(probabilities)
    top3_text = "\n".join([f"{digit}: {conf:.2f}%" for digit, conf in top3])

    return f"预测数字: {predicted}", f"置信度: {confidence:.2f}%", prob_plot, top3_text, "处理成功", history

def predict_from_upload(image):
    global history
    
    if image is None:
        return None, None, None, None, "请上传图片", history
    
    img_tensor, status = process_uploaded_image(image)
    
    if img_tensor is None:
        return "预处理失败", "N/A", None, "N/A", f"错误: {status}", history
    
    predicted, confidence, probabilities, top3 = predict_from_tensor(img_tensor)

    history.insert(0, {'数字': str(predicted), '置信度': f'{confidence:.2f}%'})
    if len(history) > 5:
        history.pop()

    prob_plot = plot_probability(probabilities)
    top3_text = "\n".join([f"{digit}: {conf:.2f}%" for digit, conf in top3])

    return f"预测数字: {predicted}", f"置信度: {confidence:.2f}%", prob_plot, top3_text, "处理成功", history

def clear_all():
    global history
    history = []
    return None, None, None, None, None, "已清空", history

with gr.Blocks(title="手写数字识别系统") as demo:
    gr.Markdown("# 🖍️ 手写数字识别系统")
    gr.Markdown("在画板上手写数字或上传图片进行识别！")

    with gr.TabItem("✏️ 手写画板"):
        with gr.Row():
            with gr.Column():
                sketchpad = gr.Sketchpad(label="手写区域", height=280, width=280)
                with gr.Row():
                    clear_btn = gr.Button("清空画板", variant="secondary")
                    sketch_btn = gr.Button("识别", variant="primary")

            with gr.Column():
                sketch_result = gr.Label(label="识别结果")
                sketch_confidence = gr.Textbox(label="置信度")
                with gr.Row():
                    sketch_plot = gr.Plot(label="概率分布")
                    sketch_top3 = gr.Textbox(label="Top-3 预测")
                sketch_status = gr.Textbox(label="状态")

    with gr.TabItem("📤 上传图片"):
        with gr.Row():
            with gr.Column():
                upload_image = gr.Image(label="上传手写数字图片", height=280, width=280)
                upload_btn = gr.Button("识别", variant="primary")

            with gr.Column():
                upload_result = gr.Label(label="识别结果")
                upload_confidence = gr.Textbox(label="置信度")
                with gr.Row():
                    upload_plot = gr.Plot(label="概率分布")
                    upload_top3 = gr.Textbox(label="Top-3 预测")
                upload_status = gr.Textbox(label="状态")

    gr.Markdown("### 📜 历史识别记录")
    history_list = gr.List(label="最近识别", value=[])

    sketch_btn.click(
        fn=predict_from_sketch,
        inputs=sketchpad,
        outputs=[sketch_result, sketch_confidence, sketch_plot, sketch_top3, sketch_status, history_list]
    )

    upload_btn.click(
        fn=predict_from_upload,
        inputs=upload_image,
        outputs=[upload_result, upload_confidence, upload_plot, upload_top3, upload_status, history_list]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[sketchpad, sketch_result, sketch_confidence, sketch_plot, sketch_top3, sketch_status, history_list]
    )

    gr.Markdown("---")
    gr.Markdown("### 💡 使用提示")
    gr.Markdown("- **手写画板**：使用鼠标在画板上书写数字（建议使用较粗笔触）")
    gr.Markdown("- **上传图片**：上传白底黑字的手写数字图片")
    gr.Markdown("- 数字尽量居中，避免超出边界")

if __name__ == '__main__':
    import os
    PORT = int(os.environ.get("PORT", 7870))
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=True)
