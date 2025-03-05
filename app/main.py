from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
from .preprocessing import preprocess_audio
from .model import load_model
import tempfile
import os

app = FastAPI()

# 加载模型（假设模型已经训练好并保存）
MODEL_PATH = "chord_model.pth"
model = load_model(MODEL_PATH)

# 和弦标签映射
CHORD_LABELS = ['A', 'Am', 'Bm', 'C', 'D', 'Dm', 'E', 'Em', 'F', 'G']  # 示例和弦列表

@app.post("/predict")
async def predict_chord(file: UploadFile = File(...)):
    # 创建临时文件保存上传的音频
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        
        try:
            # 预处理音频
            features = preprocess_audio(temp_file.name)
            
            # 转换为 PyTorch tensor
            features = torch.FloatTensor(features)
            
            # 进行预测
            with torch.no_grad():
                outputs = model(features)
                predicted = torch.argmax(outputs, dim=1)
            
            # 获取预测的和弦
            predicted_chord = CHORD_LABELS[predicted.item()]
            
            return {"chord": predicted_chord}
            
        finally:
            # 删除临时文件
            os.unlink(temp_file.name) 