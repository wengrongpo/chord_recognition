import torch
import torch.nn as nn

class ChordClassifier(nn.Module):
    def __init__(self, num_chords=24):  # 假设我们识别24种和弦
        super(ChordClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_chords)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path):
    """
    加载训练好的模型
    """
    model = ChordClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model 