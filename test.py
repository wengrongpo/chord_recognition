import librosa
import numpy as np
import matplotlib.pyplot as plt

from app.preprocessing import preprocess_and_convert_audio
from app.core.chord_templates import chordTemplates
from app.core.recognition import convert_and_merge, detect_chords, detect_silence, detect_transition, mark_merge_regions
#C G AM. EM. F. C. DM. G
input_path="./C.m4a"
output_path="test.wav"

preprocess_and_convert_audio(input_path,output_path)
# 1. 加载音频并提取特征
y, sr = librosa.load("test.wav")

hop_length = 1024  # 必须一致
frame_length = 2048  # 必须一致
num_states = 12  # 假设有 12 个和弦
transition_probs = np.ones((num_states, num_states)) / num_states  # 均匀分布的转移概率
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

# 2. 检测静音和过渡段（使用动态阈值）
silence_frames, auto_threshold = detect_silence(
    y, sr, 
    hop_length=hop_length,  # 与 Chromagram 一致
    frame_length=frame_length,
    percentile=10,   # 可调参数：更小值会更严格（如5）
    offset_db=6      # 可调参数：增大值会减少静音段（如8）
)

transition_frames = detect_transition(chroma)

merge_flags = mark_merge_regions(silence_frames, transition_frames, chroma.shape[1])

# print(f"动态静音阈值: {auto_threshold:.2f} dB")  # 输出实际阈值


# 3. 执行整曲识别
detected_chords = detect_chords(chroma, chordTemplates)
# print("识别结果（时间帧 → 和弦）：", detected_chords)

# 4. 合并处理
merged = convert_and_merge(
    detected_chords,
    hop_length=1024,
    sr=22050,
    aggregate_window=10,
    merge_flags=merge_flags      # 传入需合并的段标记
    # min_duration=0.5           # 过滤短于0.5秒的段
)

# 5. 输出结果
for start, end, chord in merged:
    print(f"{start:.2f}-{end:.2f}s: {chord}")
