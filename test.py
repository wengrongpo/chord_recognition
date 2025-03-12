import librosa
import numpy as np
import matplotlib.pyplot as plt

from app.preprocessing import preprocess_and_convert_audio
from app.core.chord_templates import chordTemplates
from app.core.recognition import convert_frames_to_seconds, detect_chords

input_path="./special_person.mp3"
output_path="test.wav"

preprocess_and_convert_audio(input_path,output_path)
y, sr = librosa.load("test.wav")

# 和弦模板字典：{和弦名称: 音级掩码（0/1表示是否属于该和弦）}




 # 将根音移到第一个位置
# 使用C大调模板匹配转调后的Chroma

# 使用 Viterbi 解码
# 假设有一些隐状态和转移概率，你需要根据实际情况调整这些值
# 例如，设定一些状态数和概率
num_states = 12  # 假设有 12 个和弦
transition_probs = np.ones((num_states, num_states)) / num_states  # 均匀分布的转移概率
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
# chords = librosa.sequence.viterbi(chroma.T,transition_probs)
# 示例：处理单帧 Chroma 特征
# chroma_frame = chroma[:, 100]  # 取第100帧的Chroma向量
# chord, confidence = match_chord(chroma_frame, chord_templates)
# print(f"检测到和弦：{chord}，置信度：{confidence:.2f}")


# 执行整曲识别
detected_chords = detect_chords(chroma, chordTemplates)
# print("识别结果（时间帧 → 和弦）：", detected_chords)

# 3. 转换为秒并合并
merged = convert_frames_to_seconds(detected_chords, hop_length=1024, sr=22050, aggregate_window=10)

# 4. 输出结果
for start, end, chord in merged:
    print(f"{start:.2f}-{end:.2f}s: {chord}")

# root_index = detect_key(chroma)
# shifted_chroma = np.roll(chroma, -root_index, axis=0) 



# plt.figure(figsize=(12, 4))
# librosa.display.specshow(
#     chroma,
#     y_axis='chroma',
#     x_axis='time',
#     cmap='coolwarm'
# )
# plt.colorbar()
# plt.title('Chroma特征与和弦识别结果')
# # 在对应时间点标注和弦
# for frame, chord in detected_chords:
#     plt.text(frame/100, 0.5, chord, color='white', fontsize=8)
# plt.show()

# 计算准确率（需有标注数据）
# ground_truth = [("C", 0, 10), ("G", 10, 20)]  # 示例标注：和弦+起止时间帧
# correct = 0
# total = 0
# for chord_start, chord_name in detected_chords:
#     # 检查是否与标注重叠
#     for gt in ground_truth:
#         if chord_start >= gt[1] and chord_start <= gt[2]:
#             if chord_name == gt[0]:
#                 correct += 1
#             total += 1
# accuracy = correct / total if total > 0 else 0
# print(f"准确率：{accuracy:.2%}")