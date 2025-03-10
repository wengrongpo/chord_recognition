import librosa
import numpy as np
import matplotlib.pyplot as plt

from app.preprocessing import preprocess_and_convert_audio

input_path="./special_person.mp3"
output_path="test.wav"

preprocess_and_convert_audio(input_path,output_path)
y, sr = librosa.load("test.wav")

# 和弦模板字典：{和弦名称: 音级掩码（0/1表示是否属于该和弦）}
chord_templates = {
    # --------------------------
    # 大三和弦 (Major Triads)
    # 结构：根音 + 大三度 + 纯五度
    # --------------------------
    "C":  [1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0],  # C-E-G
    "C#": [0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0],  # C#-F-G#
    "D":  [0, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0],  # D-F#-A
    "D#": [0, 0, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0],  # D#-G-A#
    "E":  [0, 0, 0, 0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0],  # E-G#-B
    "F":  [1.0, 0, 0, 0, 0, 1.5, 0, 0, 0, 1.0, 0, 0],  # F-A-C
    "F#": [0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 0, 1.0, 0],  # F#-A#-C#
    "G":  [0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 0, 1.0],  # G-B-D
    "G#": [1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 0],  # G#-C-D#
    "A":  [0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0],  # A-C#-E
    "A#": [0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0],  # A#-D-F
    "B":  [0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5],  # B-D#-F#

    # --------------------------
    # 小三和弦 (Minor Triads)
    # 结构：根音 + 小三度 + 纯五度
    # --------------------------
    "Cm":  [1.5, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0],  # C-D#-G
    "C#m": [0, 1.5, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0],  # C#-E-G#
    "Dm":  [0, 0, 1.5, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0],  # D-F-A
    "D#m": [0, 0, 0, 1.5, 0, 0, 1.0, 0, 0, 0, 1.0, 0],  # D#-F#-A#
    "Em":  [0, 0, 0, 0, 1.5, 0, 0, 1.0, 0, 0, 0, 1.0],  # E-G-B
    "Fm":  [1.0, 0, 0, 0, 0, 1.5, 0, 0, 1.0, 0, 0, 0],  # F-G#-C
    "F#m": [0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 1.0, 0, 0],  # F#-A-C#
    "Gm":  [0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 1.0, 0],  # G-A#-D
    "G#m": [1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0, 0],  # G#-B-D#
    "Am":  [0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0, 0],  # A-C-E
    "A#m": [0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5, 0],  # A#-C#-F
    "Bm":  [0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 1.5],  # B-D-F#

    # --------------------------
    # 属七和弦 (Dominant 7th)
    # 结构：根音 + 大三度 + 纯五度 + 小七度
    # --------------------------
    "C7":  [1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0],  # C-E-G-A#
    "C#7": [0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0],  # C#-F-G#-B
    # ... 其他属七和弦（按相同规则生成）

    # --------------------------
    # 大七和弦 (Major 7th)
    # 结构：根音 + 大三度 + 纯五度 + 大七度
    # --------------------------
    "Cmaj7": [1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0, 0, 0],  # C-E-G-B
    "C#maj7": [0, 1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 1.0, 0],  # C#-F-G#-C
    # ... 其他大七和弦

    # --------------------------
    # 挂二和弦 (Sus2)
    # 结构：根音 + 大二度 + 纯五度
    # --------------------------
    "Csus2": [1.5, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0],  # C-D-G
    "C#sus2": [0, 1.5, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0],  # C#-D#-G#
    # ... 其他挂二和弦

    # --------------------------
    # 挂四和弦 (Sus4)
    # 结构：根音 + 纯四度 + 纯五度
    # --------------------------
    "Csus4": [1.5, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0],  # C-F-G
    "C#sus4": [0, 1.5, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0],  # C#-F#-G#
    # ... 其他挂四和弦
}

def merge_chord_segments(time_chords, max_gap=0.5):
    """
    合并相邻的相同和弦，允许微小间隔（max_gap秒内视为连续）
    :param time_chords: 列表，元素为 (start_time, end_time, chord)
    :param max_gap: 允许合并的最大间隔（秒）
    :return: 合并后的时间段列表
    """
    if not time_chords:
        return []
    
    # 按开始时间排序
    sorted_segments = sorted(time_chords, key=lambda x: x[0])
    merged = [list(sorted_segments[0])]  # 转换为list以便修改
    
    for seg in sorted_segments[1:]:
        last = merged[-1]
        current_start, current_end, current_chord = seg
        
        # 判断是否可合并：和弦相同且时间连续或间隔小于max_gap
        if (current_chord == last[2] and 
            current_start <= last[1] + max_gap):
            # 扩展合并段的结束时间
            last[1] = max(last[1], current_end)
        else:
            merged.append(list(seg))
    
    return merged
def convert_frames_to_seconds(detected_chords, hop_length, sr, aggregate_window=10):
    # 计算每帧的持续时间（秒）
    frame_duration = hop_length / sr
    # 每个检测窗口的持续时间（秒）
    window_duration = frame_duration * aggregate_window
    
    # 转换时间帧为秒（注意：detected_chords中的frame_idx是窗口起始帧）
    time_chords = []
    for frame_idx, chord in detected_chords:
        # 计算实际时间
        start_time = frame_idx * frame_duration
        end_time = start_time + window_duration
        time_chords.append((start_time, end_time, chord))
    
    # 合并连续/重叠的相同和弦
    merged = merge_chord_segments(time_chords, max_gap=2.0)
    
    # 合并后处理：移除过短的段（可选）
    merged = [ (s, e, c) for s, e, c in merged if e - s >= 3 ]  # 短于0.05秒的段视为噪声
    
    return merged
def match_chord(chroma_vector, templates):
    scores = {}
    for chord, template in templates.items():
        # 余弦相似度（归一化后）
        score = np.dot(chroma_vector, template) / (np.linalg.norm(chroma_vector) * np.linalg.norm(template))
        scores[chord] = score
    return max(scores.items(), key=lambda x: x[1])

def detect_chords(chroma, templates, aggregate_window=10):
    num_frames = chroma.shape[1]
    chords = []
    for i in range(0, num_frames, aggregate_window):
        # 聚合窗口内的Chroma特征（取均值）
        window = chroma[:, i:i+aggregate_window]
        aggregated = np.mean(window, axis=1)
        chord, _ = match_chord(aggregated, templates)
        chords.append((i, chord))
    return chords

def detect_key(chroma):
    # 计算每个音级作为根音的总能量
    chroma_sum = np.sum(chroma, axis=1)
    root_index = np.argmax(chroma_sum)
    return root_index  # 0=C, 1=C#, ..., 11=B

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
detected_chords = detect_chords(chroma, chord_templates)
# print("识别结果（时间帧 → 和弦）：", detected_chords)

# 3. 转换为秒并合并
merged = convert_frames_to_seconds(detected_chords, hop_length=1024, sr=22050, aggregate_window=50)

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