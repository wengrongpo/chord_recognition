import numpy as np

def merge_chord_segments(time_chords):
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
        if current_chord == last[2] :
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
    merged = merge_chord_segments(time_chords)
    
    # 合并后处理：移除过短的段（可选）
    # merged = [ (s, e, c) for s, e, c in merged if e - s >= 3 ]  # 短于0.05秒的段视为噪声
    
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