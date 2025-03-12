import librosa
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

def detect_silence(y, sr, frame_length=2048, hop_length=512, percentile=10, offset_db=6):
    """
    使用动态阈值检测静音段
    :param percentile: 计算噪声基底的能量百分位（如最低10%）
    :param offset_db: 阈值 = 噪声基底 + offset_db
    """
    # 计算分帧能量（dB）
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).squeeze()
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)

    # 动态计算阈值
    noise_floor = np.percentile(rms_db, percentile)  # 取最低10%能量的分位值

    threshold_db = noise_floor + offset_db
    
    # 标记静音帧
    silence_frames = np.where(rms_db < threshold_db)[0]
    return silence_frames, threshold_db  # 返回阈值用于调试

def detect_transition(chroma, threshold=0.5):
    # 计算相邻帧的Chroma差异
    diff = np.linalg.norm(np.diff(chroma, axis=1), axis=0)
    # 标记过渡帧（差异超过阈值）
    transition_frames = np.where(diff > threshold)[0]
    return transition_frames

def mark_merge_regions(silence_frames, transition_frames, total_frames):
    # 合并静音和过渡段
    merge_flags = np.zeros(total_frames, dtype=bool)
    merge_flags[silence_frames] = True
    merge_flags[transition_frames] = True
    return merge_flags

def merge_chord_segments(detected_chords, merge_flags, max_gap=0.5):
    merged = []
    current_chord = None
    current_start = 0.0
    current_end = 0.0

    for (start, end, chord) in detected_chords:
        # 检查当前段是否需合并
        is_merge = merge_flags[int(start)] if start < len(merge_flags) else False

        if current_chord is None:
            current_chord = chord
            current_start = start
            current_end = end
        elif chord == current_chord and not is_merge:
            # 相同和弦且无需合并，扩展当前段
            current_end = end
        else:
            if is_merge:
                # 合并到前一段
                current_end = end
            else:
                # 保存当前段并开始新段
                merged.append((current_start, current_end, current_chord))
                current_chord = chord
                current_start = start
                current_end = end
    # 添加最后一段
    if current_chord is not None:
        merged.append((current_start, current_end, current_chord))
    return merged

# 根据音频动态范围自适应静音阈值
def adaptive_silence_threshold(rms_db):
    noise_floor = np.percentile(rms_db, 10)  # 取最低10%能量作为噪声基底
    return noise_floor + 6  # 阈值高于噪声基底6dB

def context_aware_merge(segments):
    # 若静音段前后和弦不同，选择持续时间更长的和弦
    for i in range(1, len(segments)-1):
        prev = segments[i-1]
        curr = segments[i]
        next = segments[i+1]
        if curr[2] == "SILENCE":
            if prev[2] == next[2]:
                # 合并到相同和弦
                prev = (prev[0], next[1], prev[2])
                del segments[i:i+2]
                segments[i-1] = prev
    return segments

def convert_and_merge(
    detected_chords, 
    hop_length, 
    sr, 
    aggregate_window=10, 
    merge_flags=None, 
    min_duration=0.1
):
    """
    整合功能：时间转换 + 合并静音/过渡段
    :param detected_chords: 原始检测结果（帧索引 → 和弦）
    :param merge_flags: 需合并的帧标记（True表示该帧需合并到相邻段）
    :param min_duration: 最小保留时长（秒）
    """
    # 计算时间参数
    frame_duration = hop_length / sr
    window_duration = frame_duration * aggregate_window
    
    # 转换帧索引为秒，并标记需合并的段
    time_chords = []
    for frame_idx, chord in detected_chords:
        start = frame_idx * frame_duration
        end = start + window_duration
        # 标记是否需合并（若未提供merge_flags，默认不合并）
        is_merge = merge_flags[frame_idx] if merge_flags is not None and frame_idx < len(merge_flags) else False
        time_chords.append((start, end, chord, is_merge))
    
    # 单次遍历合并（结合连续相同和弦和静音标记）
    merged = []
    if not time_chords:
        return merged
    
    current_start, current_end, current_chord, _ = time_chords[0]
    for start, end, chord, is_merge in time_chords[1:]:
        # 合并条件：和弦相同 OR 标记需合并
        if dynamic_merge_condition(current_chord,chord,is_merge):
            current_end = max(current_end, end)
        else:
            if (current_end - current_start) >= min_duration:
                merged.append((current_start, current_end, current_chord))
            current_start, current_end, current_chord = start, end, chord
    
    # 添加最后一个段
    if (current_end - current_start) >= min_duration:
        merged.append((current_start, current_end, current_chord))
    
    return merged

def dynamic_merge_condition(current_chord, new_chord, is_merge):
    # 自定义合并逻辑（如静音段强制合并，即使和弦不同）
    if is_merge:
        return True  # 强制合并
    elif current_chord == new_chord:
        return True  # 相同和弦合并
    else:
        return False