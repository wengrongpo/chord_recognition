import json
import librosa
import numpy as np

from app.core.chord_templates import chordTemplates

from app.core.recognition import convert_and_merge, detect_chords, detect_silence, detect_transition, mark_merge_regions, match_chord
from app.preprocessing import preprocess_and_convert_audio
import functools
import pytest
def validate_test_case(func):
    @functools.wraps(func)
    def wrapper(test_case):
        if not (test_case / "test.wav").exists():
            pytest.skip(f"缺少 test.wav: {test_case}")
        return func(test_case)
    return wrapper

def load_expected_result(test_case_dir):
    """加载预期结果"""
    expected_path = test_case_dir / "expected.json"
    with open(expected_path, 'r') as f:
        return json.load(f)
    
@validate_test_case
def test_chord_recognition(test_case):
    """动态生成的测试用例"""
    # 输入输出路径
    input_wav = test_case / "test.wav"
    expected = load_expected_result(test_case)

    preprocess_and_convert_audio(input_wav,input_wav)
    # 1. 加载音频并提取特征
    y, sr = librosa.load(input_wav)

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
    
    # 转换为与expected相同的格式
    result_formatted = [[c] for (_, _, c) in merged]
    
    # 断言结果匹配（允许0.1秒误差）
    assert len(result_formatted) == len(expected)
    for (actual, expected) in zip(result_formatted, expected):
        # assert abs(actual[0] - expected[0]) < 0.1
        # assert abs(actual[1] - expected[1]) < 0.1
        assert actual[0] == expected

def test_c_major_template():
    # 理想C大三和弦的Chroma向量（C=1.5, E=1.0, G=1.0）
    chroma = np.array([1.5, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0])
    # 匹配模板
    chord, score = match_chord(chroma, chordTemplates)
    assert chord == "C"
    assert score > 0.99  # 预期接近完美匹配

def test_am_minor_template():
    # 理想Am小三和弦的Chroma向量（A=1.5, C=1.0, E=1.0）
    chroma = np.array([1.0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 1.5, 0, 0])
    chord, score = match_chord(chroma, chordTemplates)
    assert chord == "Am"

# 验证输入音频参数
def test_audio_metadata(test_case):
    input_wav = test_case / "test.wav"
    sr = librosa.get_samplerate(input_wav)
    assert sr == 22050, "采样率不符合要求"

@pytest.mark.skip(reason="未实现")
def test_performance(benchmark, test_case):
    input_wav = test_case / "test.wav"
    benchmark(convert_and_merge, ...)
