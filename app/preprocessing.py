import os
import librosa
import numpy as np
from pydub import AudioSegment

def extract_chroma(audio_file, sr=22050, hop_length=512):
    """
    从音频文件中提取 Chroma 特征
    """
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=sr,mono=True)
    # 若需统一长度，裁剪/填充到固定时长（如30秒）
    if len(y) < 30 * 22050:
        y = librosa.util.fix_length(y, size=30 * 22050)
    # 提取 Chroma 特征
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    
    # 将特征调整为固定长度（例如：取平均值）
    if chroma.shape[1] > 100:
        chroma = np.mean(chroma.reshape(12, -1, 100), axis=1)
    else:
        chroma = np.pad(chroma, ((0, 0), (0, 100 - chroma.shape[1])))
    
    return chroma
# 使用pydub将MP3转为WAV（需安装ffmpeg）


def preprocess_and_convert_audio(input_path, output_path):
    """
    预处理音频文件，将其转换为WAV格式并提取Chroma特征，返回模型所需的输入格式
    """
    # 如果output_path已经有文件了就不转化了
    if not os.path.exists(output_path):
        # 转换音频文件为WAV格式
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(22050)  # 转单声道+重采样
        audio = audio.normalize(headroom=0.0)  # 强制满量程（无余量）
        audio.export(output_path, format="wav")
    
    # # 提取Chroma特征
    # chroma = extract_chroma(output_path)
    # # 添加批次维度并转换为 PyTorch 所需的格式
    # chroma = np.expand_dims(chroma, axis=0)
    # return chroma 

# import webrtcvad  # 需安装webrtcvad

# def remove_silence(y, sr):
#     vad = webrtcvad.Vad(2)  # 激进度2（中等）
#     frame_duration = 30  # 每帧30ms
#     frames = split_to_frames(y, sr, frame_duration)
#     filtered_frames = [frame for frame in frames if vad.is_speech(frame, sr)]
#     return np.concatenate(filtered_frames)

def validate_audio(y, sr):
    assert sr == 22050, f"采样率应为22050Hz，当前为{sr}Hz"
    assert y.ndim == 1, "必须为单声道音频"
    assert len(y) >= 5 * sr, "音频过短（至少5秒）"
    max_amp = np.max(np.abs(y))
    assert max_amp > 0.01, "音频振幅过低（可能为静音文件）"