import librosa
import numpy as np

from app.preprocessing import preprocess_and_convert_audio

input_path="./special_person.mp3"
output_path="test.wav"

preprocess_and_convert_audio(input_path,output_path)
y, sr = librosa.load("test.wav")
# 使用 Viterbi 解码
# 假设有一些隐状态和转移概率，你需要根据实际情况调整这些值
# 例如，设定一些状态数和概率
num_states = 12  # 假设有 12 个和弦
transition_probs = np.ones((num_states, num_states)) / num_states  # 均匀分布的转移概率
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chords = librosa.sequence.viterbi(chroma.T,transition_probs)

print(chords)