import numpy as np

from app.core.chord_templates import chordTemplates

from app.core.recognition import match_chord


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