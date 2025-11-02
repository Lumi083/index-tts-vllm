from dataclasses import asdict, dataclass
import os
from typing import List, Optional
import requests

SERVER_PORT = 23467
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

url = F"http://127.0.0.1:{SERVER_PORT}/tts_url"

# 0. GET /voice/indextts/presets?id=0&emo_control_method=1&emo_id=&vec1=0.0&vec2=0.0&vec3=0.0&vec4=0.0&vec5=0.0&vec6=0.0&vec7=0.0&vec8=0.0&emo_weight=0.6&stream=False&max_text_tokens_per_segment=120&quick_token=0&lang=zh&audio_format=wav&_verify=0&text=你好，请问你是谁？ HTTP/1.
# @dataclass
# class IndexTTS2PresetRequestData:
#     id: str
#     emo_control_method: int = 0
#     emo_id: str = ""
#     vec1: float = 0.0
#     vec2: float = 0.0
#     vec3: float = 0.0
#     vec4: float = 0.0
#     vec5: float = 0.0
#     vec6: float = 0.0
#     vec7: float = 0.0
#     vec8: float = 0.0
#     emo_weight: float = 1.0
#     stream: bool = False
#     max_text_tokens_per_segment: int = 120
#     quick_token: int = 0
#     lang: str = "zh"
#     audio_format: str = "wav"
#     _verify: int = 0
#     text: str = ""

#     def to_dict(self) -> str:
#         return asdict(self)
    
# data = IndexTTS2PresetRequestData(
#     id="0",
#     text="你好，请问你是谁？"
# )
# response = requests.get(
#     f"http://127.0.0.1:{SERVER_PORT}/voice/indextts/presets", params=data.to_dict()
# )
# print(f"Preset Response: {response.status_code}, {response.reason}")


@dataclass
class IndexTTS2RequestData:
    text: str
    spk_audio_path: str
    emo_control_method: int = 0
    emo_ref_path: Optional[str] = None
    emo_weight: float = 1.0
    emo_vec: List[float] = None
    emo_text: Optional[str] = None
    emo_random: bool = False
    max_text_tokens_per_sentence: int = 120
    stream: bool = False

    def __post_init__(self):
        # 保证 emo_vec 默认长度为 8 的 0 向量
        if self.emo_vec is None:
            self.emo_vec = [0.0] * 8

    def to_dict(self) -> str:
        return asdict(self)

print("Starting API requests to IndexTTS2 server...")
# 1. 情感与音色参考音频相同
data = IndexTTS2RequestData(
    text="""This approach uses a WAV header""",
    spk_audio_path="assets/jay_promptvn.wav"
)
import wave
import io
import soundfile as sf


if data.stream:
    import time
    start_time = time.time()
    binary_chunks = []
    # API: return StreamingResponse(audio_streamer(), media_type="audio/wav")
    with requests.post(url, json=data.to_dict(), stream=True) as response:
        first_chunk_received = False
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                if not first_chunk_received:
                    first_chunk_time = time.time()
                    print(f"First chunk received in {first_chunk_time - start_time:.2f} seconds.")
                    first_chunk_received = True
                binary_chunks.append(chunk)
    wav_bytes = b"".join(binary_chunks)
    print(f"Data Content: {wav_bytes[:60]}... (total {len(wav_bytes)} bytes)")
    import numpy as np
    output_path = os.path.join(output_dir, "output1_streamed.wav")
    if wav_bytes.startswith(b"RIFF") or wav_bytes.startswith(b"RIFX"):
        # Full WAV file with header
        print("Received WAV file from stream.")
        with open(output_path, "wb") as f:
            f.write(wav_bytes)

else:
    response = requests.post(url, json=data.to_dict())
    print(f"Response: {response.status_code}, {response.reason}")
    # print(f"Response: \n{response.content.decode('utf-8')}")
    with open(os.path.join(output_dir, "output1.wav"), "wb") as f:
        f.write(response.content)
exit(0)

# 2. 使用情感参考音频
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=1,
    emo_ref_path="assets/vo_card_klee_endOfGame_fail_01.wav",
    emo_weight=0.6
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output2.wav"), "wb") as f:
    f.write(response.content)

# 3. 使用情感向量控制
# ["喜", "怒", "哀", "惧", "厌恶", "低落", "惊喜", "平静"]
emo_vec = [0, 0, 0.55, 0, 0, 0, 0, 0]

data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=2,
    emo_vec=emo_vec
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output3.wav"), "wb") as f:
    f.write(response.content)

# 4. 使用情感描述文本控制
data = IndexTTS2RequestData(
    text="还是会想你，还是想登你",
    spk_audio_path="assets/jay_promptvn.wav",
    emo_control_method=3,
    emo_text="极度悲伤"
)

response = requests.post(url, json=data.to_dict())
with open(os.path.join(output_dir, "output4.wav"), "wb") as f:
    f.write(response.content)
