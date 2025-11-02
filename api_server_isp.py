import os
import io
import traceback
from fastapi import FastAPI, Request, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import time
import soundfile as sf
import sys
import numpy as np

now_dir = os.getcwd()
from indextts.stream_infer_vllm_v2 import IndexTTS2

"""
import torchaudio
### monkey patch
_original_torchaudio_save = torchaudio.save
def patched_save(uri, src, sample_rate, format=None, **kwargs):
    if format is None:
        format = 'wav'
    return _original_torchaudio_save(uri, src, sample_rate, format=format, **kwargs)
torchaudio.save = patched_save
###
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
parser.add_argument("--no_qwen_emo", action="store_true", default=False, help="Disable Qwen_emotion, which can save about 2GB VRAM, but text emotion prompt will be no longer available.")
cmd_args = parser.parse_args()
args=cmd_args

tts = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    tts = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        is_fp16=args.fp16,
        # use_deepspeed=args.use_deepspeed,
        use_cuda_kernel=args.cuda_kernel,
        use_qwen_emo=not args.no_qwen_emo,
    )
    yield


app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "TTS model not initialized"
            }
        )
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Service is running",
            "timestamp": time.time()
        }
    )


@app.get("/voice/indextts/presets", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_preset_get_endpoint(
    text: str = None,
    id: str = None,
    emo_id: str = None,
    emo_control_method: int = 0,
    emo_text: str = None,
    emo_weight: float = 1.0,
    emo_vec: list = None,
    emo_random: bool = False,
    max_text_tokens_per_sentence: int = 120,
    stream: bool = False,
):
    if emo_vec is None:
        emo_vec = [0] * 8
    data = {
        "text": text,
        "id": id,
        "emo_id": emo_id,
        "emo_control_method": emo_control_method,
        "emo_text": emo_text,
        "emo_weight": float(emo_weight),
        "emo_vec": emo_vec,
        "emo_random": emo_random,
        "max_text_tokens_per_sentence": max_text_tokens_per_sentence,
        "stream": stream,
    }
    return await tts_preset_handle(data)


@app.post("/voice/indextts/presets", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_preset_post_endpoint(request: Request):
    try:
        data = await request.json()
        return await tts_preset_handle(data)
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request):
    try:
        data = await request.json()
        return await tts_api_url_internal(data)
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


async def tts_preset_handle(data: dict):
    try:
        api_root_folder = now_dir
        id = data.get("id")
        emo_id = data.get("emo_id")
        
        spk_audio_path = None
        if id:
            for ext in ["wav", "mp3"]:
                path = os.path.join(api_root_folder, "presets", "voice", id, f"prompt.{ext}")
                if os.path.exists(path):
                    spk_audio_path = path
                    break
        
        emo_ref_path = None
        if emo_id:
            for ext in ["wav", "mp3"]:
                path = os.path.join(api_root_folder, "presets", "emo", f"{emo_id}.{ext}")
                if os.path.exists(path):
                    emo_ref_path = path
                    break
        
        data["spk_audio_path"] = spk_audio_path
        data["emo_ref_path"] = emo_ref_path
        return await tts_api_url_internal(data)
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


async def tts_api_url_internal(data: dict):
    try:
        emo_control_method = data.get("emo_control_method", 0)
        text = data["text"]
        spk_audio_path = data["spk_audio_path"]
        emo_ref_path = data.get("emo_ref_path", None)
        emo_weight = data.get("emo_weight", 1.0)
        emo_vec = data.get("emo_vec", [0] * 8)
        emo_text = data.get("emo_text", None)
        emo_random = data.get("emo_random", False)
        max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 120)

        global tts
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        if emo_control_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        if emo_control_method == 1:
            emo_weight = emo_weight
        if emo_control_method == 2:
            vec = emo_vec
            vec_sum = sum(vec)
            if vec_sum > 1.5:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "error": "情感向量之和不能超过1.5，请调整后重试。"
                    }
                )
        else:
            vec = None

        stream = data.get("stream", False)
        if stream:
            
            async def audio_streamer():
                import wave as _wave
                
                with io.BytesIO() as _buf:
                    _wf = _wave.open(_buf, 'wb')
                    _wf.setnchannels(1)
                    _wf.setsampwidth(2)
                    _wf.setframerate(22050)
                    _wf.writeframes(b'')
                    header_bytes = _buf.getvalue()

                first_chunk_sent = False
                async for wav_chunk in tts.infer_stream(spk_audio_prompt=spk_audio_path, text=text,
                                                        output_path=None,
                                                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                                                        emo_vector=vec,
                                                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                                                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence)):
                    try:
                        arr = wav_chunk.cpu().numpy()
                    except Exception:
                        raise RuntimeError("Failed to convert tensor to numpy array for streaming.")

                    if arr.dtype.kind == 'f':
                        max_abs = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
                        if max_abs > 1.5:
                            arr_int16 = arr.astype(np.int16)
                        else:
                            arr_int16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                    elif arr.dtype == np.int16:
                        arr_int16 = arr
                    else:
                        arr_int16 = arr.astype(np.int16)

                    chunk_bytes = arr_int16.tobytes()
                    if not first_chunk_sent:
                        yield header_bytes + chunk_bytes
                        first_chunk_sent = True
                    else:
                        yield chunk_bytes
            
            return StreamingResponse(audio_streamer(), media_type="audio/wav")

        sr, wav = await tts.infer(spk_audio_prompt=spk_audio_path, text=text,
                        output_path=None,
                        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                        emo_vector=vec,
                        use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                        max_text_tokens_per_sentence=int(max_text_tokens_per_sentence))
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


if __name__ == "__main__":
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    uvicorn.run(app=app, host=args.host, port=args.port, log_level="debug")