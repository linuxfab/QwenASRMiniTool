"""
engine/ — 統一 ASR 推理引擎模組

將 OpenVINO / Vulkan / CUDA 三種後端統一為相同介面，
前端（Tkinter / Streamlit）只需透過 create_engine() 建立引擎，
呼叫 load() / transcribe() / process_file() 即可。
"""
from .base import ASREngineBase
from .factory import create_engine

__all__ = ["ASREngineBase", "create_engine"]
