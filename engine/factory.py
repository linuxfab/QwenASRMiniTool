"""
engine/factory.py — 引擎工廠函式

依設定建立對應的 ASR 推理引擎實例。
"""
from __future__ import annotations

from .base import ASREngineBase


def create_engine(backend: str = "openvino", model_size: str = "0.6B") -> ASREngineBase:
    """依後端與模型大小建立引擎。

    Parameters
    ----------
    backend : str
        "openvino" | "chatllm" | "cuda"
    model_size : str
        "0.6B" | "1.7B"（僅 openvino 有分）

    Returns
    -------
    ASREngineBase
        尚未載入模型的引擎實例，需呼叫 .load() 載入。
    """
    if backend == "chatllm":
        from .vulkan_backend import VulkanEngine
        return VulkanEngine()
    elif backend == "cuda":
        from .cuda_backend import CUDAEngine
        return CUDAEngine()
    elif "1.7B" in model_size or model_size == "1.7B":
        from .openvino_backend import OpenVINO17BEngine
        return OpenVINO17BEngine()
    else:
        from .openvino_backend import OpenVINO06BEngine
        return OpenVINO06BEngine()
