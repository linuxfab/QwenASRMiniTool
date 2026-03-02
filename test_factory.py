"""
test_factory.py — engine/factory.py 單元測試

覆蓋 create_engine() 的後端選擇邏輯。
僅測試回傳類型正確（不載入模型）。
"""
from unittest.mock import patch

import pytest


class TestCreateEngine:
    """engine/factory.py create_engine 的後端路由邏輯。"""

    def test_openvino_06b_default(self):
        from engine.factory import create_engine
        eng = create_engine(backend="openvino", model_size="0.6B")
        assert type(eng).__name__ == "OpenVINO06BEngine"

    def test_openvino_17b(self):
        from engine.factory import create_engine
        eng = create_engine(backend="openvino", model_size="1.7B")
        assert type(eng).__name__ == "OpenVINO17BEngine"

    def test_openvino_17b_in_string(self):
        """model_size 含有 '1.7B' 即觸發 1.7B 引擎。"""
        from engine.factory import create_engine
        eng = create_engine(backend="openvino", model_size="Qwen3-ASR-1.7B")
        assert type(eng).__name__ == "OpenVINO17BEngine"

    def test_chatllm_backend(self):
        from engine.factory import create_engine
        eng = create_engine(backend="chatllm")
        assert type(eng).__name__ == "VulkanEngine"

    def test_cuda_backend(self):
        from engine.factory import create_engine
        eng = create_engine(backend="cuda")
        assert type(eng).__name__ == "CUDAEngine"

    def test_default_is_openvino_06b(self):
        """不帶參數時預設回傳 OpenVINO 0.6B。"""
        from engine.factory import create_engine
        eng = create_engine()
        assert type(eng).__name__ == "OpenVINO06BEngine"
