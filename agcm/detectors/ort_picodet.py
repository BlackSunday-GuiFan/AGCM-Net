"""Custom ONNX-runtime wrapper for PaddleDetection PicoDet inference.

Mirrors PicoDetDetector.detect_jobs_batch interface so it can drop into the
AGCM pipeline as a direct replacement. ONNX file already fuses multiclass_nms3
inside, so host only does preprocess + bbox decode.

Note: PaddleDetection ONNX export supports batch=1 only (per EXPORT_ONNX_MODEL.md
"PicoDet  11  仅支持batch=1推理"). We loop over single images for batch=1.
"""
from __future__ import annotations
import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ORTPicoDetDetector:
    name = 'picodet-ort'

    def __init__(self, onnx_path, imsz=640, device='cuda:0', use_tensorrt=True):
        import onnxruntime as ort  # lazy: env without onnxruntime can still import this module
        providers = self._make_providers(device, use_tensorrt, ort)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.imsz = imsz
        self.providers = providers
        print('ORT PicoDet providers:', providers, flush=True)

    @staticmethod
    def _make_providers(device='cuda:0', use_tensorrt=True, ort=None):
        if ort is None:
            import onnxruntime as _ort
            ort = _ort
        available = ort.get_available_providers()
        if use_tensorrt and 'TensorrtExecutionProvider' in available:
            return [('TensorrtExecutionProvider', {
                'trt_fp16_enable': True, 'trt_max_workspace_size': 2 << 30,
            }), 'CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _pre_one(self, img):
        arr = img.astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return arr[None, :, :, ::-1].transpose(0, 3, 1, 2).copy()

    def _forward_one(self, img):
        x = self._pre_one(img)
        sf = np.ones((1, 2), dtype=np.float32)
        out = self.sess.run(None, {'image': x, 'scale_factor': sf})[0]
        bs = []
        for det in out:
            cls_id, score, x1, y1, x2, y2 = det
            if cls_id < 0 or score < 0.001:
                continue
            bs.append({'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        'class': int(cls_id), 'confidence': float(score)})
        return bs

    def detect_jobs_batch(self, jobs):
        boxes = []
        for crop, ox, oy in jobs:
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            ri = cv2.resize(crop, (self.imsz, self.imsz), interpolation=cv2.INTER_AREA)
            sx, sy = ri.shape[1] / 640.0, ri.shape[0] / 640.0
            for b in self._forward_one(ri):
                x1 = b['bbox'][0] * sx + ox
                y1 = b['bbox'][1] * sy + oy
                x2 = (b['bbox'][0] + b['bbox'][2]) * sx + ox
                y2 = (b['bbox'][1] + b['bbox'][3]) * sy + oy
                boxes.append({'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                              'class': b['class'], 'confidence': b['confidence']})
        return boxes
