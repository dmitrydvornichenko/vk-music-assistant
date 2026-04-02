"""Silero VAD через ONNX Runtime — без зависимости на torch/torchaudio.

API совместим с silero_vad:
    load_silero_vad()  -> model
    get_speech_timestamps(audio, model, sampling_rate=16000) -> list[dict]
"""
import os
import logging
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

ONNX_MODEL_PATH = os.environ.get("SILERO_ONNX_PATH", "/silero_vad.onnx")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))

_CHUNK = 512  # сэмплов на один шаг модели (требование Silero VAD @ 16 kHz)


class _SileroOnnx:
    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # ERROR only
        self._sess = ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._reset()

    def _reset(self):
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def _infer(self, chunk_f32: np.ndarray, sr: int) -> float:
        out, h, c = self._sess.run(
            None,
            {
                "input": chunk_f32.reshape(1, -1),
                "sr": np.array(sr, dtype=np.int64),
                "h": self._h,
                "c": self._c,
            },
        )
        self._h, self._c = h, c
        return float(out)

    def get_speech_timestamps(
        self, audio: np.ndarray, sampling_rate: int = 16000
    ) -> list:
        if audio.dtype == np.int16:
            af = audio.astype(np.float32) / 32768.0
        else:
            af = audio.astype(np.float32)

        self._reset()
        speeches = []
        in_speech = False
        speech_start = 0

        for i in range(0, len(af) - _CHUNK + 1, _CHUNK):
            prob = self._infer(af[i : i + _CHUNK], sampling_rate)
            if prob >= VAD_THRESHOLD and not in_speech:
                in_speech = True
                speech_start = i
            elif prob < VAD_THRESHOLD and in_speech:
                in_speech = False
                speeches.append({"start": speech_start, "end": i})

        if in_speech:
            speeches.append({"start": speech_start, "end": len(af)})

        return speeches


def load_silero_vad() -> _SileroOnnx:
    log.info("Loading Silero VAD ONNX model from %s", ONNX_MODEL_PATH)
    return _SileroOnnx(ONNX_MODEL_PATH)


def get_speech_timestamps(
    audio: np.ndarray, model: _SileroOnnx, sampling_rate: int = 16000
) -> list:
    return model.get_speech_timestamps(audio, sampling_rate)
