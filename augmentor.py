from dataclasses import dataclass
from pydub import AudioSegment
from typing import List
from array import array

import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import time
import os


# ============================================================
# 증강 결과 DataClass
# ============================================================
@dataclass
class AugmentResult:
    augment_type: str      # pitch, phase
    elapsed_sec: float     # 처리 시간 (초 단위, 소수점 3자리)
    output_path: str       # 증강 결과물 저장 경로
    engine: str            # 사용된 엔진: librosa / pydub


# ============================================================
# 공통 시간 측정 데코레이터
# ============================================================
def measure_time(func):
    """함수 실행 시간을 측정하여 결과와 경과 시간을 반환"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_sec = round(time.time() - start, 6)   # 0.000 형식
        return result, elapsed_sec
    return wrapper


# ============================================================
# Hybrid Audio Augmentor
# librosa + pydub 하이브리드 방식
# 각 항목별로 가장 빠르거나, 품질 좋은 엔진 자동 선택
# ============================================================
class HybridAudioAugmentor:
    def __init__(self, sr=16000, speed_engine="pydub"):
        """
        sr: librosa에서 로드할 때 사용할 샘플레이트
        speed_engine:
            'pydub'   → 속도 빠름. pitch 변함. (기본값)
            'librosa' → pitch 유지하며 속도 조절. 정확하지만 느림
        """
        self.sr = sr
        self.speed_engine = speed_engine

    # --------------------------------------------------------
    # Loading 함수 (librosa + pydub 둘 다 로드)
    # librosa → waveform 기반 DSP용
    # pydub   → FFmpeg 기반 빠른 편집 작업용
    # --------------------------------------------------------
    @measure_time
    def load(self, path):
        y, _ = librosa.load(path, sr=self.sr)          # float32 waveform
        audio_segment = AudioSegment.from_file(path)   # pydub AudioSegment
        return (y, audio_segment)

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------
    def save_waveform(self, path, y):
        """librosa 기반 DSP 결과 저장"""
        sf.write(path, y, self.sr)

    def save_segment(self, path, seg):
        """pydub 기반 결과 저장"""
        seg.export(path, format="wav")

    # --------------------------------------------------------
    # librosa 기반 DSP 효과 (정밀 처리)
    # --------------------------------------------------------
    def add_noise(self, y, noise_factor=0.005):
        """랜덤 노이즈 추가 → STT 학습용 기본 증강"""
        noise = np.random.randn(len(y))
        return y + noise_factor * noise

    def pitch_shift(self, y, n_steps=2):
        """음높이를 반음 단위로 조절"""
        return librosa.effects.pitch_shift(y=y, sr=self.sr, n_steps=n_steps)

    def speed_librosa(self, y, rate=1.1):
        """속도 변화 (pitch 유지). 고품질 DSP"""
        return librosa.effects.time_stretch(y=y, rate=rate)

    # --------------------------------------------------------
    # pydub 기반 효과 (빠른 처리)
    # --------------------------------------------------------
    def gain(self, seg, db=5):
        """볼륨 증감 (빠르고 안정적)"""
        return seg + db

    def speed_pydub(self, seg, rate=1.1):
        """
        속도 변화 (빠르지만 pitch 변함)
        → frame_rate 조절 방식
        """
        new_fr = int(seg.frame_rate * rate)
        return seg._spawn(seg.raw_data, overrides={"frame_rate": new_fr}).set_frame_rate(seg.frame_rate)

    def lowpass(self, seg, cutoff=3000):
        """저역 통과 필터 (고음 제거)"""
        return seg.low_pass_filter(cutoff)

    def highpass(self, seg, cutoff=300):
        """고역 통과 필터 (저음 제거)"""
        return seg.high_pass_filter(cutoff)

    def phase_invert(self, seg):
        """
        위상 반전 (파형 polarity 반전)
        array.array로 변환 후 pydub spawn
        """
        samples = seg.get_array_of_samples()
    
        # numpy 배열로 변환 (int16)
        arr = np.array(samples, dtype=np.int32)
    
        # 반전
        arr = -arr
    
        # int16 범위로 clip
        arr = np.clip(arr, -32768, 32767)
    
        # 다시 array.array 로 변환해 AudioSegment 로 spawn
        inverted = array(samples.typecode, arr.astype(np.int16).tolist())
        return seg._spawn(inverted)

    # --------------------------------------------------------
    # Main Runner : 모든 증강 항목 순회하며 처리
    # --------------------------------------------------------
    def run_augmentation(self, input_path, output_dir) -> List[AugmentResult]:
        # 1) 파일 로딩
        (y, seg), load_time = self.load(input_path)
        filename = os.path.splitext(os.path.basename(input_path))[0]

        results = []

        # ---------------------------------------------------
        # librosa 증강 항목 (DSP 기반)
        # ---------------------------------------------------
        librosa_items = [
            # ("noise", self.add_noise, y, "librosa"),
            ("pitch", self.pitch_shift, y, "librosa"),
        ]

        # speed 엔진이 librosa면 librosa 방식 speed 추가
        # if self.speed_engine == "librosa":
        #     librosa_items.append(("speed", self.speed_librosa, y, "librosa"))

        # ---------------------------------------------------
        # pydub 증강 항목 (빠른 필터 & 볼륨)
        # ---------------------------------------------------
        pydub_items = [
            ("gain", self.gain, seg, "pydub"),
            # ("lowpass", self.lowpass, seg, "pydub"),
            # ("highpass", self.highpass, seg, "pydub"),
            ("phase", self.phase_invert, seg, "pydub"),
        ]

        # speed 엔진이 pydub이면 pydub 방식 speed 추가
        # if self.speed_engine == "pydub":
        #     pydub_items.append(("speed", self.speed_pydub, seg, "pydub"))

        # ---------------------------------------------------
        # 2) 각 항목 실행 및 시간 측정
        # ---------------------------------------------------
        for augment_type, func, data, engine in (librosa_items + pydub_items):

            augmented, t = measure_time(func)(data)

            # 저장 경로 형식: original_speed_librosa.wav 등
            save_path = os.path.join(
                output_dir,
                f"{filename}_{augment_type}.wav"
            )

            # 엔진 구분하여 다른 방식으로 저장
            if engine == "librosa":
                self.save_waveform(save_path, augmented)
            else:
                self.save_segment(save_path, augmented)

            results.append(
                AugmentResult(
                    augment_type=augment_type,
                    elapsed_sec=t,
                    output_path=save_path,
                    engine=engine
                )
            )

        return results


def run_batch_augmentation(input_dir, output_dir, augmentor: HybridAudioAugmentor):
    """
    input_dir  : 원본 오디오들이 있는 디렉토리
    output_dir : 증강 결과들이 저장될 최상위 폴더
    augmentor  : HybridAudioAugmentor 객체
    """

    os.makedirs(output_dir, exist_ok=True)

    records = []

    for file in os.listdir(input_dir):
        if not file.lower().endswith(".mp3") and not file.lower().endswith(".wav"):
            continue

        input_path = os.path.join(input_dir, file)

        # ---- mp3 duration ----
        mp3_seg = AudioSegment.from_file(input_path)
        duration_sec = float(f"{mp3_seg.duration_seconds:.6f}")

        # ---- 증강 처리 ----
        results: List[AugmentResult] = augmentor.run_augmentation(
            input_path=input_path,
            output_dir=output_dir
        )

        for r in results:
            folder = os.path.join(output_dir, r.augment_type)
            os.makedirs(folder, exist_ok=True)

            moved_path = os.path.join(folder, os.path.basename(r.output_path))
            os.replace(r.output_path, moved_path)

            records.append({
                "filename": file,
                "input_path": input_path,
                "duration_sec": duration_sec,           
                "augment_type": r.augment_type,
                "elapsed_sec": float(f"{r.elapsed_sec:.6f}"),
                "output_path": moved_path
            })

    df = pd.DataFrame(records)

    excel_path = os.path.join(output_dir, "augmentation_results.xlsx")
    df.to_excel(excel_path, index=False)

    print()
    print(f" 증강 완료: {len(df)}개 결과 생성")
    print(f" 엑셀 저장: {excel_path}")

    return df


if __name__ == '__main__':
    augmentor = HybridAudioAugmentor()

    input_dir = "./data/augment/input"
    output_dir = "./data/augment/output"

    df = run_batch_augmentation(input_dir, output_dir, augmentor)
