"""固定训练身份 cohort 的自适应对称分数归一化配置。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SubmissionCell:
    track: str
    language: str
    trials_file: str
    score_file: str
    expected_pairs: int


@dataclass(frozen=True)
class Config:
    experiment_id: str = "v007"
    project_root: Path = Path(__file__).resolve().parents[2]
    output_directory: str = "outputs"
    training_python: str = "/home/xxc/miniconda3/envs/my_project_env/bin/python"
    feature_source_run: str = "outputs/v001/11-04-052404-458965"
    token_source_run: str = "outputs/v004/11-05-080559-29f4dd"
    comparison_run: str = "outputs/v006/11-05-105039-7dddce"
    mode: str = "full"
    compute_budget_seconds: int = 120
    device: str = "cuda:0"
    seed: int = 42
    cpu_threads: int = 4
    gpu_memory_fraction: float = 0.35
    allow_tf32: bool = False
    face_grid_size: int = 7
    face_dimension: int = 512
    voice_dimension: int = 1536
    face_global_dimension: int = 4096
    voice_global_dimension: int = 192
    voice_tokens: int = 32
    embedding_dimension: int = 128
    attention_dimension: int = 128
    position_base: float = 10000.0
    global_mix_initial: float = 0.2
    dropout: float = 0.5
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-5
    batchnorm_momentum: float = 0.1
    batchnorm_epsilon: float = 1e-5
    normalization_epsilon: float = 1e-12
    linear_init_gain: float = 1.0
    linear_init_bias: float = 0.01
    validation_fraction: float = 0.2
    validation_pairs: int = 5000
    evaluation_batch_size: int = 256
    cohort_top_k: int = 20
    cohort_std_correction: int = 1
    cohort_std_floor: float = 1e-4
    gender_codes: tuple[str, ...] = ("f", "m")
    wandb_mode: Literal["offline", "disabled"] = "offline"
    wandb_project: str = "face-voice"
    plot_font_file: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    plot_dpi: int = 150
    smoke_speakers: int = 4
    smoke_samples_per_speaker: int = 1
    submission_data_directory: str = "data/FLAG_2027/dev_set"
    submission_cohort_run: str = "outputs/v007/11-05-110706-48da29"
    submission_budget_seconds: int = 600
    submission_batch_size: int = 16
    submission_log_batches: int = 16
    submission_smoke_pairs: int = 2
    submission_archive: str = "submission.zip"
    submission_score_direction: Literal["distance"] = "distance"
    submission_platform_snapshot: str = "refs/FLAG_2027/11-05-platform.json"
    submission_cells: tuple[SubmissionCell, ...] = (
        SubmissionCell(
            "no_gender",
            "English",
            "English_test.txt",
            "sub_score_v4_English_heard.txt",
            1008,
        ),
        SubmissionCell(
            "no_gender",
            "Bangla",
            "Bangla_test.txt",
            "sub_score_v4_Bangla_unheard.txt",
            1406,
        ),
        SubmissionCell(
            "gender",
            "English",
            "English_test.txt",
            "sub_score_v4_English_heard.txt",
            982,
        ),
        SubmissionCell(
            "gender",
            "Bangla",
            "Bangla_test.txt",
            "sub_score_v4_Bangla_unheard.txt",
            1468,
        ),
    )
    face_weights: str = "data/pretrained/vggface/vgg_face_weights.h5"
    face_weights_sha256: str = (
        "759266b9614d0fd5d65b97bf716818b746cc77ab5944c7bffc937c6ba9455d8c"
    )
    voice_weights: str = "data/pretrained/ecapa/embedding_model.ckpt"
    voice_weights_sha256: str = (
        "a9c25a7b62e68b1ad75831cda9b4c5e5e21672580c9a11cb369db7e1f2250bbf"
    )
    face_size: int = 224
    face_bgr_mean: tuple[float, ...] = (93.5940, 104.7624, 129.1863)
    sample_rate: int = 16000
    audio_seconds: float = 6.0
    audio_mels: int = 80
    ecapa_channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    ecapa_kernels: tuple[int, ...] = (5, 3, 3, 3, 1)
    ecapa_dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    ecapa_attention_channels: int = 128


CONFIG = Config()


def configure(mode: str) -> Config:
    if mode == "smoke":
        return replace(CONFIG, mode=mode, validation_pairs=8, wandb_mode="disabled")
    if mode != "full":
        raise ValueError(f"未知运行模式：{mode}")
    return replace(CONFIG, mode=mode)
