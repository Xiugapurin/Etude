# etude/config/schema.py

"""
Pydantic configuration schema for Etude project.

All configuration parameters are defined here with type hints and defaults.
YAML files only need to specify values that differ from defaults.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ============================================================
# Environment Configuration
# ============================================================


class EnvConfig(BaseModel):
    """Global environment settings."""

    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    separation_backend: Literal["spleeter", "demucs"] = "spleeter"
    spleeter_env_name: str = "py38_spleeter"
    seed: int = 1234


# ============================================================
# Path Configuration
# ============================================================


class PathConfig(BaseModel):
    """All path configurations in one place."""

    # Checkpoints
    checkpoints_dir: Path = Path("checkpoints")
    extractor_model: Path = Path("checkpoints/extractor/latest.pth")
    beat_detector_model: Path = Path("checkpoints/beat_detector/latest.pt")
    decoder_model: Path = Path("checkpoints/decoder/latest.pth")
    decoder_config: Path = Path("checkpoints/decoder/etude_decoder_config.json")
    decoder_vocab: Path = Path("checkpoints/decoder/vocab.json")
    hft_model: Path = Path("checkpoints/hft_transformer/latest.pkl")

    # Dataset
    dataset_dir: Path = Path("dataset")
    raw_dir: Path = Path("dataset/raw")
    processed_dir: Path = Path("dataset/processed")
    aligned_dir: Path = Path("dataset/aligned")
    tokenized_dir: Path = Path("dataset/tokenized")
    dataset_vocab: Path = Path("dataset/vocab.json")
    dataset_csv: Path = Path("assets/dataset.csv")

    # Outputs
    outputs_dir: Path = Path("outputs")
    train_output_dir: Path = Path("outputs/train")
    infer_output_dir: Path = Path("outputs/infer")
    eval_output_dir: Path = Path("outputs/evaluation")


# ============================================================
# Model Configurations
# ============================================================


class ExtractorFeatureConfig(BaseModel):
    """Feature extraction parameters for AMT-APC extractor."""

    sr: int = 16000
    hop_sample: int = 256
    mel_bins: int = 256
    n_bins: int = 256
    fft_bins: int = 2048
    window_length: int = 2048
    log_offset: float = 1e-8
    window: str = "hann"
    pad_mode: str = "constant"


class ExtractorInputConfig(BaseModel):
    """Input parameters for AMT-APC extractor."""

    margin_b: int = 32
    margin_f: int = 32
    num_frame: int = 512
    min_value: float = -18.0


class ExtractorMidiConfig(BaseModel):
    """MIDI representation parameters."""

    note_min: int = 21
    note_max: int = 108
    num_note: int = 88
    num_velocity: int = 128


class ExtractorModelConfig(BaseModel):
    """Model architecture parameters for AMT-APC."""

    cnn_channel: int = 4
    cnn_kernel: int = 5
    dropout: float = 0.1
    transformer_hid_dim: int = 256
    transformer_pf_dim: int = 512
    encoder_n_head: int = 4
    encoder_n_layer: int = 3
    decoder_n_head: int = 4
    decoder_n_layer: int = 3
    sv_dim: int = 24


class ExtractorInferConfig(BaseModel):
    """Inference parameters for AMT-APC extractor."""

    onset_threshold: float = 0.5
    offset_threshold: float = 1.0
    frame_threshold: float = 0.5
    min_duration: float = 0.08


class ExtractorConfig(BaseModel):
    """Complete AMT-APC extractor configuration."""

    feature: ExtractorFeatureConfig = Field(default_factory=ExtractorFeatureConfig)
    input: ExtractorInputConfig = Field(default_factory=ExtractorInputConfig)
    midi: ExtractorMidiConfig = Field(default_factory=ExtractorMidiConfig)
    model: ExtractorModelConfig = Field(default_factory=ExtractorModelConfig)
    infer: ExtractorInferConfig = Field(default_factory=ExtractorInferConfig)


class BeatDetectorModelConfig(BaseModel):
    """Model architecture for Beat-Transformer."""

    attn_len: int = 5
    instr: int = 5
    ntoken: int = 2
    dmodel: int = 256
    nhead: int = 8
    d_hid: int = 1024
    nlayers: int = 9
    norm_first: bool = True


class BeatDetectorConfig(BaseModel):
    """Beat-Transformer configuration."""

    # DBN parameters
    min_bpm: float = 70.0
    max_bpm: float = 250.0
    fps_divisor: int = 1024
    threshold: float = 0.2
    beats_per_bar: List[int] = Field(default=[3, 4])

    # Model architecture
    model: BeatDetectorModelConfig = Field(default_factory=BeatDetectorModelConfig)


class HFTFeatureConfig(BaseModel):
    """Feature extraction for HFT-Transformer."""

    sr: int = 16000
    hop_sample: int = 256
    mel_bins: int = 256
    n_bins: int = 256
    fft_bins: int = 2048
    window_length: int = 2048
    log_offset: float = 1e-8
    window: str = "hann"
    pad_mode: str = "constant"


class HFTInputConfig(BaseModel):
    """Input parameters for HFT-Transformer."""

    margin_b: int = 32
    margin_f: int = 32
    num_frame: int = 128
    min_value: float = -80.0


class HFTInferConfig(BaseModel):
    """Inference parameters for HFT-Transformer."""

    mode: str = "combination"
    thred_mpe: float = 0.5
    thred_onset: float = 0.75
    thred_offset: float = 0.5
    n_stride: int = 32
    bpm: float = 120.0


class HFTConfig(BaseModel):
    """HFT-Transformer (transcription) configuration."""

    feature: HFTFeatureConfig = Field(default_factory=HFTFeatureConfig)
    input: HFTInputConfig = Field(default_factory=HFTInputConfig)
    midi: ExtractorMidiConfig = Field(default_factory=ExtractorMidiConfig)
    infer: HFTInferConfig = Field(default_factory=HFTInferConfig)


class DecoderConfig(BaseModel):
    """EtudeDecoder model configuration."""

    # Architecture
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024

    # Task-specific
    num_classes: int = 3
    num_attribute_bins: int = 3
    attribute_emb_dim: int = 64
    pad_class_id: int = 0
    attribute_pad_id: int = 0
    context_num_past_xy_pairs: int = 4

    # Generation
    temperature: float = 0.0
    top_p: float = 0.9
    max_output_tokens: int = 25600
    max_bar_token_limit: int = 512


# ============================================================
# Pipeline Configurations
# ============================================================


class PrepareDownloadConfig(BaseModel):
    """Download stage configuration."""

    # No additional params beyond paths


class PrepareAlignConfig(BaseModel):
    """Align & filter stage configuration."""

    wp_std_threshold: float = 1.0


class PrepareTokenizeConfig(BaseModel):
    """Tokenize stage configuration."""

    save_format: str = "npy"


class PrepareConfig(BaseModel):
    """Data preparation pipeline configuration."""

    align: PrepareAlignConfig = Field(default_factory=PrepareAlignConfig)
    tokenize: PrepareTokenizeConfig = Field(default_factory=PrepareTokenizeConfig)


class TrainConfig(BaseModel):
    """Training configuration."""

    # Run
    run_id: str = ""
    resume_from_checkpoint: Optional[str] = None

    # Data
    data_format: str = "npy"
    num_workers: int = 4

    # Training
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    num_epochs: int = 200
    warmup_epochs: int = 10
    gradient_accumulation_steps: int = 4
    clip_grad_norm: float = 1.0
    scheduler: str = "cosine_with_warmup"

    # Checkpointing
    save_every_n_epochs: int = 10


class InferConfig(BaseModel):
    """Inference configuration."""

    cleanup_intermediate: bool = False


class EvalMetricsConfig(BaseModel):
    """Evaluation metrics configuration."""

    # WPD
    wpd_subsample_step: int = 1
    wpd_trim_seconds: int = 10

    # RGC
    rgc_top_k: int = 8

    # IPE
    ipe_n_gram: int = 8
    ipe_n_clusters: int = 16


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    eval_dir: Path = Path("docs/songs")
    metadata_path: Path = Path("docs/songs/metadata.json")
    report_image_filename: str = "evaluation_summary.png"
    report_csv_filename: str = "evaluation_results.csv"

    versions: Dict[str, str] = Field(
        default={
            "human": "Human",
            "etude_e": "Etude Extractor",
            "etude_d_d": "Etude Decoder - Default",
            "etude_d": "Etude Decoder - Prompted",
            "picogen": "PiCoGen",
            "amtapc": "AMT-APC",
            "music2midi": "Music2MIDI",
        }
    )

    metrics: EvalMetricsConfig = Field(default_factory=EvalMetricsConfig)


# ============================================================
# Root Configuration
# ============================================================


class EtudeConfig(BaseModel):
    """
    Root configuration for the Etude project.

    All parameters have sensible defaults. Override via YAML or programmatically.

    Example YAML (only specify what differs from defaults):
        env:
          separation_backend: "demucs"
        decoder:
          temperature: 0.8
    """

    env: EnvConfig = Field(default_factory=EnvConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    # Models
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    beat_detector: BeatDetectorConfig = Field(default_factory=BeatDetectorConfig)
    hft: HFTConfig = Field(default_factory=HFTConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)

    # Pipelines
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
