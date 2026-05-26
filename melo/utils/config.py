from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class TrainConfig(BaseModel):
    log_interval: int = 200
    eval_interval: int = 1000
    seed: int = 52
    epochs: int = 10000
    learning_rate: float = 0.0003
    betas: List[float] = [0.8, 0.99]
    eps: float = 1e-09
    batch_size: int = 6
    fp16_run: bool = False
    lr_decay: float = 0.999875
    segment_size: int = 16384
    init_lr_ratio: float = 1.0
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    skip_optimizer: bool = True

class DataConfig(BaseModel):
    training_files: str = ""
    validation_files: str = ""
    max_wav_value: float = 32768.0
    sampling_rate: int = 44100
    filter_length: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    n_mel_channels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = None
    add_blank: bool = True
    n_speakers: int = 256
    cleaned_text: bool = True
    spk2id: Dict[str, int] = Field(default_factory=dict)
    disable_bert: bool = False

class ModelConfig(BaseModel):
    use_spk_conditioned_encoder: bool = True
    use_noise_scaled_mas: bool = True
    use_mel_posterior_encoder: bool = False
    use_duration_discriminator: bool = True
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    n_layers_trans_flow: int = 3
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock: str = "1"
    resblock_kernel_sizes: List[int] = [3, 7, 11]
    resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    upsample_rates: List[int] = [8, 8, 2, 2, 2]
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = [16, 16, 8, 2, 2]
    n_layers_q: int = 3
    use_spectral_norm: bool = False
    gin_channels: int = 256

class ConfigSchema(BaseModel):
    train: TrainConfig
    data: DataConfig
    model: ModelConfig
    
    # Extra fields added during runtime (CLI arguments)
    model_dir: Optional[str] = None
    pretrain_G: Optional[str] = None
    pretrain_D: Optional[str] = None
    pretrain_dur: Optional[str] = None
    port: Optional[int] = None
