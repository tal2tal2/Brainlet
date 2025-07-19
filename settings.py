from enum import Enum
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LightningStrategy(str, Enum):
    AUTO = 'auto'
    DDP = 'ddp'


class LightningAccelerator(str, Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class ModelOptimizer(str, Enum):
    Adam = 'Adam'
    AdamW = 'AdamW'


class TrainerConfig(BaseModel):
    max_epochs: int = 20
    accumulate_grad_batches: int = 1


class DataLoaderConfig(BaseModel):
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


class LightningConfig(BaseModel):
    accelerator: LightningAccelerator = LightningAccelerator.GPU
    strategy: LightningStrategy = LightningStrategy.AUTO
    devices: str = "auto"
    precision: str = "bf16-mixed"
    num_nodes: int = 1


class ModelConfig(BaseModel):
    num_experts: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    model_optimizer: str = "adam"
    lambda_peaky:float = 0.1
    lambda_diverse:float = 0.04
    lambda_phys:float = 0.3


class GeneratorConfig(BaseModel):
    n_series: int = 20
    series_length: int = 10000
    dt: float = 0.001
    use_slds: bool = False
    input_len: int = 5
    target_len: int = 3

    def get_generator_params(self) -> dict:
        return {'n_series': self.n_series,
                'series_length': self.series_length,
                'dt': self.dt,
                'use_slds': self.use_slds,}


class Config(BaseSettings):
    data_dir: str = r"D:\TheFolder\projects\School\Master\yoni\data"
    use_fake_dataset: bool = True
    random_seed: int = 42
    checkpoint: Optional[str] = None
    save_predictions: bool = False

    training_set: float = 0.8
    validation_set: float = 0.1

    trainer: TrainerConfig = TrainerConfig()
    data: DataLoaderConfig = DataLoaderConfig()
    lightning: LightningConfig = LightningConfig()
    generator: GeneratorConfig = GeneratorConfig()
    model: ModelConfig = ModelConfig()
    model_cb: Optional[ModelCheckpoint] = None
    model_config = SettingsConfigDict(
        env_file='config.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )

    def get_predictor_params(self) -> dict:
        params = self.lightning.model_dump()
        return params

    def get_trainer_params(self) -> dict:
        params = self.lightning.model_dump()
        params.update(self.trainer.model_dump())
        self.model_cb = ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                verbose=True)
        params['callbacks'] = [
            # EarlyStopping(
            #     monitor="train_loss_epoch",
            #     patience=8,
            #     min_delta=0.01),
            self.model_cb,
        ]
        params['logger'] = WandbLogger()
        return params

    def get_wandb_config(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "model": self.model.model_dump(),
            "trainer": self.trainer.model_dump(),
            "lightning": self.lightning.model_dump(),
            "data loader": self.data.model_dump(),
            "random_seed": self.random_seed,
            "generator": self.generator.model_dump(),
        }

    def predict(self) -> None:
        self.generator.n_series=1
        self.generator.series_length=15000

