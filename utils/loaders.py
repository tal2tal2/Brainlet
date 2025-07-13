from models.moe.MoE import MixtureOfExperts
from settings import ModelConfig, GeneratorConfig


def get_model_checkpoint(checkpoint_path: str, model_config:ModelConfig, generator:GeneratorConfig):
    return MixtureOfExperts.load_from_checkpoint(checkpoint_path, config=model_config, generator=generator)
