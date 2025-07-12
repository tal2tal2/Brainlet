from models.moe.MoE import MixtureOfExperts
from settings import ModelConfig


def get_model_checkpoint(checkpoint_path: str, model_config:ModelConfig):
    return MixtureOfExperts.load_from_checkpoint(checkpoint_path, config=model_config)
