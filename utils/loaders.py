from models.moe.MoE import MixtureOfExperts


def get_model_checkpoint(checkpoint_path: str, model_config):
    return MixtureOfExperts.load_checkpoint(checkpoint_path, config=model_config)
