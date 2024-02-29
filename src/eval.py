import os
import sys
from typing import Any, Dict, List, Tuple

import hydra
from dotenv import load_dotenv
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

currentdir = os.path.dirname(os.path.realpath(__file__))   # nosec B703, B308
sys.path.append(os.path.dirname(currentdir))   # nosec B703, B308

from src.utils import RankedLogger, extras, instantiate_loggers, log_hyperparameters, task_wrapper

load_dotenv()

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>')
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f'Instantiating model <{cfg.model._target_}>')
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info('Instantiating loggers...')
    logger: List[Logger] = instantiate_loggers(cfg.get('logger'))

    log.info(f'Instantiating trainer <{cfg.trainer._target_}>')
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        'cfg': cfg,
        'datamodule': datamodule,
        'model': model,
        'logger': logger,
        'trainer': trainer,
    }

    if logger:
        log.info('Logging hyperparameters!')
        log_hyperparameters(object_dict)

    log.info('Starting testing!')
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base='1.3', config_path='../configs', config_name='eval.yaml')
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Parameters
    ----------
    cfg : DictConfig
        A DictConfig configuration composed by Hydra.

    Returns
    -------
    Optional[float]
        Optimized metric value
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == '__main__':
    main()
