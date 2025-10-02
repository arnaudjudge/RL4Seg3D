import torch
torch.set_float32_matmul_precision('medium')

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from lightning.pytorch.loggers import CometLogger
from omegaconf import OmegaConf
from lightning.pytorch import Trainer, seed_everything

from patchless_nnunet.utils import log_hyperparameters
from rl4seg3d.utils.instantiators import instantiate_callbacks

OmegaConf.register_new_resolver(
    "get_class_name", lambda name: name.split('.')[-1]
)


@hydra.main(version_base=None, config_path="config", config_name="RL_3d_runner")
def main(cfg):
    # Load any available `.env` file
    load_dotenv()

    os.environ["HYDRA_FULL_ERROR"] = "1"
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed)

    logger = instantiate(cfg.logger)

    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule, seed=cfg.seed)

    callbacks = instantiate_callbacks(cfg.callbacks)

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if isinstance(trainer.logger, CometLogger):
        logger.experiment.log_asset_folder(".hydra", log_file_name=True)
        if cfg.get("comet_tags", None):
            logger.experiment.add_tags(list(cfg.comet_tags))

    if logger:
        print("Logging hyperparams")
        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }
        log_hyperparameters(object_dict)

    if cfg.train:
        trainer.fit(train_dataloaders=datamodule, model=model)

    if cfg.trainer.max_epochs > 0 and cfg.train:
        ckpt_path = 'best'
    elif getattr(cfg, "test_from_ckpt", None):
        ckpt_path = cfg.test_from_ckpt
    else:
        ckpt_path = None

    # test with everything
    datamodule.hparams.subset_frac = 1.0
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, inference_mode=False)
    trainer.test(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)

    if getattr(cfg.model, "predict_save_dir", None) and cfg.predict_subset_frac > 0:
        datamodule.hparams.subset_frac = cfg.predict_subset_frac
        trainer.predict(model=model, dataloaders=datamodule, ckpt_path=ckpt_path)
        if cfg.get("save_csv_after_predict", None) and trainer.world_size > 1 and trainer.global_rank == 0:
            for p in Path(f"{model.temp_files_path}/").glob("temp_pred_*.csv"):
                df = pd.read_csv(p, index_col=0)
                datamodule.df.loc[df.index] = df
                os.remove(p)
            datamodule.df.to_csv(cfg.save_csv_after_predict)


if __name__ == "__main__":
    load_dotenv()
    main()
