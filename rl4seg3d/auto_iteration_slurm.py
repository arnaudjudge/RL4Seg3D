import json
import pickle
import subprocess
from datetime import datetime
from pathlib import Path

import os
import hydra
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from rl4seg3d.runner import main as runner_main

import subprocess
import shlex

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


@hydra.main(version_base=None, config_path="config", config_name="auto_iteration")
def main(cfg):
    GlobalHydra.instance().clear()
    initialize(version_base=None, config_path='config')
    load_dotenv()

    iterations = cfg.num_iter
    output_path = cfg.output_path

    checkpoint_dict = {}
    if cfg.continue_from_ckpt:
        checkpoint_dict = json.load(open(f"{output_path}/checkpoint_dict.json", "r"))
    else:
        Path(output_path + "/0/").mkdir(parents=True, exist_ok=True)
        checkpoint_dict['main_overrides'] = [f"logger.save_dir={output_path}"]
        checkpoint_dict["target_experiment"] = cfg.target
        checkpoint_dict['source_experiment'] = cfg.source

        timestamp = datetime.now().timestamp()
        checkpoint_dict['experiment_split_column'] = f"split_{timestamp}"
        checkpoint_dict['experiment_gt_column'] = f"Gt_{timestamp}"
        checkpoint_dict['pretrain_path'] = cfg.get("pretrain_path", None)
        checkpoint_dict['trainer_overrides'] = [f"++trainer.{k}={v}" for k, v in cfg.get("trainer", {}).items()]

        checkpoint_dict['start_data_path'] = os.environ['DATA_PATH']
        checkpoint_dict['current_it'] = 0
        checkpoint_dict['turn'] = 'reward'

    print(f"starting data path: {checkpoint_dict['start_data_path']}")
    if checkpoint_dict['current_it'] < 1:
        if not checkpoint_dict['pretrain_path']:
            # train supervised network for initial actor
            overrides = checkpoint_dict['main_overrides'] + checkpoint_dict['trainer_overrides'] + [f"trainer.max_epochs={cfg.sup_num_epochs}",
                                                              f'model.predict_save_dir={None}',  # no predictions here
                                                              f"model.ckpt_path={output_path}/{0}/actor.ckpt",
                                                              f"model.loss.label_smoothing={cfg.sup_loss_label_smoothing}",
                                                              f"experiment=supervised_{checkpoint_dict['source_experiment']}"]
            sub_cfg = compose(config_name=f"supervised_runner.yaml", overrides=overrides)
            print(OmegaConf.to_yaml(sub_cfg))

            # prepare dataset with custom split and gt column
            if checkpoint_dict['experiment_split_column'] != sub_cfg.datamodule.splits_column:
                df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
                df[checkpoint_dict['experiment_split_column']] = df.loc[:, sub_cfg.datamodule.splits_column]
                df[checkpoint_dict['experiment_gt_column']] = df.loc[:, sub_cfg.datamodule.gt_column]
                df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
            sub_cfg.datamodule.splits_column = checkpoint_dict['experiment_split_column']
            sub_cfg.datamodule.gt_column = checkpoint_dict['experiment_gt_column']
            OmegaConf.save(sub_cfg, "config.yaml")
            # torch.distributed.new_group(ranks=[0, 1, 2, 3], timeout=datetime.timedelta(seconds=1800), backend="nccl"
            subprocess.run(
                shlex.split(f"python {os.environ['RL4SEG3D_HOME_HOME']}/runner.py -cd ./ --config-name=config.yaml +launcher={cfg.run_launcher} hydra.launcher.timeout_min={cfg.sup_time} --multirun"))

        # Predict and test (baseline) on target domain
        overrides = checkpoint_dict['main_overrides'] + checkpoint_dict['trainer_overrides'] + cfg.rl_overrides + [f"trainer.max_epochs=0",
                                                                             f"predict_subset_frac={cfg.rl_num_predict}",
                                                                             f"model.actor.actor.pretrain_ckpt={f'{output_path}/{0}/actor.ckpt' if checkpoint_dict['pretrain_path'] is None else checkpoint_dict['pretrain_path']}",
                                                                             f"model.actor.actor.ref_ckpt={f'{output_path}/{0}/actor.ckpt' if checkpoint_dict['pretrain_path'] is None else checkpoint_dict['pretrain_path']}",
                                                                             "reward@model.reward=pixelwise_accuracy",  # will not be used
                                                                             f"model.actor_save_path={output_path}/{0}/actor.ckpt",  # no need
                                                                             f"model.critic_save_path=null",  # no need
                                                                             f"model.predict_save_dir={output_path}/rewardDS/",
                                                                             f"experiment=ppo_{checkpoint_dict['target_experiment']}",
                                                                             f"++save_csv_after_predict=null",
                                                                             f"++model.temp_files_path={output_path}"
                                                                             ]
        sub_cfg = compose(config_name=f"RL_3d_runner.yaml", overrides=overrides)
        # prepare dataset with custom split and gt column
        if checkpoint_dict['experiment_split_column'] != sub_cfg.datamodule.splits_column:
            df = pd.read_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file, index_col=0)
            df[checkpoint_dict['experiment_split_column']] = df.loc[:, sub_cfg.datamodule.splits_column]
            df[checkpoint_dict['experiment_gt_column']] = df.loc[:, sub_cfg.datamodule.gt_column]
            df.to_csv(sub_cfg.datamodule.data_dir + sub_cfg.datamodule.csv_file)
        sub_cfg.datamodule.splits_column = checkpoint_dict['experiment_split_column']
        sub_cfg.datamodule.gt_column = checkpoint_dict['experiment_gt_column']

        sub_cfg.save_csv_after_predict = \
            f"{sub_cfg.datamodule.data_dir}/{sub_cfg.datamodule.dataset_name}/{sub_cfg.datamodule.csv_file}"
        OmegaConf.save(sub_cfg, "config.yaml")
        subprocess.run(
            shlex.split(f"python {os.environ['RL4SEG3D_HOME_HOME']}/runner.py -cd ./ --config-name=config.yaml +launcher={cfg.run_launcher} hydra.launcher.timeout_min={cfg.test_pred_time} --multirun"))

        checkpoint_dict['current_it'] = 1
        json.dump(checkpoint_dict, open(f"{output_path}/checkpoint_dict.json", "w"))

    for i in range(checkpoint_dict['current_it'], iterations + 1):
        # set OS data path for copy of data to happen
        # copy data to compute node, next to RL data
        # subprocess.run(["rsync", "-a", f"{output_path}/rewardDS/", f"{os.environ['DATA_PATH']}/rewardDS/"])
        # os.environ['DATA_PATH'] = f"{output_path}/rewardDS/"

        if checkpoint_dict['turn'] == 'reward':
            # train reward net
            overrides = checkpoint_dict['main_overrides'] + checkpoint_dict['trainer_overrides'] + [f"trainer.max_epochs={cfg.rn_num_epochs}",
                                                              f"datamodule.data_path={output_path}/rewardDS/",
                                                              f"model.save_model_path={output_path}/{i - 1}/rewardnet.ckpt",
                                                              ]
                                                              # f"+model.var_file={cfg.var_file}"]
            sub_cfg = compose(config_name=f"reward_3d_runner.yaml", overrides=overrides)
            print(OmegaConf.to_yaml(sub_cfg))
            OmegaConf.save(sub_cfg, "config.yaml")
            subprocess.run(
                shlex.split(f"python {os.environ['RL4SEG3D_HOME_HOME']}/runner.py -cd ./ --config-name=config.yaml +launcher={cfg.run_launcher} hydra.launcher.timeout_min={int(cfg.reward_time)*(i+1)} --multirun"))

            checkpoint_dict['turn'] = 'RL'
            checkpoint_dict['current_it'] = i
            json.dump(checkpoint_dict, open(f"{output_path}/checkpoint_dict.json", "w"))

        next_output_path = f'{output_path}/{i}/'
        Path(next_output_path).mkdir(parents=True, exist_ok=True)

        # TODO: MAYBE RETHINK THIS WAY OF PASSING THE VARIABLE(S)
        # load temporary variable file
        # saved_vars = pickle.load(open(cfg.var_file, "rb"))

        os.environ['DATA_PATH'] = checkpoint_dict['start_data_path']
        # train PPO model with fresh reward net
        overrides = checkpoint_dict['main_overrides'] + checkpoint_dict['trainer_overrides'] + cfg.rl_overrides + \
                    [f"trainer.max_epochs={cfg.rl_num_epochs}",
                     f"predict_subset_frac={cfg.rl_num_predict}",
                     f"datamodule.splits_column={checkpoint_dict['experiment_split_column']}",
                     f"datamodule.gt_column=null", #{checkpoint_dict['experiment_gt_column']}",
                     # f"+datamodule.train_batch_size={8 * i}",
                     f"model.actor.actor.pretrain_ckpt={output_path}/{i - 1}/actor.ckpt",
                     f"model.actor.actor.ref_ckpt={output_path}/{i - 1}/actor.ckpt",
                     f"reward@model.reward={cfg.rl_reward_config_name}"
                     f"model.reward.state_dict_path.anatomical={output_path}/{i - 1}/rewardnet.ckpt",
                     # f"model.reward.temp_factor={float(saved_vars['Temperature_factor'])}",
                     f"model.actor_save_path={output_path}/{i}/actor.ckpt",
                     f"model.critic_save_path={output_path}/{i}/critic.ckpt",
                     f'model.predict_save_dir={f"{output_path}/rewardDS/"}',
                     f"model.entropy_coeff={0.15}",
                     f"model.divergence_coeff={0.015}",
                     f"experiment=ppo_{checkpoint_dict['target_experiment']}",
                     f"++save_csv_after_predict=null",
                     f"++model.temp_files_path={output_path}"
                     ]
        if Path(f"{output_path}/{i - 1}/critic.ckpt").exists():
            overrides += [f"model.actor.critic.pretrain_ckpt={output_path}/{i - 1}/critic.ckpt"]
        sub_cfg = compose(config_name=f"RL_3d_runner.yaml", overrides=overrides)

        sub_cfg.save_csv_after_predict = \
            f"{sub_cfg.datamodule.data_dir}/{sub_cfg.datamodule.dataset_name}/{sub_cfg.datamodule.csv_file}"
        print(OmegaConf.to_yaml(sub_cfg))
        OmegaConf.save(sub_cfg, "config.yaml")
        subprocess.run(
            shlex.split(f"python {os.environ['RL4SEG3D_HOME_HOME']}/runner.py -cd ./ --config-name=config.yaml +launcher={cfg.run_launcher} hydra.launcher.timeout_min={int(cfg.rl_time)*(i+1)} --multirun"))

        checkpoint_dict['current_it'] = i
        checkpoint_dict['turn'] = 'reward'
        json.dump(checkpoint_dict, open(f"{output_path}/checkpoint_dict.json", "w"))


if __name__ == '__main__':
    # Load any available `.env` file
    load_dotenv()
    main()
