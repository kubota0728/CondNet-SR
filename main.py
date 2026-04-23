# -*- coding: utf-8 -*-
"""
CLI entry point for CondNet-SR.

Parses --config, sets up logging, and delegates the actual pipeline to
engine/runner.py. The same runner is also used by the (local-only) GUI,
so CLI and GUI produce identical results given the same YAML.

Usage:
    python main.py --config configs/config.yaml
"""
import argparse
import warnings

import yaml

from utils.logger import setup_logger, close_logger
from engine.runner import prepare_output_dirs, run_train, run_eval


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda.nccl",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger, log_file, timestamp = setup_logger(logs_root="logs", enable_print_redirect=False)
    ckpt_dir = prepare_output_dirs(cfg, timestamp)

    logger.info("===== Program Started =====")
    logger.info(f"config: {args.config}")
    logger.info(f"log_file: {log_file}")
    logger.info(f"ckpt_dir: {ckpt_dir}")

    logger.info("===== Config =====")
    logger.info("\n" + yaml.dump(cfg, sort_keys=False, allow_unicode=True))

    try:
        mode = str(cfg.get("run", {}).get("mode", "train")).lower()

        if mode == "train":
            run_train(cfg, logger=logger, ckpt_dir=ckpt_dir)
        elif mode in ("eval", "test", "predict"):
            run_eval(cfg, logger=logger, ckpt_dir=ckpt_dir)
        else:
            raise ValueError(f"Unknown run.mode: {mode}")

        logger.info("===== Program Finished =====")

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise
    finally:
        close_logger(logger)
        print("Logger closed.")


if __name__ == "__main__":
    main()
