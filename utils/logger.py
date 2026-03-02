# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:26:17 2026

@author: kubota
"""

# utils/logger.py
import logging
import os
import sys


def setup_logger(logs_dir, enable_print_redirect=False):
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "log.txt")

    logger = logging.getLogger("CondNet")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for h in handlers:
        h.close()
        logger.removeHandler(h)