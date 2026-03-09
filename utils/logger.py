# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:26:17 2026

@author: kubota
"""

# utils/logger.py

import logging
import os
import sys
from datetime import datetime


def setup_logger(logs_root="logs", enable_print_redirect=False):
    os.makedirs(logs_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_root, f"{timestamp}.log")

    logger = logging.getLogger("CondNet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file, timestamp


def close_logger(logger):
    handlers = logger.handlers[:]
    for h in handlers:
        h.close()
        logger.removeHandler(h)