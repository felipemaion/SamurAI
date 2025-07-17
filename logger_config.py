import logging


def get_logger(name: str) -> logging.Logger:
    """
    Retorna um logger já configurado.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # só configura uma vez
        logger.setLevel(logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # evita mensagens duplicadas

    return logger
