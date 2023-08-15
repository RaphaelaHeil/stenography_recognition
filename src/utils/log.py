import logging
from typing import List

from src.configuration import Configuration


def initLoggers(config: Configuration, infoLoggerName: str = "info", auxLoggerNames: List[str] = None):
    logger = logging.getLogger(infoLoggerName)
    logger.setLevel(logging.INFO)

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    formatter = logging.Formatter('%(asctime)s - %(message)s', '%d-%b-%y %H:%M:%S')

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler(config.outDir / "info.log", mode="a")
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if auxLoggerNames:
        for auxName in auxLoggerNames:
            auxLogger = logging.getLogger(auxName)
            while auxLogger.hasHandlers():
                auxLogger.removeHandler(auxLogger.handlers[0])

            auxLogger.setLevel(logging.INFO)

            fileHandler = logging.FileHandler(config.outDir / f"{auxName}.log", mode="w")
            fileHandler.setLevel(logging.INFO)
            auxLogger.addHandler(fileHandler)
