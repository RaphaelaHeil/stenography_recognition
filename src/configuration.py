"""
Contains all code related to the configuration of experiments.
"""
import configparser
import random
from argparse import Namespace
from configparser import SectionProxy
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import List

import torch


class ModelName(Enum):
    GATED = auto()
    GATED_BN = auto()

    @staticmethod
    def getByName(name: str) -> "ModelName":
        """

        Args:
            name: string representation that should be converted to a ModelName

        Returns:
            ModelName

        Raises:
            LookupError: if the given name does not correspond to a model name

        """
        if name.upper() in [model.name for model in ModelName]:
            return ModelName[name.upper()]
        else:
            raise LookupError(f"unknown model name: {name}")


class DecodingMethod(Enum):
    GREEDY = auto()

    @staticmethod
    def getByName(name: str) -> "DecodingMethod":
        """

        Args:
            name: string representation that should be converted to a DecodingMethod

        Returns:
            DecodingMethod

        Raises:
            LookupError: if the given name does not correspond to a supported decoding method

        """
        if name.upper() in [model.name for model in DecodingMethod]:
            return DecodingMethod[name.upper()]
        else:
            raise LookupError(f"unknown decoding method: {name}")


class EncodingMethod(Enum):
    BASE = auto()
    NGRAM = auto()
    CHAR_SHORTFORM = auto()
    SUFFIX = auto()
    MELIN = auto()

    @staticmethod
    def getByName(name: str) -> "EncodingMethod":
        """

        Args:
            name: string representation that should be converted to a EncodingMethod

        Returns:
            EncodingMethod

        Raises:
            LookupError: if the given name does not correspond to a supported decoding method

        """
        if name.upper() in [model.name for model in EncodingMethod]:
            return EncodingMethod[name.upper()]
        else:
            raise LookupError(f"unknown encoding method: {name}")


class DataMode(Enum):
    CLEAN = auto()
    DISTURBED = auto()
    MIXED = auto()
    STRUCK_MIXED = auto()
    ADDED_MIXED = auto()
    STRUCK = auto()
    ADDED = auto()

    @staticmethod
    def getByName(name: str) -> "DataMode":
        """

        Args:
            name: string representation that should be converted to a DataMode

        Returns:
            DataMode

        Raises:
            LookupError: if the given name does not correspond to a supported data mode

        """
        if name.upper() in [model.name for model in DataMode]:
            return DataMode[name.upper()]
        else:
            raise LookupError(f"unknown data mode: {name}")


class ModelState(Enum):
    INIT = auto()
    FREEZE = auto()
    FINETUNE = auto()

    @staticmethod
    def getByName(name: str) -> "ModelState":
        """

        Args:
            name: string representation that should be converted to a ModelState

        Returns:
            ModelStateMode

        Raises:
            LookupError: if the given name does not correspond to a supported model state

        """
        if name.upper() in [model.name for model in ModelState]:
            return ModelState[name.upper()]
        else:
            raise LookupError(f"unknown model state mode: {name}")


class Configuration:
    """
    Holds the configuration for the current experiment.
    """

    def __init__(self, parsedConfig: SectionProxy, test: bool = False, fileSection: str = "DEFAULT",
                 filename: Path = None):
        self.parsedConfig = parsedConfig
        self.fileSection = fileSection

        self.headMode = ModelState.getByName(self.parsedConfig.get("head", "init"))
        self.backboneMode = ModelState.getByName(self.parsedConfig.get("backbone", "init"))

        self.finetune = not (self.headMode == ModelState.INIT and self.backboneMode == ModelState.INIT)

        if test:
            self.outDir = filename
        else:
            outDirName = f"{self.parsedConfig.get('model')}_{fileSection}_" \
                         f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{random.randint(0, 100000)}"
            if self.finetune:
                outDirName = f"finetune_{outDirName}"
            self.outDir = Path(self.parsedConfig.get("out_dir")).resolve() / outDirName
            self.parsedConfig["out_dir"] = str(self.outDir)

        self.outDir.mkdir(parents=True, exist_ok=True)

        if self.finetune:
            self.backbonePath = Path(parsedConfig.get("backbone_path", "")).resolve()
            self.headPath = Path(parsedConfig.get("head_path", "")).resolve()
            self.e2ePath = Path(parsedConfig.get("e2e_path", "")).resolve()

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.epochs = self.getSetInt("epochs", 100)
        self.learningRate = self.getSetFloat("learning_rate", 0.001)
        self.earlyStoppingEpochCount = self.getSetInt("early_stopping_epoch_count", -1)

        self.batchSize = self.getSetInt("batch_size", 4)
        self.modelSaveEpoch = self.getSetInt("model_save_epoch", 10)
        self.validationEpoch = self.getSetInt("validation_epoch", 1)
        self.dataDir = Path(self.getSetStr("data_dir")).resolve()
        self.fold = self.getSetInt("fold", 0)
        self.dataMode = DataMode.getByName(self.getSetStr("data_mode", "clean"))
        self.validationDataMode = DataMode.getByName(self.getSetStr("val_data_mode", "mixed"))

        self.transcriptionLength = self.getSetInt("transcription_length", 72)
        self.padHeight = self.getSetInt('pad_height', 64)  # 128
        self.padWidth = self.getSetInt('pad_width', 1230)  # 2460
        self.padValue = self.getSetInt("pad_value", 0)

        self.modelName = ModelName.getByName(self.getSetStr("model", "lstm"))
        self.decodingMethod = DecodingMethod.getByName(self.getSetStr("decoding", "greedy"))
        self.characterEncodingMethod = EncodingMethod.getByName(self.getSetStr("encoding", "base"))

        self.testModelFileName = self.getSetStr("test_model_filename", "best_val_loss.pth")

        self.clipNorm = self.getSetFloat("clip_norm", 1)
        self.warmup = self.getSetInt("warmup_epochs", 0)

        if not test:
            configOut = self.outDir / "config.cfg"
            with configOut.open("w+") as cfile:
                parsedConfig.parser.write(cfile)

    def getSetInt(self, key: str, default: int = None):
        value = self.parsedConfig.getint(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetFloat(self, key: str, default: float = None):
        value = self.parsedConfig.getfloat(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetBoolean(self, key: str, default: bool = None):
        value = self.parsedConfig.getboolean(key, default)
        self.parsedConfig[key] = str(value)
        return value

    def getSetStr(self, key: str, default: str = None):
        value = self.parsedConfig.get(key, default)
        self.parsedConfig[key] = str(value)
        return value

    @staticmethod
    def parseCSList(configString: str) -> List[str]:
        split = configString.split(",")
        result = [s.strip() for s in split]
        return result


def getConfiguration(args: Namespace) -> Configuration:
    """
    Loads the configuration based on the given arguments ``args``.

    Relevant arguments:
        - ``file``: path to config file, default: 'config.cfg'
        - ``section``: config file section to load, default: DEFAULT
        - ``test``: whether to load the config in train or test mode, default: False
    Args:
        args: arguments required to load the configuration

    Returns:
        the parsed configuration

    """
    fileSection = "DEFAULT"
    fileName = "config.cfg"
    test = False
    if "section" in args:
        fileSection = args.section
    if "file" in args:
        fileName = args.file.resolve()
    if "test" in args:
        test = args.test
    configParser = configparser.ConfigParser()
    configParser.read(fileName)

    if test:
        if len(configParser.sections()) > 0:
            parsedConfig = configParser[configParser.sections()[0]]
        else:
            parsedConfig = configParser["DEFAULT"]
    else:
        parsedConfig = configParser[fileSection]
        sections = configParser.sections()
        for s in sections:
            if s != fileSection:
                configParser.remove_section(s)
    if test:
        return Configuration(parsedConfig, fileSection=fileSection, test=test, filename=fileName.parent)
    else:
        return Configuration(parsedConfig, fileSection=fileSection, test=test)
