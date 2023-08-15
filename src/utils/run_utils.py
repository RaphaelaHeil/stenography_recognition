from collections import OrderedDict

import torch
from torch.nn import Module
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Grayscale

from src import dataset
from src.configuration import Configuration, ModelName, EncodingMethod, ModelState
from src.models import CRNN, Flor, GRUCNN, CRNN2, GatedBN
from src.utils.encoder import BaseEncoder, TranscriptionEncoder, VowelEncoder, NgramEncoder, CharacterShortformEncoder, \
    SuffixEncoder, PhoneticEncoder, IpaEncoder, SuffixAndCharShortformEncoder, ShortformSymbolEncoder, MelinEncoder
from src.utils.transforms import PadSequence, ResizeAndPad, ResizeToHeight


def __loadGatedCheckpoint__(model, config):
    bareStateDict = model.state_dict()
    if config.finetune:
        if config.e2ePath.exists():
            state_dict = torch.load(config.e2ePath, map_location=torch.device(config.device))
            if 'model_state_dict' in state_dict.keys():
                state_dict = state_dict['model_state_dict']
            if "head.gru1.weight_ih_l0" not in state_dict.keys():
                modified = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("cnn"):
                        modified[k] = v
                    else:
                        modified[f"head.{k}"] = v
                state_dict = modified
            if config.backboneMode == ModelState.INIT:
                print("backbone here")
                modified = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("cnn"):
                        modified[k] = bareStateDict[k]
                    else:
                        modified[k] = v
                model.load_state_dict(modified)
            elif config.headMode == ModelState.INIT:
                print("here")
                modified = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("cnn"):
                        modified[k] = v
                    else:
                        modified[k] = bareStateDict[k]
                model.load_state_dict(modified)
            else:
                model.load_state_dict(state_dict)
        elif config.backbonePath.exists():
            state_dict = torch.load(config.backbonePath, map_location=torch.device(config.device))
            if 'model_state_dict' in state_dict.keys():
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
        elif config.headPath.exists():
            state_dict = torch.load(config.headPath, map_location=torch.device(config.device))
            if 'model_state_dict' in state_dict.keys():
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
        if config.headMode == ModelState.FREEZE:
            model.head.requires_grad_(False)
            pass
        if config.backboneMode == ModelState.FREEZE:
            model.cnn.requires_grad_(False)
    return model


def getModel(config: Configuration, alphabetSize: int = 51) -> Module:
    if config.modelName == ModelName.GATED_BN:
        model = GatedBN(alphabetSize)
        return __loadGatedCheckpoint__(model, config)
    elif config.modelName == ModelName.GATED:
        model = Flor(alphabetSize)
        return __loadGatedCheckpoint__(model, config)
    else:
        return None


def getTranscriptionEncoder(config: Configuration) -> TranscriptionEncoder:
    if config.characterEncodingMethod == EncodingMethod.BASE:
        return BaseEncoder()
    elif config.characterEncodingMethod == EncodingMethod.NGRAM:
        return NgramEncoder()
    elif config.characterEncodingMethod == EncodingMethod.CHAR_SHORTFORM:
        return CharacterShortformEncoder()
    elif config.characterEncodingMethod == EncodingMethod.SUFFIX:
        return SuffixEncoder()
    elif config.characterEncodingMethod == EncodingMethod.MELIN:
        return MelinEncoder()
    else:
        raise NotImplementedError()


def composeTextTransformation(config: Configuration) -> Compose:
    if config.batchSize > 1:
        return transforms.Compose([PadSequence(length=config.transcriptionLength, padwith=dataset.PAD_token)])
    return transforms.Compose([])


def composeImageTransformation(config: Configuration) -> Compose:
    t = [Grayscale(num_output_channels=1)]
    if config.batchSize > 1:
        t.append(ResizeAndPad(height=config.padHeight, width=config.padWidth, padwith=config.padValue))
    else:
        t.append(ResizeToHeight(size=config.padHeight))
    t.append(ToTensor())
    return Compose(t)
