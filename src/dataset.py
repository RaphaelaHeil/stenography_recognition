import json
from enum import Enum
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from src.configuration import DataMode
from src.utils.encoder import TranscriptionEncoder


class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    TEST_LH = 4
    TEST_OOD = 5


PAD_token = 0


def __buildFilename__(rootDir: Path, filebase: str, dataMode: DataMode):
    if dataMode == DataMode.CLEAN:
        return rootDir / f"clean_{filebase}"
    elif dataMode == DataMode.MIXED:
        return rootDir / f"mixed_{filebase}"
    elif dataMode == DataMode.ADDED_MIXED:
        return rootDir / f"added_mixed_{filebase}"
    elif dataMode == DataMode.ADDED:
        return rootDir / f"added_{filebase}"
    elif dataMode == DataMode.STRUCK:
        return rootDir / f"struck_{filebase}"
    elif dataMode == DataMode.STRUCK_MIXED:
        return rootDir / f"struck_mixed_{filebase}"
    elif dataMode == DataMode.DISTURBED:
        return rootDir / f"disturbed_{filebase}"
    else:
        raise ValueError(f"unknown data mode: {dataMode}")


class LineCharacterDataset(Dataset):

    def __init__(self, rootDir: Path, mode: DatasetMode, imageTransforms: Compose, textTransforms: Compose, fold: int,
                 encoder: TranscriptionEncoder, dataMode: DataMode = DataMode.CLEAN):
        if imageTransforms:
            self.imageTransforms = imageTransforms
        else:
            self.imageTransforms = Compose([ToTensor()])

        if textTransforms:
            self.characterTransforms = textTransforms
        else:
            self.characterTransforms = Compose([])  # TODO: is this really the best way to represent this?

        self.encoder = encoder

        self.imageDir = rootDir
        self.data = []

        if mode in [DatasetMode.TRAIN, DatasetMode.VALIDATION]:
            filename = __buildFilename__(rootDir, f"fold_{fold}.json", dataMode)
            with filename.open("r") as inFile:
                foldData = json.load(inFile)
                if mode == DatasetMode.TRAIN:
                    self.data = foldData["train"]
                else:
                    self.data = foldData["val"]
        elif mode == DatasetMode.TEST_LH:
            with __buildFilename__(rootDir, "test_lh_lines.json", dataMode).open("r") as inFile:
                self.data = json.load(inFile)
        elif mode == DatasetMode.TEST_OOD:
            with __buildFilename__(rootDir, "test_ood_lines.json", dataMode).open("r") as inFile:
                self.data = json.load(inFile)
        else:
            with __buildFilename__(rootDir, "test_lh_lines.json", dataMode).open("r") as inFile:
                self.data = json.load(inFile)
            with __buildFilename__(rootDir, "test_ood_lines.json", dataMode).open("r") as inFile:
                self.data.extend(json.load(inFile))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        lineData = self.data[index]

        transcription = lineData["transcription"]

        transcriptionEncoding = self.encoder.encode(transcription)
        transcriptionEncoding = torch.tensor(transcriptionEncoding)

        lineImage = Image.open(self.imageDir / lineData["filename"]).convert("RGB")

        if self.imageTransforms:
            lineImage = self.imageTransforms(lineImage)

        length = transcriptionEncoding.shape[0]

        if self.characterTransforms:
            transcriptionEncoding = self.characterTransforms(transcriptionEncoding)

        return {"image_name": lineData["filename"], "image": lineImage, "transcription_plaintxt": transcription,
            "transcription": transcriptionEncoding, "t_len": length}


if __name__ == '__main__':
    from src.utils.encoder import BaseEncoder

    base = Path("../data/lines")
    ld = LineCharacterDataset(base, DatasetMode.TRAIN, None, None, 0, BaseEncoder())
    data = ld.__getitem__(0)
    print(data["transcription"], data["transcription_plaintxt"])
