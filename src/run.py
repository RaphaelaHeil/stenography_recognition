import json
import logging
import sys
import time
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from src.configuration import getConfiguration, Configuration, DecodingMethod, DataMode
from src.dataset import LineCharacterDataset, DatasetMode
from src.utils.log import initLoggers
from src.utils.run_utils import composeImageTransformation, composeTextTransformation, getModel, getTranscriptionEncoder


class EvalMode(Enum):
    NONE = 1
    VALIDATION = 2
    TEST = 3


class Runner:

    def __init__(self, config: Configuration, evalMode: EvalMode = EvalMode.NONE, outFileName: str = "test.json"):
        self.config = config
        self.outFileName = outFileName

        self.transcriptionEncoder = getTranscriptionEncoder(self.config)

        self.model = getModel(self.config, self.transcriptionEncoder.alphabetSize())

        if evalMode != EvalMode.NONE:
            state_dict = torch.load(self.config.outDir / self.config.testModelFileName,
                                    map_location=torch.device(config.device))
            if 'model_state_dict' in state_dict.keys():
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.config.device)

        self.loss = CTCLoss(zero_infinity=True)

        self.optimiser = AdamW(self.model.parameters(), lr=self.config.learningRate)

        imageTransform = composeImageTransformation(self.config)
        textTransform = composeTextTransformation(self.config)

        # set number of dataloader workers according to whether debug is active or not:
        numWorkers = 1
        gettrace = getattr(sys, "gettrace", None)
        if gettrace and gettrace():
            numWorkers = 0

        trainDataset = LineCharacterDataset(config.dataDir, DatasetMode.TRAIN, imageTransform, textTransform,
                                            self.config.fold, self.transcriptionEncoder, dataMode=self.config.dataMode)
        self.trainDataloader = DataLoader(trainDataset, batch_size=self.config.batchSize, shuffle=True,
                                          num_workers=numWorkers)

        if evalMode == EvalMode.TEST:
            evalDataset = LineCharacterDataset(config.dataDir, DatasetMode.TEST, imageTransform, textTransform,
                                               self.config.fold, self.transcriptionEncoder,
                                               dataMode=self.config.validationDataMode)
        else:
            evalDataset = LineCharacterDataset(config.dataDir, DatasetMode.VALIDATION, imageTransform, textTransform,
                                               self.config.fold, self.transcriptionEncoder,
                                               dataMode=self.config.validationDataMode)
        self.evalDataloader = DataLoader(evalDataset, batch_size=self.config.batchSize, shuffle=False,
                                         num_workers=numWorkers)
        self.infoLogger = logging.getLogger("info")

        if evalMode == EvalMode.NONE:
            self.evalLogger = logging.getLogger("validation")
        elif evalMode == EvalMode.VALIDATION:
            self.evalLogger = logging.getLogger("eval_test")
        else:
            self.evalLogger = logging.getLogger("test")

        if self.config.decodingMethod == DecodingMethod.GREEDY:
            self.decode = self.greedyDecode
        else:
            self.decode = self.greedyDecode

        self.cerMetric = CharErrorRate()
        self.werMetric = WordErrorRate()

        self.bestValLoss = float("inf")
        self.bestValLossEpoch = 0

    def train(self):
        logger = logging.getLogger("train")
        logger.info("epoch,meanBatchLoss")
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            batchLosses = []
            epochStartTime = time.time()
            datasetsize = len(self.trainDataloader)
            for batchId, data in enumerate(self.trainDataloader):
                lineImage = data["image"].to(self.config.device)
                encodedTranscription = data["transcription"].to(self.config.device)
                plaintextTranscription = data["transcription_plaintxt"]
                transcriptionLengths = data["t_len"]

                predicted = self.model(lineImage)
                predicted = predicted.log_softmax(2)

                input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0], dtype=torch.long)

                loss = self.loss(predicted, encodedTranscription, input_lengths, transcriptionLengths)
                loss.backward()

                if self.config.batchSize == 1:
                    if batchId % 5 == 4 or batchId == datasetsize - 1:
                        if self.config.clipNorm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipNorm)
                        self.optimiser.step()
                        self.optimiser.zero_grad()
                else:
                    if self.config.clipNorm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipNorm)
                    self.optimiser.step()
                    self.optimiser.zero_grad()

                batchLosses.append(loss.item())
                if epoch == 1 and batchLosses[-1] == 0.0:
                    self.infoLogger.info(f"{batchLosses[-1]}, {plaintextTranscription}")

            meanBatchLoss = np.mean(batchLosses)
            logger.info(f"{epoch},{meanBatchLoss}")
            self.infoLogger.info(
                    f"[{epoch}/{self.config.epochs}] - loss: {meanBatchLoss}, time: {time.time() - epochStartTime}")
            if epoch > 0 and self.config.modelSaveEpoch > 0 and epoch % self.config.modelSaveEpoch == 0:
                torch.save(self.model.state_dict(), self.config.outDir / Path(f'epoch_{epoch}.pth'))
                self.infoLogger.info(f'Epoch {epoch}: model saved')
            if self.config.validationEpoch > 0 and epoch % self.config.validationEpoch == 0:
                valLoss = self.validate()
                if valLoss < self.bestValLoss:
                    self.bestValLoss = valLoss
                    self.bestValLossEpoch = epoch
                    torch.save(self.model.state_dict(), self.config.outDir / Path('best_val_loss.pth'))
                    self.infoLogger.info(f'Epoch {epoch}: val loss model updated')
            if self.config.earlyStoppingEpochCount > 0 and epoch > self.config.warmup:
                if epoch - self.bestValLossEpoch >= self.config.earlyStoppingEpochCount:
                    self.infoLogger.info(
                            f'No validation loss improvement in {epoch - self.bestValLossEpoch} epochs, stopping training.')
                    break

        self.infoLogger.info(f"Best Val Loss: {self.bestValLoss} ({self.bestValLossEpoch})")

    def greedyDecode(self, predicted) -> List[str]:
        ll = []
        _, max_index = torch.max(predicted, dim=2)
        for i in range(predicted.shape[1]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())

            previous = raw_prediction[0]
            output = [previous]
            for char in raw_prediction[1:]:
                if char == output[-1]:
                    continue
                else:
                    output.append(char)

            result = self.transcriptionEncoder.decode(output)
            ll.append(result)
        return ll

    def validate(self) -> float:
        batchLosses = []
        self.model.eval()
        for batchId, data in enumerate(self.evalDataloader):
            lineImage = data["image"].to(self.config.device)
            encodedTranscription = data["transcription"].to(self.config.device)

            predicted = self.model(lineImage).log_softmax(2)

            input_lengths = torch.full(size=(predicted.shape[1],), fill_value=predicted.shape[0], dtype=torch.long)
            target_lengths = data["t_len"]

            loss = CTCLoss(zero_infinity=True)(predicted, encodedTranscription, input_lengths, target_lengths)
            batchLosses.append(loss.item())

        meanBatchLoss = np.mean(batchLosses)
        self.infoLogger.info(f"{meanBatchLoss}")
        self.evalLogger.info(f"{meanBatchLoss}")
        return meanBatchLoss

    def extractCtcScores(self) -> None:
        self.model.eval()

        outDir = self.config.outDir / "scores"
        outDir.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batchId, data in enumerate(self.evalDataloader):
                lineImage = data["image"].to(self.config.device)
                imageName = data["image_name"]

                predicted = self.model(lineImage)

                for idx in range(len(imageName)):
                    outName = outDir / f"{imageName[idx].split('/')[-1][:-4]}.pt"
                    torch.save(predicted[:, idx, :], outName)

    def test(self) -> None:
        self.model.eval()

        greedyPredictions = []
        originalExpectations = []
        encodedExpectations = []
        transliterations = []

        with torch.no_grad():
            for batchId, data in enumerate(self.evalDataloader):
                lineImage = data["image"].to(self.config.device)
                plaintextTranscription = data["transcription_plaintxt"]

                imageName = data["image_name"]

                predicted = self.model(lineImage)

                results = self.decode(predicted)

                for idx, greedy in enumerate(results):
                    encodedTranscription = self.transcriptionEncoder.replace(plaintextTranscription[idx])
                    greedyPredictions.append(greedy)
                    originalExpectations.append(plaintextTranscription[idx])
                    encodedExpectations.append(encodedTranscription)
                    transliterations.append(
                            {"greedy": greedy, "expected": plaintextTranscription[idx],
                                "expected_encoded": encodedTranscription, "image_name": imageName[idx]})

        greedyMeanOriginalCER = self.cerMetric(greedyPredictions, originalExpectations)
        greedyMeanOriginalWER = self.werMetric(greedyPredictions, originalExpectations)
        greedyMeanEncodedCER = self.cerMetric(greedyPredictions, encodedExpectations)
        greedyMeanEncodedWER = self.werMetric(greedyPredictions, encodedExpectations)
        self.infoLogger.info(f"Greedy Mean Original CER: {greedyMeanOriginalCER}, Greedy Mean Original WER: "
                             f"{greedyMeanOriginalWER}, Greedy Mean Encoded CER: {greedyMeanEncodedCER}, "
                             f"Greedy Mean Encoded WER: {greedyMeanEncodedWER}")

        logging.getLogger("test").info(
                f"Greedy Mean CER: {greedyMeanOriginalCER}, Greedy Mean WER: {greedyMeanOriginalWER}")
        with (self.config.outDir / self.outFileName).open("w") as outFile:
            json.dump(transliterations, outFile, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")  # turn off pytorch user warnings for now

    torch.backends.cudnn.benchmark = True
    argParser = ArgumentParser()
    argParser.add_argument("-file", help="path to config-file", default="config.cfg", type=Path)
    argParser.add_argument("-section", help="section of config-file to use", default="DEFAULT")
    argParser.add_argument("-test", action="store_true", help="if set, will load config in test mode")
    args = argParser.parse_args()

    config = getConfiguration(args)

    if args.test:
        initLoggers(config, auxLoggerNames=["test"])
    else:
        initLoggers(config, auxLoggerNames=["train", "validation", "eval_test", "test"])

    if args.test:
        runner = Runner(config, EvalMode.TEST)
        runner.test()
    else:
        runner = Runner(config, EvalMode.NONE)
        logging.getLogger("info").info("Starting training ...")
        runner.train()
        logging.getLogger("info").info("Training complete, evaluating on validation set ...")
        runner = Runner(config, EvalMode.VALIDATION, outFileName="validation_results.json")
        runner.test()
        logging.getLogger("info").info("Training complete, evaluating on validation set ...")
        config.validationDataMode = DataMode.MIXED
        runner = Runner(config, EvalMode.TEST, outFileName="test_results.json")
        runner.test()
