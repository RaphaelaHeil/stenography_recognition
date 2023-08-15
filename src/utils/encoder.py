import string
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List


class TranscriptionEncoder(ABC):

    @abstractmethod
    def encode(self, input: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, input: List[int]) -> str:
        pass

    @abstractmethod
    def replace(self, input: str) -> str:
        """Applies the same transformation as encode but returns the resulting string representation, using the
        respective alphabet."""
        pass

    @abstractmethod
    def alphabetSize(self) -> int:
        pass

    @abstractmethod
    def getAlphabet(self, withCtcToken: bool = False) -> str:
        pass


class BaseEncoder(TranscriptionEncoder):

    def __init__(self):
        self.alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_lowercase)
        self.alphabet.sort()
        self.alphabet.insert(0, "#")  # blank
        self.alphabet.append(" ")  # space

    def encode(self, input: str) -> List[int]:
        return [self.alphabet.index(c) for c in input]

    def decode(self, input: List[int]) -> str:
        return "".join([self.alphabet[c] for c in input if c != 0])

    def replace(self, input):
        return input

    def alphabetSize(self) -> int:  # TODO: turn this into a property?
        return len(self.alphabet)

    def getAlphabet(self, withCtcToken: bool = False) -> str:
        if withCtcToken:
            return "".join(self.alphabet)
        else:
            return "".join(self.alphabet[1:])


class MelinEncoder(TranscriptionEncoder):
    WORDS = ["alldeles", "bland", "bättre", "de", "den", "det", "då", "där", "efter", "eller", "en", "ett", "genom",
        "gång", "har", "honom", "här", "ingen", "inte", "jag", "just", "kan", "kom", "kunna", "med", "med", "men",
        "mot", "mycket", "måste", "och", "om", "själv", "skulle", "som", "under", "upp", "var", "varit", "är", "även"]
    PREFIXES = ["an", "be", "fort", "fram", "hän", "in", "kon", "kun", "kän", "lång", "märk", "någo", "på", "slut",
        "särskil", "till", "tänk", "ut", "verk", "vill", "över"]
    SUFFIXES = ["ande", "de", "are", "het", "kring", "ning", "ing"]
    INFIXES = ["ng", "sj", "tj", "br", "fr", "tr", "kv", "tv", "sk", "sl", "sm", "sn", "sp", "st", "sv", "nst", "ns",
        "nk", "nkt", "av", "för"]

    def __init__(self):
        self.encDict = {0: "#"}
        alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        alphabet.extend(string.digits)
        alphabet.extend(string.ascii_lowercase)
        counter = 1
        for char in alphabet:
            self.encDict[counter] = char
            counter += 1
        for char in set(self.WORDS + self.PREFIXES + self.SUFFIXES + self.INFIXES):
            self.encDict[counter] = char
            counter += 1
        self.encDict[counter] = " "
        self.reverseDict = {v: k for k, v in self.encDict.items()}

    def __processSubword__(self, subword: str) -> List[int]:
        out = []
        for infix in self.INFIXES:
            if subword.startswith(infix):
                out.append(self.reverseDict[infix])
                subword = subword[len(infix):]
                break
        else:
            out.append(self.reverseDict[subword[0]])
            subword = subword[1:]
        if len(subword) > 0:
            out.extend(self.__processSubword__(subword))

        return out

    def __processSuffixes__(self, subword: str) -> List[int]:
        for suffix in self.SUFFIXES:
            if subword.endswith(suffix):
                subword = subword[:-len(suffix)]
                if len(subword) > 0:
                    return self.__processSubword__(subword) + [self.reverseDict[suffix]]
                else:
                    return [self.reverseDict[suffix]]
        return self.__processSubword__(subword)

    def encode(self, input: str) -> List[int]:
        out = []
        for word in input.split():
            if word in self.WORDS:
                out.append(self.reverseDict[word])
                out.append(self.reverseDict[" "])
            else:
                for prefix in self.PREFIXES:
                    if word.startswith(prefix):
                        if prefix not in ["an", "in"]:
                            out.append(self.reverseDict[prefix])
                            word = word[len(prefix):]
                        else:
                            if len(word) == 2 or word[2] not in ["a", "e", "i", "o", "u", "y", "ä", "ö", "å"]:
                                out.append(self.reverseDict[prefix])
                                word = word[len(prefix):]
                if len(word) > 0:
                    out.extend(self.__processSuffixes__(word))
                out.append(self.reverseDict[" "])
        return out[:-1]

    def decode(self, input: List[int]) -> str:
        return "".join([self.encDict[x] for x in input if x != 0])

    def replace(self, input: str) -> str:
        return input

    def alphabetSize(self) -> int:
        return len(self.reverseDict)

    def getAlphabet(self, withCtcToken: bool = False) -> str:
        alphabet = [k for k in self.reverseDict.keys()]
        if withCtcToken:
            return alphabet
        else:
            alphabet.remove("#")
            return alphabet


class NgramEncoder(TranscriptionEncoder):
    NGRAMS = ["nsion", "nskt", "nsj", "nst", "nsk", "nkt", "skt", "ng", "sj", "tj", "br", "fr", "tr", "kv", "tv", "sk",
        "sl", "sm", "sn", "sp", "st", "sv", "nt", "nd", "ns", "nj", "nk", "bt", "pt", "kt", "xt"]
    NGRAM_ENCODING = {}

    def __init__(self):
        self.alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_lowercase)
        self.alphabet.extend(self.NGRAMS)
        self.alphabet.sort()
        self.alphabet.insert(0, "#")  # blank
        self.alphabet.append(" ")  # space
        self.NGRAM_ENCODING = {ngram: [self.alphabet.index(c) for c in ngram] for ngram in self.NGRAMS}

    def encode(self, input: str) -> List[int]:
        result = [self.alphabet.index(c) for c in input]

        for ngram in self.NGRAMS:
            ngramEncoding = self.NGRAM_ENCODING[ngram]
            mylist = result
            pattern = ngramEncoding
            matches = []
            for i in range(len(mylist)):
                if mylist[i] == pattern[0] and mylist[i:i + len(pattern)] == pattern:
                    matches.append((i, i + len(pattern)))
            for match in matches[::-1]:  # go backwards, in case there is more than one match
                mylist[match[0]:match[1]] = [self.alphabet.index(ngram)]
        return result

    def decode(self, input: List[int]) -> str:
        return "".join([self.alphabet[c] for c in input if c != 0])

    def alphabetSize(self) -> int:
        return len(self.alphabet)

    def replace(self, input: str) -> str:
        return input

    def getAlphabet(self, withCtcToken: bool = False) -> str:
        if withCtcToken:
            return "".join(self.alphabet)
        else:
            return "".join(self.alphabet[1:])


class CharacterShortformEncoder(TranscriptionEncoder):
    SHORTFORMS = {"av": "a", "bar": "b", "de": "d", "en": "e", "från": "f", "har": "h", "jag": "j", "kan": "k",
        "men": "m", "och": "o", "ut": "u", "var": "v", "är": "ä", "över": "ö"}
    REVERSE_SHORTFORMS = {}

    def __init__(self):
        self.alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_lowercase)
        self.alphabet.sort()
        self.alphabet.insert(0, "#")  # blank
        self.alphabet.append(" ")  # space
        self.REVERSE_SHORTFORMS = {v: k for (k, v) in self.SHORTFORMS.items()}
        # replace all character-based shortforms, e.g. och -> o, men -> m, etc.

    def encode(self, input: str) -> List[int]:
        result = []
        s = input.split(" ")  # TODO: refactor this to use groupby?!
        for word in s:
            if word in self.SHORTFORMS.keys():
                result.append(self.alphabet.index(self.SHORTFORMS[word]))
            else:
                result.extend([self.alphabet.index(c) for c in word])
            result.append(self.alphabet.index(" "))

        return result[:-1]  # drop the last blank

    def decode(self, input: List[int]) -> str:
        grouped = [list(g) for k, g in groupby(input, lambda s: s == self.alphabet.index(self.alphabet[-1]))]
        result = ""
        for word in grouped:
            tmp = "".join([self.alphabet[c] for c in word if c != 0])
            if tmp in self.REVERSE_SHORTFORMS:
                result += self.REVERSE_SHORTFORMS[tmp]
            else:
                result += tmp
        return result

    def alphabetSize(self) -> int:
        return len(self.alphabet)

    def replace(self, input: str) -> str:
        result = []
        for word in input.split(" "):
            if word in self.SHORTFORMS.keys():
                result.append(self.SHORTFORMS[word])
            else:
                result.append(word)
        return " ".join(result)

    def getAlphabet(self, withCtcToken: bool = False) -> str:
        if withCtcToken:
            return "".join(self.alphabet)
        else:
            return "".join(self.alphabet[1:])


class SuffixEncoder(TranscriptionEncoder):
    SUFFIXES = ["en", "et", "are", "ing"]
    SUFFIX_ENCODING = {}

    def __init__(self):
        self.alphabet = list("()\"äåö,!?.:'~")  # lower case, digits, special chars: äåö,!"'
        self.alphabet.extend(string.digits)
        self.alphabet.extend(string.ascii_lowercase)
        self.alphabet.extend(self.SUFFIXES)
        self.alphabet.sort()
        self.alphabet.insert(0, "#")  # blank
        self.alphabet.append(" ")  # space
        self.SUFFIX_ENCODING = {ngram: [self.alphabet.index(c) for c in ngram] for ngram in self.SUFFIXES}

    def encode(self, input: str) -> List[int]:
        result = []
        words = input.split(" ")
        for idx, word in enumerate(words):
            wordEncoding = []
            for suffix in self.SUFFIXES:
                # this only works as long as the suffixes do not overlap, e.g. ing and ng would not work properly here
                if word.endswith(suffix):
                    wordEncoding = [self.alphabet.index(c) for c in word[:-len(suffix)]]
                    wordEncoding.append(self.alphabet.index(suffix))
                    continue
            if not wordEncoding:
                wordEncoding = [self.alphabet.index(c) for c in word]

            if idx < len(words) - 1:
                wordEncoding.append(self.alphabet.index(" "))
            result.extend(wordEncoding)

        return result

    def decode(self, input: List[int]) -> str:
        return "".join([self.alphabet[c] for c in input if c != 0])

    def alphabetSize(self) -> int:
        return len(self.alphabet)

    def replace(self, input: str) -> str:
        return input

    def getAlphabet(self, withCtcToken: bool = False) -> str:
        if withCtcToken:
            return "".join(self.alphabet)
        else:
            return "".join(self.alphabet[1:])
