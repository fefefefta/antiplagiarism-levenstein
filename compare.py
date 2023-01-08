import argparse
import ast
import dataclasses
import re
from abc import ABC, abstractmethod

import numpy as np


@dataclasses.dataclass
class ComparisonPathsPair:
    original: str
    replica: str


class CLI:
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file')
    parser.add_argument('scores_file')

    @classmethod
    def get_cli_args(cls) -> (str, str):
        args = cls.parser.parse_args()

        return args.input_file, args.scores_file


class FileManager:
    def __init__(self):
        self.input_path, self.scores_path = CLI().get_cli_args()

    def get_comparison_pairs(self) -> list[ComparisonPathsPair]:
        comparison_pairs = []

        with open(self.input_path, 'r') as input_file:
            for line in input_file:
                original, replica = line.split()
                comparison_pairs.append(
                    ComparisonPathsPair(original=original, replica=replica)
                )
        return comparison_pairs

    def write_scores(self, scores: list[float]) -> None:
        with open(self.scores_path, 'w') as scores_file:
            for score in scores:
                scores_file.write(str(score) + '\n')

    @staticmethod
    def read_file(path):
        with open(path, 'r') as file:
            file_data = file.read()
        return file_data.strip()


class Comparison:
    def __init__(self, comparison_pairs: list[ComparisonPathsPair]):
        self.path_pairs = comparison_pairs
        self.scores = []
        self.calc_mechanism: AntiplagiarismMechanism = LevensteinDistance()

    def execute(self) -> list[float]:
        for path_pair in self.path_pairs:
            result = self._compare(
                path_pair.original,
                path_pair.replica,
            )

            self.scores.append(result)
        return self.scores

    def _compare(self, original_path: str, replica_path: str) -> float:
        original = FileManager.read_file(original_path)
        replica = FileManager.read_file(replica_path)

        normalized_original = self._normalize(original)
        normalized_replica = self._normalize(replica)

        score = self.calc_mechanism.calc_duplication_level(
            normalized_original, normalized_replica)

        return score

    def _normalize(self, file_data: str) -> str:
        file_data = re.sub(r'"""[\s\S]*?"""', '', file_data)
        file_data = re.sub(r"'''[\s\S]*?'''", '', file_data)
        file_data = re.sub(r'#.*', '', file_data)

        tree = ast.unparse(ast.parse(file_data))
        return tree


class AntiplagiarismMechanism(ABC):
    """Интерфейс для удобства расширения списка способов подсчета"""
    @abstractmethod
    def calc_duplication_level(self, original, replica):
        pass


class LevensteinDistance(AntiplagiarismMechanism):

    def calc_duplication_level(self, original: str, replica: str):
        self.original = original
        self.replica = replica

        levenstein_distance = self.calc_levenstein_distance()

        levenstein_coefficient = (
            levenstein_distance / max(len(original), len(replica)))

        duplication_level = round(1 - levenstein_coefficient, 2)

        return duplication_level

    def calc_levenstein_distance(self) -> np.int64:
        dp = np.zeros(
            (len(self.original) + 1,
             len(self.replica) + 1), int)

        for i in range(len(self.original) + 1):
            for j in range(len(self.replica) + 1):
                if i == 0 or j == 0:
                    dp[i][j] = max(i, j)
                else:
                    dp[i][j] = min(dp[i][j - 1], dp[i - 1][j]) + 1
                    if self.original[i - 1] == self.replica[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                    else:
                        dp[i][j] = min(dp[i][j], dp[i - 1][j - 1] + 1)
        return dp[-1][-1]


def main() -> None:
    io = FileManager()
    comparison_pairs = io.get_comparison_pairs()
    scores = Comparison(comparison_pairs).execute()
    io.write_scores(scores)


if __name__ == '__main__':
    main()