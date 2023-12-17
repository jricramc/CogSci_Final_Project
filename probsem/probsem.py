import collections
import pathlib
import typing

import numpy as np
import pandas as pd
from tqdm import tqdm

from probsem.abstract import Object
from probsem.benchmarks import Prompt, TestSuite
from probsem.models import Model
from probsem.utils import normalize, print_sample, print_summary, sanitize_filename


class ProbSem(Object):
    def __init__(self, prompt: str, suite: str, model: str) -> None:
        super().__init__()
        self._run_id = sanitize_filename(f"{prompt}_{suite}_{model}")
        self._prompt = Prompt(prompt)
        self._suite = TestSuite(prompt, suite)
        self._model = Model(model)

    @property
    def _samples(self) -> typing.Iterable[typing.Dict[str, typing.Any]]:
        for index, samples in self._suite.samples:
            iterator = range(len(samples))
            yield {
                "prompt": [self._prompt.text for _ in iterator],
                "text": ["\n".join(samples[i].split("\n")[:-1]) for i in iterator],
                "programs": [samples[i].split("\n")[-1] for i in iterator],
                "correct": index,
            }

    def _extract_options(self, sample: dict) -> typing.Tuple[str, str]:
        """
        Extracts the two options being compared from the sample.

        Args:
        sample (dict): A sample containing the 'programs' key with comparison queries.

        Returns:
        tuple: A tuple containing the two options being compared.
        """
        # Example implementation, needs to be adjusted based on the actual format of 'programs'
        program = sample["programs"][0]  # Assuming the first program contains the options
        # Parse the program to extract the options
        # Example: "(condition (likely-to-win 'option1 'option2))"
        options = program.split("'")[1:3]  # Adjust this based on the actual format
        return tuple(options)

    def _score(self, prompt: str, text: str, program: str, options) -> np.float64:
        full_text = "\n".join([prompt, text, program])
        return self._model.score(full_text, program)

    def _export_results_table(
        self, samples: typing.List[typing.Dict[str, typing.Any]]
    ) -> None:
        fname = (
            pathlib.Path(__file__).parents[1]
            / "outputs"
            / f"{self._run_id}_results.csv"
        )
        fname.parent.mkdir(parents=True, exist_ok=True)
        table = collections.defaultdict(list)
        for sample in samples:
            table["text"].extend(sample["text"])
            table["program"].extend(sample["programs"])
            table["weights"].extend(sample["weights"])
            table["score"].extend(sample["scores"])
        pd.DataFrame(table).to_csv(fname, index=False)

    def run(self) -> None:
        samples = []
        for sample in tqdm(self._samples, total=self._suite.n_examples):
            assert len(set(sample["prompt"])) == 1
            assert len(set(sample["text"])) == 1
            assert len(set(sample["programs"])) == self._suite.n_programs
            options = self._extract_options(sample) 
            sample["weights"] = []
            for prompt, text, program in zip(sample["prompt"], sample["text"], sample["programs"]):
                score = self._score(prompt, text, program, options)
                sample["weights"].append(score)
            sample["weights"] = np.array(sample["weights"])
            sample["scores"] = normalize(sample["weights"])
            samples.append(sample)
            self.info(print_sample(sample))
        self.info(print_summary(samples))
        self._export_results_table(samples)
