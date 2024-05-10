import json
import os
import time
from argparse import ArgumentParser
from typing import Dict, Optional, List

import pandas as pd
from dotenv import load_dotenv
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm

load_dotenv()

"""
NOTE: This class was taken from Schick et al. 2021, and the original can be found at https://github.com/timoschick/self-debiasing

This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.

# Update - We modified also from https://github.com/Robert-Morabito/Instructive-Debiasing Morabito et al. 2023.
"""


class PerspectiveApiScorer:
    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    def __init__(self, api_key: str):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key,
                                        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                                        static_discovery=False)

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
        }

        response = None
        while not response:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
            except HttpError as e:
                print(f'Perspective API threw an error: {e}\n Retrying in 5 seconds...')
                time.sleep(5)

        return {attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value'] for attribute in
                requested_attributes}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    dataset = pd.read_excel(input_file)

    outputs = []

    scorer = PerspectiveApiScorer(api_key=os.getenv('PERSPECTIVE_API'))

    for row in tqdm(dataset.to_dict(orient='records'), total=len(dataset)):
        initial_scores = scorer.get_scores(row["biased_text"])
        post_scores = scorer.get_scores(row["unbiased_text"])
        row["initial_scores"] = initial_scores
        row["post_scores"] = post_scores

        outputs.append(row)

    with open(output_file, "w") as out_file:
        for row in outputs:
            json.dump(row, out_file)
            out_file.write('\n')
