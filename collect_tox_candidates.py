#!/usr/bin/env python3

import json
import re

import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
    set_seed,
)

detector_model_path = "martin-ha/toxic-comment-model"
detector_target_class = "toxic"

detector_model = AutoModelForSequenceClassification.from_pretrained(
    detector_model_path
)
detector_tokenizer = AutoTokenizer.from_pretrained(
    detector_model_path
)
detector = TextClassificationPipeline(
    model=detector_model, tokenizer=detector_tokenizer
)


attempts = json.load(open("/home/leon/corpora/red_team_attempts.jsonl", "r"))
outfile = open('hhrlhf-toxic.jsonl', 'w', buffering=1)

# what do we want?
# from convos that end in toxicity,
#  a bunch of system-human turn pairs

for attempt in tqdm.tqdm(attempts):
    transcript = re.split(r"\n\n([A-Z][a-z]+): ", attempt['transcript'])
    if not transcript[0].strip():
        transcript.pop(0)
    toxic = False
    turns = []
    for i in range(0,len(transcript), 2):
        actor = transcript[i]
        utterance = transcript[i+1].replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("  ", " ").replace("\u2019", "'")

        turns.append((actor, utterance))

    try:
        detector_raw_results = detector([t[1] for t in turns if t[0] != 'Human'])
        if detector_raw_results[0]['label'] == detector_target_class:
            toxic = True
    except:
        pass

    if toxic:
        for i in range(len(turns) - 1):
            if i == 0:
                outfile.write(json.dumps({"prompt":"", "response":turns[i][1]}) + "\n")
            if turns[i][0] == "Assistant":
                outfile.write(json.dumps({"prompt":turns[i][1], "response":turns[i+1][1]}) + "\n")



