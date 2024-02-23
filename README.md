# Beyond Retrieval: Topic-based Alignment of Scientific Papers to Research Proposal

Welcome to the official Dataset Repository for our work on the topic-based alignment of scientific papers to research proposals. This repository houses the datasets and resources essential for understanding and reproducing the experiments conducted in our study.

## Abstract
Existing approaches to automated literature review generation either summarize or generate citation text for individual scientific articles relevant to the target manuscript independently without considering their relationship to other relevant articles. Alternatively, some approaches generate one monolithic review for all the relevant scientific articles, adversely affecting readability. This work is a precursor to the generation of a comprehensive literature review for a research proposal that is well organized into a set of topics. We focus on the task of mapping relevant scientific articles to one or more topics in a proposal. We assume the availability of (1) a corpus of relevant scientific articles retrieved using the provided title and abstract of the proposal and (2) a catalog, which is a set of high-level topics relevant to the proposal, to which the scientific articles in the corpus need to be aligned. Unlike existing scientific literature retrieval approaches, we do not assume the availability of a detailed topic description and/or a well-defined citation text or paragraph for the relevant articles. This assumption is unrealistic during the initial research proposal writing stage, where the literature review containing the detailed topic description and/or citation text or paragraph is yet to be created. Due to the unavailability of existing datasets, we synthesize a new dataset for the proposed task and establish human and zero-shot Large Language Model (LLM)-based baselines along with our fine-tuning approaches. A higher accuracy of the human baseline demonstrates the feasibility of the task. In contrast, a significant gap (25.82%) in the performance of human versus LLM-based and fine-tuning approaches indicates the inherent challenges of the task.

## tl;dr
This work defines a novel task of topic-based alignment of scientific papers to a research proposal, a precursor to literature review generation. It introduces a new dataset for the task and establishes baselines by human, LLM and fine-tuned models.


## Datasets

* [df_target_proposals](Datasets) : Metadata about the Target Proposals/Papers used.

* [df_train_classifier](Datasets) : Training split for fine-tuning Flan-T5 and RoBERTa models as Reference Paper Topic Classifier (TC).

* [df_val_classifier](Datasets) : Validation split for evaluating models during Reference Paper TC training.

* [df_train_retriever](Datasets) : Training split for fine-tuning T5 model for Topic-based Citation Text Span Retrieval (TR).

* [df_val_retriever](Datasets) : Validation split for evaluating models during TR training.

* [df_test](Datasets) : Test split for evaluating models.

For more details, read [Datasets](Datasets)

## Train

Explore examples for training models:

### Topic-based Citation Text Span Retreiver

T5

```bash

python3 t5_retriever.py -t <path/to/train/dataset> -v <path/to/validation/dataset> -c <path/to/cache/directory> -o <directory/to/save/model/checkpints>

```


### Reference Paper Topic Classifier

Flan-T5

```bash

python3 flanT5_classifier.py -t <path/to/train/dataset> -v <path/to/validation/dataset> -c <path/to/cache/directory> -o <directory/to/save/model/checkpints>

```

RoBERTa

```bash

python3 roberta_classifier.py -t <path/to/train/dataset> -v <path/to/validation/dataset> -c <path/to/cache/directory> -o <directory/to/save/model/checkpints>

```

Visit [Train](Train) for more details.


## Infer

Examples for model inference:

### Topic-based Citation Text Span Retreiver

T5

```bash

python3 t5_retriever.py -t <path/to/test/dataset> -o <directory/to/ranked/paragraphs> -m <path/to/model/directory> -p <path/to/directory/storing/paragraphs/of/reference/paper> 
```


### Reference Paper Topic Classifier

Flan-T5

```bash

python3 flanT5_classifier.py -t <path/to/test/dataset> -o <directory/to/ranked/paragraphs> -m <path/to/model/directory>

```

RoBERTa

```bash

python3 roberta_classifier.py -t <path/to/test/dataset> -o <directory/to/ranked/paragraphs> -m <path/to/model/directory>

```

Visit [Inference](Inference) for more details.

### Evaluation

The directory contains scripts to evaluate model perfomance of the Reference Paper Topic Classifier as an independent or as part of the pipeline. For more details, visit [Evaluation](Evaluation).

### PDF Extractor

The [script](pdf_extractor.py) used to extract the content from the Reference Paper PDF Files.


