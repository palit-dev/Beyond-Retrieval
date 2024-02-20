# Datasets

Contain Datasets.

## Description of Datasets

### Target Proposals

This [dataset](df_target_proposals.pkl) consists of the following metadata of the target papers/proposals that exist in the dataset.

* **arxiv_id** : The ArXiv ID of the Target Paper/Proposal.
* **title** : Title of the Target Paper/Proposal.
* **abstract** : Abstract of the Target Paper/Proposal.
* **publish_date** : Publishing date of the Target Paper/Proposal.
* **categories** : Category of the Target Paper/Proposal.   
    * **cs.CV** : Computer Vision
    * **cs.CL** : Computational Linguistics
    * **cs.AI** : Artificial Intelligence
    * **cs.LG** : Machine Learning
    * **cs.CV cs.CL** : Computer Vision and Computational Linguistics

## Train, Test and Validation Datasets

This section consists of all the features common across the train, validation and test split:

* **sample_id** : Unique identifier for each sample.
* **target_arxiv_id** : The ArXiv ID of the Target Paper/Proposal.
* **target_title** : Title of the Target Paper/Proposal.
* **topic** : Topic that cites the Reference Paper.
* **bib_id** : Unique Identifier for Reference Papers.
* **bib_title** : Title of Reference Papers.
* **citation_text** : Sentence (Citation Text) used to cite the Reference Paper in the Target Paper/Proposal.
* **citation_text_span_id** : Unique identifier for the retreived Citation Text Span.



### Train Classifier
This [dataset](df_train_classifier.pkl) consists of the training samples to train the Reference Paper Topic Classifier Models.

* **count** : Number of times the sample has been seen during training the model. Used internally only during training.
* **citation_text_span** : The Citation Text Span retrieved using citation text from the Reference Paper.

### Train Retriever
This [dataset](df_train_retriever.pkl) consists of the training samples to train the Topic-based Citation Text Span Retrieval Model.

* **count** : Number of times the sample has been seen during training the model. Used internally only during training.
* **paragraph** : The top 4 paragraphs retrieved from a reference paper for a given research proposal title and topic.

### Validation Classifier
This [dataset](df_validation_classifier.pkl) consists of samples to validate the Reference Paper Topic Classifier Models during training.

* **count** : Number of times the sample has been seen during training the model. Used internally only during training.
* **citation_text_span** : The Citation Text Span retrieved using citation text from the Reference Paper.

### Validation Retriever
This [dataset](df_validation_retriever.pkl) consists of samples to validate the Topic-based Citation Text Span Retrieval Model during training.

* **count** : Number of times the sample has been seen during training the model. Used internally only during training.
* **paragraph** : The top 4 paragraphs retrieved from a reference paper for a given research proposal title and topic.

### Test
This [dataset](df_test.pkl) consists of samples to test both the Topic-based Citation Text Span Retrieval and Reference Paper Topic Classifier Models.

* **citation_text_span** : The Citation Text Span retrieved using citation text from the Reference Paper.




https://zenodo.org/records/10682636?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM1NTRhYzg2LTBkNTEtNDgwOS1hZjY4LTI1MDdjZGZjYzMwMSIsImRhdGEiOnt9LCJyYW5kb20iOiI2Mjc3YjJkN2RlMjRhYzQ2YTk3YWQwMDFhYjcwMWEzNSJ9.o7po_aBLpmtgVnYMgKn2t5MfEVPopo2hq4LvfoJAbgXG9vBw58vrRx_bIEpeFBONXP4eOaqBxiQZ96MOtF-9hQ