# Datasets

Welcome to the repository's datasets section. Here, you'll find a comprehensive collection of datasets proposed in our paper and utilized for the finetuning of the proposed supervised models.


## Target Proposals

This [dataset](df_target_proposals.pkl) encompasses essential metadata for target papers/proposals:

* **arxiv_id** : ArXiv ID of the Target Paper/Proposal.
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

This section compiles features common across train, validation, and test splits:

* **sample_id** : Unique identifier for each sample.
* **target_arxiv_id** : ArXiv ID of the Target Paper/Proposal.
* **target_title** : Title of the Target Paper/Proposal.
* **topic** : Topic that cites the Reference Paper.
* **bib_id** : Unique Identifier for Reference Papers.
* **bib_title** : Title of Reference Papers.
* **citation_text** : Sentence (Citation Text) used to cite the Reference Paper in the Target Paper/Proposal.
* **citation_text_span_id** : Unique identifier for the retreived Citation Text Span.



### Train Classifier
This [dataset](df_train_classifier.pkl) consists training samples for the Reference Paper Topic Classifier Models.

* **count** : Number of times the sample has been seen during training (internal use only during traning).

* **citation_text_span** : Citation Text Span retrieved from the Reference Paper using Mono-T5 retriever with citation text as the query.

### Train Retriever
This [dataset](df_train_retriever.pkl) consists training samples for the Topic-based Citation Text Span Retrieval Model.

* **count** : Number of times the sample has been seen during training the model (internal use only during traning).

* **paragraph** : The top 4 paragraphs retrieved from a reference paper for a given research proposal title and topic using Mono-T5 retriever with citation text as the query.

### Validation Classifier
This [dataset](df_validation_classifier.pkl) includes samples for validating Reference Paper Topic Classifier Models during training:

* **count** : Number of times the sample has been seen during training (internal use only during traning).

* **citation_text_span** : Citation Text Span retrieved from the Reference Paper using Mono-T5 retriever with citation text as the query.

### Validation Retriever
This [dataset](df_validation_retriever.pkl) comprises samples for validating the Topic-based Citation Text Span Retrieval Model during training:

* **count** : Number of times the sample has been seen during training the model (internal use only during traning).

* **paragraph** : The top 4 paragraphs retrieved from a reference paper for a given research proposal title and topic using Mono-T5 retriever with citation text as the query.

### Test
This [dataset](df_test.pkl) includes samples for testing both the Topic-based Citation Text Span Retrieval and Reference Paper Topic Classifier Models:

* **citation_text_span** : Citation Text Span retrieved from the Reference Paper using Mono-T5 retriever with citation text as the query.


## Additional Datasets

Visit [here](https://zenodo.org/records/10682636?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM1NTRhYzg2LTBkNTEtNDgwOS1hZjY4LTI1MDdjZGZjYzMwMSIsImRhdGEiOnt9LCJyYW5kb20iOiI2Mjc3YjJkN2RlMjRhYzQ2YTk3YWQwMDFhYjcwMWEzNSJ9.o7po_aBLpmtgVnYMgKn2t5MfEVPopo2hq4LvfoJAbgXG9vBw58vrRx_bIEpeFBONXP4eOaqBxiQZ96MOtF-9hQ) to access datasets used for training the proposed models, consisting of negative samples hosted on Zenodo.

The train, validation, and test datasets contain an additional column, `sample_type`, denoting one of the following types:

* **POS:** The top-k chunks retrieved from $p_i$, using citation text $ct_{ij}$ of topic $t_j$, serve as positive samples for the topic $t_j$ of proposal $R$.

* **NEG_POS:** (type 1) The bottom-k chunks retrieved from $p_i$, using citation text $ct_{ij}$ of topic $t_j$, serve as easy negatives for the topic $t_j$ of proposal $R$.

* **NEG_TOPIC_POS:** (type 2) Citation text span (top-k chunks) retrieved from reference paper $p_i$, \textbf{NOT} cited in topic $t_j$,  but cited in $t_k$ where $k \neq j$, using the citation text $ct_{ik}$. These serve as easy negatives for the topic $t_j$ of proposal $R$. 

* **NEG_TOPIC:** (type 3) We take  citation texts $ct_{ij}$ for  papers cited in the topic $t_j$. With each $ct_{ij}$  as the query, we retrieve top-k chunks  from reference papers $p_i$, \textbf{NOT} cited in topic $t_j$,  but cited in $t_k$ where $k \neq j$. The top-k chunks demonstrating maximum similarity with one of the $ct_{ij}$  serve as hard negatives for the topic $t_j$ of proposal $R$. 

### Baselines

Explore various baselines and evaluation results on the dataset:

* **df_baselines_independent:** Baselines established using Human, LLM and supervised approaches for the independent evaluation of Reference Paper Topic Classifier (TC) using citation text span retreived using citation text-based Retriever.

* **df_baselines_pipeline:** Baselines established using Human, LLM and supervised approaches for the evaluation of Reference Paper Topic Classifier (TC) in the pipeline using citation text span retreived using topic-based Retriever.

* **df_test_flanT5:** Evaluation result of Flan-T5 as TC on the entire test split of the dataset.

* **df_test_roberta:** Evaluation result of RoBERTa as TC on the entire test split of the dataset.

* **df_test_t5:** Evaluation result of T5 as Topic-based Retriever on the test split of the dataset.



