# Modules
import os
import pandas as pd
from tqdm.auto import tqdm
import shutil
import torch
from torch.utils.data import Dataset
import logging
import evaluate
import numpy as np
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainerCallback,
    TrainingArguments, 
    Trainer
)

# Configure the Dataset
class BertDataset(Dataset):
    """
    Custom PyTorch dataset for handling data required for RoBERTa model training.

    Parameters:
    - df_path (str): Path to the DataFrame pickle file.
    - tokenizer: Tokenizer for encoding text.
    - is_train (bool): Flag indicating whether the dataset is for training.

    Attributes:
    - DF_PATH (str): Path to the DataFrame pickle file.
    - TEMP_PATH (str): Temporary path for storing modified DataFrame.
    - BACKUP_PATH (str): Backup path for the original DataFrame.
    - TOKENIZER: Tokenizer for encoding text.
    - SEP_TOKEN (str): Separator token used for formatting text.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __read_dataframe(path): Reads and returns a DataFrame from the specified path.
    - __save_dataframe(df, path): Saves the DataFrame to the specified path.
    - sample(): Samples and preprocesses data for training.
    - __getitem__(idx): Retrieves and preprocesses a sample from the dataset.

    Example:
    >>> dataset = BertDataset('path/to/data.pkl', tokenizer, is_train=True)
    >>> len(dataset)
    1000
    >>> sample_data = dataset[0]
    """

    def __init__(self, df_path: str, tokenizer,  is_train: bool = True):

        self.DF_PATH = df_path
        self.TEMP_PATH = os.path.splitext(self.DF_PATH)[0] + '.temp'
        self.BACKUP_PATH = os.path.splitext(self.DF_PATH)[0] + '.backup'
        self.TOKENIZER = tokenizer
        self.SEP_TOKEN = self.TOKENIZER.sep_token

        if is_train:
            self.df = None
            self.sample()
        else:
            self.df = pd.read_pickle(self.DF_PATH)

    def __len__(self):
        return len(self.df)

    def __read_dataframe(self, path):
        return pd.read_pickle(path)

    def __save_dataframe(self, df, path):
        df.to_pickle(path)

    def sample(self):

        try:
            df = self.__read_dataframe(self.DF_PATH).copy()

            # Backup DataFrame
            self.__save_dataframe(df, self.BACKUP_PATH)

        except:
            logging.warning('File corrupted... Trying to restore from backup...')
            df = self.__read_dataframe(self.BACKUP_PATH).copy()

            # Backup DataFrame
            self.__save_dataframe(df, self.DF_PATH)


        # Get all positive samples
        df_pos = df[df['sample_type'] == 'POS']
        limit_1_3 = len(df_pos)//3

        # Get least seen Hard Negatives
        df_hard = df[df['sample_type'] == 'NEG_TOPIC'].sort_values(by = 'count').iloc[:limit_1_3]

        # Get least seen Easy Negative sample
        df_easy_1 = df[df['sample_type'] == 'NEG_POS'].sort_values(by = 'count').iloc[:limit_1_3]
        df_easy_2 = df[df['sample_type'] == 'NEG_TOPIC_POS'].sort_values(by = 'count').iloc[:limit_1_3]

        # Concatenate the datasets
        self.df = pd.concat([df_pos, df_hard, df_easy_1, df_easy_2]).sample(frac = 1).copy()

        # Update the counts
        df.loc[df['sample_id'].isin(self.df['sample_id']), 'count'] += 1

        # Save the Dataframe to a temporary path
        self.save_dataframe(df, self.TEMP_PATH)
        
        # If modifications are successful, replace the original file with the temporary file
        shutil.move(self.TEMP_PATH, self.DF_PATH)


    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        label = 0 if 'NEG' in sample['sample_type'] else 1
        text = f"Query_Title: {sample['target_title']}{self.SEP_TOKEN}Query_Topic: {sample['topic']}{self.SEP_TOKEN}Reference_Title: {sample['bib_title']}{self.SEP_TOKEN}Reference_Paragraph: {sample['citation_text_span']}"
        example = { 'label': label}
        example.update(self.TOKENIZER(text, truncation=True))
        return example


class CustomCallback(TrainerCallback):
    """
    Custom callback for the Trainer to perform actions at different stages of training.
    In this case, it calls the sample() method of the dataset at the beginning of each epoch.

    Methods:
    - on_epoch_begin(args, state, control, train_dataloader, **kwargs): Performs actions at the beginning of each epoch.
    """

    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        train_dataloader.dataset.sample()


class Train():
    """
    Class for configuring and training the RoBERTa model.

    Parameters:
    - cache_dir (str): Path to the cache directory.
    - df_train_path (str): Path to the training DataFrame pickle file.
    - df_val_path (str): Path to the validation DataFrame pickle file.
    - output_dir (str): Path to the directory to store checkpoints.

    Methods:
    - compute_metrics(eval_pred): Computes evaluation metrics.
    - __prepare_trainer(): Prepares the Trainer object for model training.
    - train(): Initiates the training process.

    Example:
    >>> trainer = Train('cache_dir', 'train_df.pkl', 'val_df.pkl', 'output_dir')
    >>> trainer.train()
    """

    def __init__(self, cache_dir, df_train_path, df_val_path, output_dir):
        
        # CONSTANTS
        self.BASE_MODEL_NAME = 'roberta-base'
        self.BATCH_SIZE = 3
        self.GRADIENT_ACCUMULATION_STEP = 80
        self.LEARNING_RATE = 5e-5
        self.EPOCHS = 60

        # Paths
        self.CACHE_DIR = os.path.normpath(cache_dir)
        self.TRAIN_DF_PATH = os.path.normpath(df_train_path)
        self.VALIDATION_DF_PATH = os.path.normpath(df_val_path)
        self.MODEL_OUTPUT_DIR = os.path.normpath(output_dir)

        # GPU Availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(123)

        # Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_NAME, cache_dir = self.CACHE_DIR)
        self.id2label = {0: "false", 1: "true"}
        self.label2id = {"false": 0, "true": 1}
        self.model = AutoModelForSequenceClassification.from_pretrained(self.BASE_MODEL_NAME, num_labels = 2, id2label = self.id2label, label2id = self.label2id, cache_dir = self.CACHE_DIR)
        
        # Load Train and Test Dataframes
        self.df_train = BertDataset(self.TRAIN_DF_PATH, tokenizer = self.tokenizer, is_train = True)
        self.df_val = BertDataset(self.VALIDATION_DF_PATH, tokenizer = self.tokenizer, is_train = False)
        self.data_collator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        self.evaluation_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        # Trainer
        self.trainer = self.__prepare_trainer()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.evaluation_metrics.compute(predictions=predictions, references=labels)


    def __prepare_trainer(self):

        train_args = TrainingArguments(
            output_dir = self.MODEL_OUTPUT_DIR,
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            logging_strategy = 'epoch',
            per_device_train_batch_size = self.BATCH_SIZE,
            per_device_eval_batch_size = self.BATCH_SIZE,
            gradient_accumulation_steps = self.GRADIENT_ACCUMULATION_STEP,
            learning_rate = self.LEARNING_RATE,
            weight_decay = 0.01,
            num_train_epochs = self.EPOCHS,
            warmup_steps = 1000,
            save_safetensors = False,
            seed = 1,
            disable_tqdm=False,
            load_best_model_at_end = False,
            dataloader_pin_memory = False,
            report_to = "tensorboard"
        )

        trainer =  Trainer(
            model = self.model,
            args=train_args,
            train_dataset = self.df_train,
            eval_dataset = self.df_val,
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = self.compute_metrics,
            callbacks=[CustomCallback]
        )

        return trainer

    def train(self) -> None:
        """
        Initiates the RoBERTa model training process.

        Returns:
        - None
        """

        try:
            self.trainer.train(resume_from_checkpoint = True)
        except ValueError:
            self.trainer.train()
        self.trainer.save_model(self.MODEL_OUTPUT_DIR)
        self.trainer.save_state()


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_df_path = os.path.join(dir_path, os.path.normpath('Datasets/df_train_classifier.pkl'))
    val_df_path = os.path.join(dir_path, os.path.normpath('Datasets/df_val_classifier.pkl'))
    output_dir = os.path.join(dir_path, os.path.normpath('FlanT5_Output'))

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_path", help="Path to Train dataset", default = train_df_path, type = str)
    parser.add_argument("-v", "--val_path", help="Path to Validation dataset", default = val_df_path, type = str)
    parser.add_argument("-c", "--cache_dir", help="Path to Cache dir", default = None, type = str)
    parser.add_argument("-o", "--output_dir", help="Path to directory to store checkpoints", default = output_dir, type = str)

    args = parser.parse_args()
    if not (os.path.isfile(args.train_path)):
        logging.error('Incorrect path to Train Dataset')

    if not (os.path.isfile(args.val_path)):
        logging.error('Incorrect path to Validation Dataset')

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t = Train(
        cache_dir = args.cache_dir, 
        df_train_path = args.train_path, 
        df_val_path = args.val_path, 
        output_dir = args.output_dir
    )

    logging.info('Initiating RoBERTa Classifier Training...')
    t.train()
    logging.info('RoBERTa Classifier Training completed')