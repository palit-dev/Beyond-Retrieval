# Modules
import os
import pandas as pd
from tqdm.auto import tqdm
import shutil
import torch
from torch.utils.data import Dataset
import logging
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

# Configure the Dataset
class FlanT5Dataset(Dataset):
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

    def __init__(self, df_path: str, tokenizer, is_train: bool = True):
        self.DF_PATH = df_path
        self.TEMP_PATH = os.path.splitext(self.DF_PATH)[0] + '.temp'
        self.BACKUP_PATH = os.path.splitext(self.DF_PATH)[0] + '.backup'
        self.TOKENIZER = tokenizer

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
        self.__save_dataframe(df, self.TEMP_PATH)
        
        # If modifications are successful, replace the original file with the temporary file
        shutil.move(self.TEMP_PATH, self.DF_PATH)


    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        label = 'false' if 'NEG' in sample['sample_type'] else 'true'
        text = f"Query_Title: {sample['target_title']} Query_Topic: {sample['topic']} Reference_Title: {sample['bib_title']} Reference_Paragraph: {sample['citation_text_span']} Relevant:"
        tokenized = self.TOKENIZER(text, truncation=True, padding=True)
        tokenized['labels'] = self.TOKENIZER(label)["input_ids"]
        return tokenized


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
    Class for configuring and training the Flan T5 model.

    Parameters:
    - cache_dir (str): Path to the cache directory.
    - df_train_path (str): Path to the training DataFrame pickle file.
    - df_val_path (str): Path to the validation DataFrame pickle file.
    - output_dir (str): Path to the directory to store checkpoints.

    Methods:
    - __prepare_trainer(): Prepares the Trainer object for model training.
    - train(): Initiates the training process.

    Example:
    >>> trainer = Train('cache_dir', 'train_df.pkl', 'val_df.pkl', 'output_dir')
    >>> trainer.train()
    """

    def __init__(self, cache_dir, df_train_path, df_val_path, output_dir):
        
        # CONSTANTS
        self.BASE_MODEL_NAME = 'google/flan-t5-base'
        self.BATCH_SIZE = 3
        self.GRADIENT_ACCUMULATION_STEP = 50
        self.LEARNING_RATE = 3e-4
        self.EPOCHS = 30

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.BASE_MODEL_NAME, cache_dir = self.CACHE_DIR).to(self.device)
        
        # Load Train and Test Dataframes
        self.df_train = FlanT5Dataset(self.TRAIN_DF_PATH, tokenizer = self.tokenizer, is_train = True)
        self.df_val = FlanT5Dataset(self.VALIDATION_DF_PATH, tokenizer = self.tokenizer, is_train = False)

        # Trainer
        self.trainer = self.__prepare_trainer()


    def __prepare_trainer(self):

        train_args = Seq2SeqTrainingArguments(
            output_dir = self.MODEL_OUTPUT_DIR,
            evaluation_strategy = 'epoch',
            logging_strategy = 'epoch',
            save_strategy = 'epoch',
            prediction_loss_only = True,
            per_device_train_batch_size = self.BATCH_SIZE,
            per_device_eval_batch_size = self.BATCH_SIZE,
            gradient_accumulation_steps = self.GRADIENT_ACCUMULATION_STEP,
            learning_rate = self.LEARNING_RATE,
            weight_decay = 5e-5,
            num_train_epochs = self.EPOCHS,
            warmup_steps = 1000,
            seed = 1,
            disable_tqdm=False,
            save_safetensors = False,
            load_best_model_at_end = False,
            dataloader_pin_memory = False,
            report_to = "tensorboard"
        )

        trainer =  Seq2SeqTrainer(
            model = self.model,
            args = train_args,
            train_dataset = self.df_train,
            eval_dataset = self.df_val,
            tokenizer = self.tokenizer,
            callbacks=[CustomCallback]
        )

        return trainer

    def train(self):
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

    logging.info('Initiating Flan T5 Classifier Training...')
    t.train()
    logging.info('Flan T5 Classifier Training completed')