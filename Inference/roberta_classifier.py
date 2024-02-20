# Modules
import os
import pandas as pd
from tqdm import tqdm
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

class BertDataset(Dataset):
    """
    Custom PyTorch dataset for handling data required for RoBERTa model testing.

    Parameters:
    - df_path (str): Path to the DataFrame pickle file.
    - tokenizer: Tokenizer for encoding text.

    Attributes:
    - DF_PATH (str): Path to the DataFrame pickle file.
    - TOKENIZER: Tokenizer for encoding text.
    - df: Pandas DataFrame containing the data.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Retrieves and preprocesses a sample from the dataset.

    Example:
    >>> dataset = FlanT5Dataset('path/to/data.pkl', tokenizer)
    >>> len(dataset)
    1000
    >>> sample_data = dataset[0]
    """


    def __init__(self, df_path: str, tokenizer):
        self.DF_PATH = df_path
        self.TOKENIZER = tokenizer
        self.SEP_TOKEN = self.TOKENIZER.sep_token
        self.df = pd.read_pickle(self.DF_PATH)

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        label = 0 if 'NEG' in sample['sample_type'] else 1
        text = f"Query_Title: {sample['target_title']}{self.SEP_TOKEN}Query_Topic: {sample['topic']}{self.SEP_TOKEN}Reference_Title: {sample['bib_title']}{self.SEP_TOKEN}Reference_Paragraph: {sample['citation_text_span']}"
        example = {'sample_id' : sample['sample_id'], 'label': label, 'text': text}
        return example


class Test():
    """
    Class for testing the RoBERTa model.

    Parameters:
    - model_dir (str): Path to the directory containing the model checkpoint.
    - test_df_path (str): Path to the testing DataFrame pickle file.
    - output_path (str): Path to the directory to store inferred files.

    Methods:
    - test(): Initiates the testing process.

    Example:
    >>> tester = Test('model_checkpoint', 'test_data.pkl', 'output_directory')
    >>> tester.test()
    """

    def __init__(self, model_dir: str, test_df_path: str, output_path: str) -> None:
        
        # CONSTANTS
        self.MODEL_CHECKPOINT_DIR = model_dir
        self.BATCH_SIZE = 5

        # Paths
        self.TEST_DF_PATH = os.path.normpath(test_df_path)
        self.RESULT_PATH = os.path.normpath(output_path)
        
        # GPU Availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and Tokenizer
        self.id2label = {0: "false", 1: "true"}
        self.label2id = {"false": 0, "true": 1}
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_CHECKPOINT_DIR, 
            num_labels = 2, 
            id2label = self.id2label, 
            label2id = self.label2id
        )
        
        
        # Load Train and Test Dataframes
        self.df_test = BertDataset(self.TEST_DF_PATH, tokenizer = self.tokenizer)
        self.loader_test = DataLoader(self.df_test, batch_size = self.BATCH_SIZE)


    def test(self):
        
        if os.path.isfile(self.RESULT_PATH.format(os.path.basename(self.MODEL_CHECKPOINT_DIR))):
            return
        
        self.model.to(self.device)
        self.model.eval()
        
        predictions = {
            'sample_id': [],
            'true_output': [],
            'predicted_output': []
        }


        for batch in tqdm(self.loader_test, total = len(self.loader_test), desc = 'Samples'):
            predictions['sample_id'].extend(batch['sample_id'])
            predictions['true_output'].extend(batch['label'].tolist())
            encodings = self.tokenizer(batch['text'], truncation = True, padding = 'max_length', return_tensors="pt")
       
            encodings = {k: v.to(self.device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = self.model(**encodings)
            logits = outputs.logits 
            predictions['predicted_output'].extend(logits.argmax(-1).tolist())
        
        df_result = pd.DataFrame.from_dict(predictions)
        df_result.to_pickle(self.RESULT_PATH)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_df_path = os.path.join(dir_path, os.path.normpath('Datasets/df_test.pkl'))
    output_dir = os.path.join(dir_path, os.path.normpath('FlanT5_Output'))

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_path", help="Path to Test dataset", default = train_df_path, type = str)
    parser.add_argument("-o", "--output_dir", help="Path to directory to store inferred files", default = output_dir, type = str)
    parser.add_argument("-m", "--model_dir", help="Path to Model directory", default = None, type = str)

    args = parser.parse_args()
    if not (os.path.isfile(args.test_path)):
        logging.error('Incorrect path to Test Dataset')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t = Test(
        model_dir = args.model_dir, 
        test_df_path = args.test_path, 
        output_path = args.output_dir
    )

    logging.info('Initiating RoBERTa Classifier Testing....')
    t.test()
    logging.info('RoBERTa Classifier Testing completed')