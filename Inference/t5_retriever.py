from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

class T5Dataset(Dataset):
    """
    Custom PyTorch dataset for handling data required for T5 model testing.

    Parameters:
    - df_path (str): Path to the DataFrame pickle file.
    - topic (str): Topic information.
    - bib_title (str): Bibliography title.

    Attributes:
    - DF_PATH (str): Path to the DataFrame pickle file.
    - TOPIC (str): Topic information.
    - BIB_TITLE (str): Bibliography title.
    - df: Pandas DataFrame containing the data.

    Methods:
    - __len__(): Returns the length of the dataset.
    - __getitem__(idx): Retrieves and preprocesses a sample from the dataset.

    Example:
    >>> dataset = T5Dataset('path/to/data.pkl', 'Machine Learning', 'Introduction')
    >>> len(dataset)
    1000
    >>> sample_data = dataset[0]
    """

    def __init__(self, df_path: str, topic: str, bib_title: str):
        self.DF_PATH = df_path
        self.TOPIC = topic
        self.BIB_TITLE = bib_title
        self.df = pd.read_pickle(self.DF_PATH)
        
    
    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        text = f"Topic: {self.TOPIC} Title: {self.BIB_TITLE} Paragraph: {sample['paragraph']} Relevant:"
        example = {'pid' : sample['pid'], 'text': text}
        return example


class Test():
    """
    Class for testing the T5 Retriever model.

    Parameters:
    - model_dir (str): Path to the directory containing the model checkpoint.
    - test_df_path (str): Path to the testing DataFrame pickle file.
    - output_dir (str): Path to the directory to store inferred files.
    - paragraph_dir (str): Path to the directory containing paragraph data.

    Methods:
    - test(): Initiates the testing process.

    Example:
    >>> tester = Test('model_checkpoint', 'test_data.pkl', 'output_directory', 'paragraph_data_directory')
    >>> tester.test()
    """

    def __init__(self, model_dir: str, test_df_path: str, output_dir: str, paragraph_dir: str):
        
        # CONSTANTS
        self.MODEL_CHECKPOINT_DIR = model_dir
        self.BATCH_SIZE = 10

        # Paths
        self.TEST_DF_PATH = os.path.normpath(test_df_path)
        self.RESULT_DIR = os.path.normpath(output_dir)
        self.PARAGRAPH_DIR = os.path.normpath(paragraph_dir)
        
        Path(self.RESULT_DIR).mkdir(parents=True, exist_ok=True) 
        
        # GPU Availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_CHECKPOINT_DIR)
        self.token_false_id = self.tokenizer.get_vocab()['▁false']
        self.token_true_id  = self.tokenizer.get_vocab()['▁true']
        
        
        # Load Train and Test Dataframes
        self.df_test = pd.read_pickle(self.TEST_DF_PATH)
        self.df_test = self.df_test[self.df_test['sample_type'].isin(['POS', 'NEG_TOPIC'])]
        self.df_test.reset_index(inplace = True, drop = True)


    def test(self):
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            for _, item in self.df_test.iterrows():

                result_path = os.path.join(self.RESULT_DIR, f"{item['sample_id']}.pkl")

                if os.path.isfile(result_path):
                    continue
                    
                loader_score = DataLoader(
                    T5Dataset(
                        df_path = os.path.join(self.PARAGRAPH_DIR, f"{item['citation_text_span_id']}.pkl"),
                        topic = item['topic'],
                        bib_title = item['bib_title']
                    ),
                    batch_size = self.BATCH_SIZE
                )
                
                predictions = {'pid': [], 'score': []}
        
                for batch in loader_score:
                    predictions['pid'].extend(batch['pid'])
                    encodings = self.tokenizer(batch['text'], truncation = True, padding = 'max_length', return_tensors="pt")
        
                    input_ids = encodings['input_ids'].to(self.device)
                    attention_mask = encodings['attention_mask'].to(self.device)
                    decode_ids = torch.full((input_ids.size(0), 1), self.model.config.decoder_start_token_id, dtype=torch.long).to(self.device)
                
                    encoder_outputs = self.model.get_encoder()(input_ids, attention_mask = attention_mask)
                    model_inputs = self.model.prepare_inputs_for_generation(decode_ids, encoder_outputs = encoder_outputs, past = None, attention_mask =attention_mask, use_cache=True)
                    outputs = self.model(**model_inputs)
                    
                    next_token_logits = outputs[0][:, -1, :]
                    batch_scores = next_token_logits[:, [self.token_false_id, self.token_true_id]]
                    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim = 1)
                    predictions['score'].extend(batch_scores[:, 1].tolist())
                
                df_result = pd.DataFrame.from_dict(predictions)
                df_result.to_pickle(result_path)


if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_df_path = os.path.join(dir_path, os.path.normpath('Datasets/df_test.pkl'))
    output_dir = os.path.join(dir_path, os.path.normpath('FlanT5_Output'))

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_path", help="Path to Test dataset", default = train_df_path, type = str)
    parser.add_argument("-o", "--output_dir", help="Path to directory to store inferred files", default = output_dir, type = str)
    parser.add_argument("-m", "--model_dir", help="Path to Model directory", default = None, type = str)
    parser.add_argument("-p", "--par_dir", help="Path to Paragraph directory", default = None, type = str)

    args = parser.parse_args()
    if not (os.path.isfile(args.test_path)):
        logging.error('Incorrect path to Test Dataset')

    if not (os.path.isdir(args.par_dir)):
        logging.error('Incorrect path to Paragraph directory')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    t = Test(
        model_dir = args.model_dir, 
        test_df_path = args.test_path, 
        output_path = args.output_dir,
        paragraph_dir = args.par_dir
    )

    logging.info('Initiating T5 Retriever Testing....')
    t.test()
    logging.info('T5 Retriever Testing completed')