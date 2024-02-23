import fitz
import re
import logging
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.cli import tqdm

logging.basicConfig(filename='./Logs/pdf_extractor.log', filemode='w', datefmt='%H:%M:%S', level=logging.ERROR)

# Add path to directory which contain the reference papers
BIB_DIR = './bib_papers'

# Add path to directory to contain the extracted content from PDFs
OUTPUT_DIR = './extracted_content/'


class PDFExtractor():
    def __init__(self) -> None:
        self.start_pattern = re.compile(r'Abstract(.*)|ABSTRACT(.*)', re.DOTALL)
        self.end_pattern = re.compile(r'(.+?)(?:Acknowledgements|References|Bibliography|ACKNOWLEDGEMENTS|REFERENCES|BIBLIOGRAPHY|$)', re.DOTALL)


    def __cleanText(self, text: str) -> str:

        # Remove multiple white spaces and newline characters
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')

        # Remove word continuation ('- ') by joining hyphenated words
        text = re.sub(r'-\s', '', text)

        # Regex pattern to remove everything except alphabetical text and punctuations
        text = re.sub(r'[^A-Za-z\s,.!?\'"-]', '', text)

        # Regex pattern to remove links (URLs)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        return text.strip()


    def extractContent(self, file_path: str, truncate_content: bool = True) -> str:

        with fitz.open(file_path) as doc:
            text = ' '.join([page.get_text() for page in doc]).encode('cp1252', errors='ignore').decode('cp1252')

        text = self.__cleanText(text)

        if truncate_content:

            # Keep text from abstract
            matches = re.search(self.start_pattern, text)

            if matches:
                text = matches.group().strip()

            # print(text)

            # Keep Text till reference
            matches = re.search(self.end_pattern, text)

            if matches:
                text = matches.group(1).strip()

        return text
    

class DatasetGenerator():

    def __init__(self) -> None:    
        self.pdf_extractor = PDFExtractor()


    def __sentenceTokenizer(self, text: str) -> list:

        # Split content on .,?,!
        sentences = re.split(r'[.!?]', text)

        # Remove sentences with less than 5 words
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]

        # Try to remove sentences part of references
        sentences = [s for s in sentences if not any(w in s.lower() for w in ['vol', 'pp', 'page', 'doi', 'ieee', 'arxiv'])]

        return sentences
    
    
    def __gettextBlock(self, text: str) -> list:

        sentences = self.__sentenceTokenizer(text)

        blocks = []
        num_sentences = len(sentences)
        block_size = 7
        stride = 3

        for i in range(0, num_sentences - block_size + 1, stride):
            block = '. '.join(sentences[i:i+block_size])
            blocks.append(block)

        return blocks
    

    def processPDF(self, bib_id: str):

        # Get PDF Content
        try:
            content = self.pdf_extractor.extractContent(file_path = os.path.join(BIB_DIR, f'{bib_id}.pdf'))

        except Exception as e:
            logging.error(f'[File Error]: {bib_id} : {e}')
            return None


        # Chunk into Text Blocks
        blocks = self.__gettextBlock(content)

        # Ensure sufficient data has been extracted
        for flag in [True, False]:

            # Get PDF Content
            try:
                content = self.pdf_extractor.extractContent(file_path = os.path.join(BIB_DIR, f'{bib_id}.pdf'), truncate_content = flag)

            except Exception as e:
                logging.error(f'[File Error]: {bib_id} : {e}')
                return None
            
            # Chunk into Text Blocks
            blocks = self.__gettextBlock(content)

            if len(blocks) >= 6:
                break
                
        else:
            logging.error(f'[Incomplete Extraction]: {bib_id}')
            return None
        

        # Save Paragraph into DataFrame
        pd.DataFrame.from_dict(
            {
                'bib_id': [bib_id]*len(blocks),
                'pid': list(range(len(blocks))),
                'paragraph': blocks
            }
        ).to_pickle(os.path.join(OUTPUT_DIR, f'{bib_id}.pkl'))

        return None


if __name__ == '__main__':
    
    dataset_generator = DatasetGenerator()

    print('Fetching bib_ids of PDFs...')
    bib_ids = pd.read_pickle('./DataFrames/df_bibs_v3.pkl').index.tolist()

    # Get bib_ids not processed yet
    print('Checking PDFs yet to process...')
    bib_ids = [bib_id for bib_id in bib_ids if not os.path.isfile(os.path.join(OUTPUT_DIR, f'{bib_id}.pkl'))]

    print('Processing PDFs...')
    with tqdm(total = len(bib_ids)) as pbar:
        with ThreadPoolExecutor(max_workers = 10) as executor:
            futures = [executor.submit(dataset_generator.processPDF, bib_id) for bib_id in bib_ids]
            for future in as_completed(futures):
                pbar.update(1)

    print('Done')