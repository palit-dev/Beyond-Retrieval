import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import argparse

class ClassifierEvaluation:
    def __init__(self, checkpoint_df_paths: list, output_df_path: str = '', annotated_df_path: str = '' ) -> None:
        self.checkpoint_df_paths = sorted(checkpoint_df_paths, key=self.get_checkpoint_number)
        self.df_annotated = pd.DataFrame()
        self.output_path = output_df_path
        
        if output_df_path == '':
            self.output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'evalution_result.pkl'
            )
        
        if annotated_df_path != '':
            self.df_annotated = pd.read_pickle(annotated_df_path)
            
            
    def get_checkpoint_number(self, path: str) -> int:
        return int(path.split('-')[-1].split('.')[0].strip())
    
    
    def evaluate(self) -> None:
        results = {
            'steps': [self.get_checkpoint_number(path) for path in self.checkpoint_df_paths],
            'tp': [0]*len(self.checkpoint_df_paths),
            'fp': [0]*len(self.checkpoint_df_paths),
            'tn': [0]*len(self.checkpoint_df_paths),
            'fn': [0]*len(self.checkpoint_df_paths),
            'precision': [float('nan')]*len(self.checkpoint_df_paths),
            'recall': [float('nan')]*len(self.checkpoint_df_paths),
            'f1 score': [float('nan')]*len(self.checkpoint_df_paths),
        }
        is_annotated_eval = not self.df_annotated.empty
        
        for index, path in enumerate(tqdm(self.checkpoint_df_paths, desc = 'Checkpoints')):
            df_test = pd.read_pickle(path)
            if is_annotated_eval:
                df_test = df_test[df_test['sample_id'].isin(self.df_annotated['sample_id'])]

            for _, item in df_test.iterrows():
                if item['true_output'] == 1 and item['predicted_output'] == 1:
                    results['tp'][index] += 1
                elif item['true_output'] == 1 and item['predicted_output'] == 0:
                    results['fn'][index] += 1
                elif item['true_output'] == 0 and item['predicted_output'] == 0:
                    results['tn'][index] += 1
                elif item['true_output'] == 0 and item['predicted_output'] == 1:
                    results['fp'][index] += 1

            try:
                results['precision'][index] = results['tp'][index] / (results['tp'][index] + results['fp'][index])
                results['recall'][index] = results['tp'][index] / (results['tp'][index] + results['fn'][index])
                results['f1 score'][index] = 2 * results['precision'][index] * results['recall'][index] / (
                    results['precision'][index] + results['recall'][index]
                )

            except ZeroDivisionError :
                pass
            
        df_results = pd.DataFrame.from_dict(results)
        df_results.to_pickle(self.output_path)
        return df_results



if __name__ == '__main__':
    
    checkpoint_df_path = glob(os.path.normpath('DataFrames/df_test_flanT5.pkl'))
    output_df_path = os.path.normpath('evaluation_result_flan.pkl')

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_df_path", help="Path to inferred Dataset", default = checkpoint_df_path, type = str)
    parser.add_argument("-o", "--output_path", help="Path to file to store metrics", default = output_df_path, type = str)
    parser.add_argument("-a", "--annotated_df_path", help="Path to annotated dataset", default = '', type = str)
    
    args = parser.parse_args()
    if not (os.path.isfile(args.result_path)):
        logging.error('Incorrect path to Inferred Dataset')
    
    c = ClassifierEvaluation(
        checkpoint_df_paths = [args.result_df_path], 
        output_df_path = args.output_path,
        annotated_df_path = args.annotated_df_path
    )
    c.evaluate()