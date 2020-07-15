import torch
import pandas as pd
from os.path import join
from torch.utils.data import Dataset

class LJSpeechDataset(Dataset):
    def __init__(self, path, partition_table, split, bucketing, batch_size, use_spkr):
        """
        Arg:
            split: one in ['train', 'dev', 'test']
            path: root directory of LJSpeech (i.e. /yourPath/LJSpeech-1.1)
            bucket_size: bucket size for bucketing
        """
        # Setup
        self.path = path
        self.use_spkr = use_spkr # Will not be used.
        self.split = split
        self.batch_size = batch_size
        self.bucketing = bucketing and (split not in ['dev','test'])
        self.bs_for_collate = 1 if self.bucketing else self.batch_size
        
        # Select split and sort lenght
        table = pd.read_csv(partition_table, index_col=0)
        table = table[table.split == split]
        if len(table)==0:
            # Empty partition
            self.table = table.append({'split':split,'duration':0}, ignore_index=True)
        else:
            table['file_path'] = table.apply(lambda row: join(path, 'wavs', row.name+'.wav'), axis=1)
            table = table.sort_values('duration', axis=0, ascending=False)
            self.table = table

    def get_statics(self):
        return '           | {} size = {}\t| Duration = {:.1f}\t| Bucketing = {} '\
               .format(self.split.replace('unpaired','unpair'), len(self.table), self.table.duration.sum()/60, self.bucketing)

    def __getitem__(self,index):
        if self.bucketing:
            # Return a bucket
            index = min(len(self.table)-self.batch_size,index)
            wav_list = self.table.iloc[index:index+self.batch_size].file_path.tolist()
            return list(zip(wav_list, [None]*self.batch_size))
        else:
            return self.table.iloc[index].file_path, None

    def __len__(self):
        return len(self.table)
