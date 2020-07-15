import json
import torch
import pandas as pd
from os.path import join
from torch.utils.data import Dataset



class VCTKDataset(Dataset):
    def __init__(self, path, partition_table, split, bucketing, batch_size, spkr_map):
        """
        Arg:
            split: one in ['train', 'dev', 'test']
            path: root directory of VCTK (i.e. /yourPath/VCTK-Corpus)
            bucket_size: bucket size for bucketing
        """
        # Setup
        self.path = path
        self.split = split
        self.batch_size = batch_size
        self.bucketing = bucketing and (split not in ['dev','test'])
        self.bs_for_collate = 1 if self.bucketing else self.batch_size
        self.spkr_map = json.load(open(spkr_map))
        # Select split and sort lenght
        table = pd.read_csv(partition_table, index_col=0)
        table = table[table.split == split]
        if len(table)==0:
            # Empty partition
            self.table = table.append({'speaker':0,'split':split,'duration':0}, ignore_index=True)
        else:
            table['file_path'] = table.apply(lambda row: join(path, row.speaker, row.name+'.wav'), axis=1)
            table['speaker'] = table.apply(lambda row: self.spkr_map[row.speaker], axis=1)
            table = table.sort_values('duration', axis=0, ascending=False)
            self.table = table if split != 'test' else table[table.speaker != self.spkr_map['lj']]
        self.n_spkr = len(spkr_map)

    def get_statics(self):
        return '           | {} size = {}\t| Duration = {:.1f}\t| Bucketing = {} '\
               .format(self.split.replace('unpaired','unpair'), len(self.table), self.table.duration.sum()/60, self.bucketing)

    def __getitem__(self,index):
        if self.bucketing:
            # Return a bucket
            index = min(len(self.table)-self.batch_size,index)
            wav_list = self.table.iloc[index:index+self.batch_size].file_path.tolist()
            spkr_list = self.table.iloc[index:index+self.batch_size].speaker.tolist()
            return list(zip(wav_list, spkr_list))
        else:
            return self.table.iloc[index].file_path, self.table.iloc[index].speaker

    def __len__(self):
        return len(self.table)
