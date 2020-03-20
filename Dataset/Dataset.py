import unicodedata
from pandas import DataFrame
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, df: DataFrame):
        self.data_frame = df
        self.data_frame['name'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
        self.data_frame['first'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())
        self.data_frame['last'].apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]

        return str(row['name']), str(row['first']), str(row['middle']), str(row['last']), int(row['format'])
