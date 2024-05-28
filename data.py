from torch.utils.data import Dataset

class CNNDailyMailDataset(Dataset):
    def __init__(self, dataset, split='test', transform=None):
        """
        Args:
            dataset (DatasetDict): The Hugging Face dataset dictionary.
            split (str): Which split of the dataset to use ('train', 'validation', 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = dataset[split]
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            sample (dict): A dictionary containing the article and summary.
        """
        article = self.data[idx]['article']
        summary = self.data[idx]['highlights']
        sample = {'article': article, 'summary': summary}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Collater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        articles = [item['article'] for item in batch]
        summaries = ["summarize: " + item['summary'] for item in batch]

        # Tokenize the articles and summaries
        article_encodings = self.tokenizer(articles, padding=True, truncation=True, return_tensors='pt')
        #summary_encodings = self.tokenizer(summaries, padding=True, truncation=True, return_tensors='pt')

        return {
            'input_ids': article_encodings['input_ids'],
            "summaries": summaries,
            #'attention_mask': article_encodings['attention_mask'],
            #'summary_ids': summary_encodings['input_ids'],
            #'summary_attention_mask': summary_encodings['attention_mask'],
        }