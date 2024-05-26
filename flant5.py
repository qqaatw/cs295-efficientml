from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
import torch.utils.benchmark as benchmark
import itertools
import argparse

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

def main(args):
    # Load the wikitext-2-raw-v1 dataset
    dataset = load_dataset('cnn_dailymail', '3.0.0')

    # Create an instance of the custom dataset
    test_dataset = CNNDailyMailDataset(dataset, split='test')

    # Print some basic information about the dataset
    #print("Number of training samples:", len(train_data))
    #print("Number of validation samples:", len(validation_data))
    print("Number of test samples:", len(test_dataset))

    # Print the first testing example
    print("\nFirst testing example:")
    print(test_dataset[0]['article'])
    print(test_dataset[0]['summary'])

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model, device_map="auto")
    assistant_model = T5ForConditionalGeneration.from_pretrained(args.assistant_model, device_map="auto")

    collator = Collater(tokenizer)

    # Create a dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    input_text = test_dataset[0]['article']
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    #outputs = model.generate(input_ids, assistant_model=assistant_model)
    #decoded_output = tokenizer.decode(outputs[0])

    rouge = evaluate.load('rouge')
    predictions = []
    references = []
    benchmark_results = []

    def forward(batch, enabled):
        inputs = batch["input_ids"].cuda()
        outputs = model.generate(inputs, assistant_model=assistant_model if enabled else None)
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_output

    def forward_with_score(batch, enabled):
        decoded_output = forward(batch, enabled)
        predictions.append(decoded_output[0])
        references.append(batch['summaries'][0])
        return decoded_output

    for (enabled,) in itertools.product([True, False]):
        def profile():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    print("idx", i)
                if i == args.num_gen:
                    break
                decoded_output = forward(batch, enabled)
    
        t = benchmark.Timer(
                        stmt='profile()',
                        label='Speculative Decoding',
                        sub_label=f"",
                        globals=locals(),
                        description= f"Enabled: {enabled}",
                    ).blocked_autorange(min_run_time=5)
        benchmark_results.append(t)
    
    compare = benchmark.Compare(benchmark_results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()

    #results = rouge.compute(predictions=predictions, references=references)
    #print(results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for model options')

    # Add arguments
    parser.add_argument('--model', type=str, default="google/flan-t5-base", help='Path or name of the main model')
    parser.add_argument('--assistant_model', type=str, default="google/flan-t5-small", help='Path or name of the assistant model')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp32'], default="fp32", help='Precision of the model')
    parser.add_argument('--num_gen', type=int, default=100, help="")

    args = parser.parse_args()
    main(args)