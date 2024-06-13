import sys

def print_virtual_environment():
    print("Current virtual environment:", sys.prefix)

print_virtual_environment()

from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune
import evaluate
import torch.utils.benchmark as benchmark
import itertools
import argparse
import logging
import time
import warnings

warnings.filterwarnings('ignore')

class CNNDailyMailDataset(Dataset):
    def __init__(self, dataset, split='test', transform=None):
        self.data = dataset[split]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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

        article_encodings = self.tokenizer(articles, padding=True, truncation=True, return_tensors='pt')
        summary_encodings = self.tokenizer(summaries, padding=True, truncation=True, return_tensors='pt')

        return {
            'input_ids': article_encodings['input_ids'].cuda(),
            "summaries": summaries,
            'summary_ids': summary_encodings['input_ids'].cuda(),
        }

def main(args):
    logging.basicConfig(filename='base-small-fp16-output.log', level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()

    start_time = time.time()

    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = CNNDailyMailDataset(dataset, split='train')
    test_dataset = CNNDailyMailDataset(dataset, split='test')

    logger.info("Number of training samples: %d", len(train_dataset))
    logger.info("Number of test samples: %d", len(test_dataset))

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).cuda()
    assistant_model = T5ForConditionalGeneration.from_pretrained(args.assistant_model).cuda()

    collator = Collater(tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    ## Global Pruning Parameters [Feed Forward Layer]
    no_epochs = 5
    no_iterations = 5
    pruning_amount = 0.2

    parameters_to_prune = ()
    for module in model.decoder.block:
        ff_module = module.layer[2].DenseReluDense
        parameters_to_prune += (
            (ff_module.wi_0, 'weight'),
            (ff_module.wi_1, 'weight'),
            (ff_module.wo, 'weight'),
        )
    
    rouge = evaluate.load('rouge')

    def forward(current_model, batch, enabled):
        inputs = batch["input_ids"]
        outputs = current_model.generate(inputs, assistant_model=assistant_model if enabled else None)
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_output

    def forward_with_score(current_model, batch, enabled):
        decoded_output = forward(current_model, batch, enabled)
        predictions.append(decoded_output[0])
        references.append(batch['summaries'][0])
        return decoded_output
    

    if args.pruning_type == 'iterative' or args.pruning_type == "movement":

        class MovementPruningMethod(prune.BasePruningMethod):
            """
            Prune entries in a tensor based on their movement from the previous iteration.
            """
            PRUNING_TYPE = 'unstructured'

            def __init__(self, previous_t, iteration, amount):
                super().__init__()
                self.previous_t = previous_t
                self.iteration = iteration
                self.amount = amount

            def compute_mask(self, t, default_mask):
                if self.iteration > 0:
                    movement = torch.abs(t - self.previous_t)
                    sorted_indices = torch.argsort(movement, descending=True)
                    num_to_keep = int((1 - self.amount) * t.numel())
                    flat_tensor = t.view(-1)
                    mask = default_mask.view(-1)
                    mask[sorted_indices[num_to_keep:]] = 0
                    mask = mask.view_as(default_mask)
                else:
                    mask = default_mask

                return mask
        
        def movement_unstructured(module, name, amount, previous_t, iteration):
            movement_pruning = MovementPruningMethod(previous_t, iteration, amount)
            movement_pruning.apply(module, name)
            return module

        for iteration in range(no_iterations):
            logger.info('-' * 100)
            logger.info('Iteration - %d', iteration)
            logger.info('-' * 100)

            predictions = []
            references = []
            benchmark_results = []

            # Pruning
            if args.pruning_type == 'iterative':
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_amount,
                )
            else:
                for module, name in parameters_to_prune:
                    movement_unstructured(module, name, pruning_amount, module[name], iteration)

            ## Fine-Tuning Process
            for epoch in range(no_epochs):
                model.train()
                for i, batch in enumerate(train_dataloader):
                    loss = model(input_ids=batch["input_ids"], labels=batch['summary_ids']).loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i == 1000:
                        break

                logger.info('Epoch - %d, Loss - %f completed', epoch, loss)
            
            # Evaluation and Inference

            model.eval()
            for (enabled,) in itertools.product([True, False]):
                def profile():
                    for i, batch in enumerate(test_dataloader):
                        if i == args.num_gen:
                            break
                        decoded_output = forward_with_score(model, batch, enabled)
            
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

            results = rouge.compute(predictions=predictions, references=references)

            for key, value in results.items():
                logger.info('%s: %f', key, value)

            elapsed_time = time.time() - start_time
            logger.info('Total time elapsed: %f seconds', elapsed_time)
    elif args.pruning_type == 'structured':
        ## Structured Pruning
        for module, type in parameters_to_prune:
            prune.ln_structured(module, type, amount=pruning_amount, dim=1)

    elif args.pruning_type == 'unstructured':
        ## Unstructured Pruning
        for module, type in parameters_to_prune:
            prune.ln_unstructured(module, type, amount=pruning_amount)
    
    elif args.pruning_type == 'unstructured':
        ## Global Pruning
        prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount,
        )
    
    if args.pruning_type != 'iterative' and  args.pruning_type != 'movement':
        for (enabled,) in itertools.product([True, False]):
                def profile():
                    for i, batch in enumerate(test_dataloader):
                        if i == args.num_gen:
                            break
                        decoded_output = forward_with_score(model, batch, enabled)
            
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

        results = rouge.compute(predictions=predictions, references=references)

        for key, value in results.items():
            logger.info('%s: %f', key, value)

        elapsed_time = time.time() - start_time
        logger.info('Total time elapsed: %f seconds', elapsed_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument parser for model options')
    parser.add_argument('--model', type=str, default="google/flan-t5-base", help='Path or name of the main model')
    parser.add_argument('--assistant_model', type=str, default="google/flan-t5-small", help='Path or name of the assistant model')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp32'], default="fp16", help='Precision of the model')
    parser.add_argument('--num_gen', type=int, default=100, help="Number of generations")
    parser.add_argumets('--pruning_type', type=str, default="unstructured", help="Flag for Pruning Type")

    args = parser.parse_args()
    main(args)
