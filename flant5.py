from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import evaluate
import torch.utils.benchmark as benchmark
import itertools
import argparse

from data import CNNDailyMailDataset, Collater

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

    rouge = evaluate.load('rouge')
    predictions = []
    references = []
    benchmark_results = []

    def forward(model, assistant_model, batch, enabled):
        inputs = batch["input_ids"].cuda()
        outputs = model.generate(inputs, assistant_model=assistant_model if enabled else None)
        return outputs

    def forward_with_score(model, assistant_model, batch, enabled):
        outputs = forward(batch, enabled)
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.append(decoded_output[0])
        references.append(batch['summaries'][0])
        return decoded_output

    for (enabled, assistant_model_name, main_model_name, precision) in itertools.product(
            [True, False],
            ["google/flan-t5-small"],
            ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"],
            ["INT8", torch.bfloat16, torch.float16, torch.float32],
        ):

        sublabel = f"Main model: {main_model_name} Assistant model: {assistant_model_name} Precision: {precision}"

        print(f"Testing: {sublabel}")

        tokenizer = T5Tokenizer.from_pretrained(main_model_name)
        if isinstance(precision, torch.dtype):
            model = T5ForConditionalGeneration.from_pretrained(main_model_name, device_map="auto", torch_dtype= precision)
            assistant_model = T5ForConditionalGeneration.from_pretrained(assistant_model_name, device_map="auto", torch_dtype=precision)
        else:
            model = T5ForConditionalGeneration.from_pretrained(main_model_name, device_map="auto", load_in_8bit=True)
            assistant_model = T5ForConditionalGeneration.from_pretrained(assistant_model_name, device_map="auto", load_in_8bit=True)
            

        collator = Collater(tokenizer)

        # Create a dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collator)
        
        def profile():
            for i, batch in enumerate(test_dataloader):
                if i % 10 == 0:
                    print("idx", i)
                if i == args.num_gen:
                    break
                outputs = forward(model, assistant_model, batch, enabled)
    
        t = benchmark.Timer(
                        stmt='profile()',
                        label='Speculative Decoding',
                        sub_label=sublabel,
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
    #parser.add_argument('--model', type=str, default="google/flan-t5-base", help='Path or name of the main model')
    #parser.add_argument('--assistant_model', type=str, default="google/flan-t5-small", help='Path or name of the assistant model')
    #parser.add_argument('--precision', type=str, choices=['fp16', 'fp32'], default="fp32", help='Precision of the model')
    parser.add_argument('--num_gen', type=int, default=100, help="")

    args = parser.parse_args()
    main(args)