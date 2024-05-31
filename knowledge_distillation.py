import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from flant5 import CNNDailyMailDataset, Collater  # Ensure you have this module available
import argparse
import os
import random

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(state, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        loss = state['loss']
        print(f"Checkpoint loaded: epoch {epoch}, loss {loss}")
        return epoch, loss
    else:
        print("No checkpoint found")
        return 0, float('inf')

def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return checkpoints[-1]

# Prune the student model by removing transformer layers
def prune_student_model(student, pruning_rate):
    total_layers = len(student.encoder.block)
    num_layers_to_prune = int(total_layers * pruning_rate)
    print(f"Pruning {num_layers_to_prune} layers out of {total_layers}")
    
    decoder_indices = list(range(total_layers))
    random.shuffle(decoder_indices)
    prune_indices = decoder_indices[:num_layers_to_prune]
    for idx in sorted(prune_indices, reverse=True):
        del student.decoder.block[idx]

    return student

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, early_stopping, checkpoint_dir):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        start_epoch, _ = load_checkpoint(student, optimizer, os.path.join(checkpoint_dir, latest_checkpoint), device)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            inputs, labels = batch['input_ids'].to(device), batch['summary_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['summary_ids'].to(device)  # Assuming the labels can be used as decoder input ids

            optimizer.zero_grad()

            # Forward pass with the teacher model
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                teacher_logits = teacher_outputs.logits

            # Forward pass with the student model
            student_outputs = student(input_ids=inputs, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            student_logits = student_outputs.logits

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits.view(-1, student.config.vocab_size), labels.view(-1))

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, Loss: {running_loss / (step + 1)}")

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Save checkpoint after each epoch
        save_checkpoint(student, optimizer, epoch, epoch_loss, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))


        early_stopping(epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save the student model checkpoint
    student.save_pretrained('/pub/yujeony/student_model_checkpoint')

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation Training')
    parser.add_argument('--teacher_model', type=str, default='google/flan-t5-base', help='Teacher model for distillation')
    parser.add_argument('--pruning_rate', type=float, default=0.5, help='Pruning rate for student model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--checkpoint_dir', type=str, default='/pub/yujeony/logs', help='Directory to save checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model = T5ForConditionalGeneration.from_pretrained(args.teacher_model).to(device)
    student_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(device)
    student_model = prune_student_model(student_model, args.pruning_rate)

    dataset = load_dataset('cnn_dailymail', '3.0.0')
    train_dataset = CNNDailyMailDataset(dataset, split='train')
    tokenizer = T5Tokenizer.from_pretrained(args.teacher_model)
    collator = Collater(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)

    early_stopping = EarlyStopping(patience=3, min_delta=0)
 
    # Train the student model using knowledge distillation
    train_knowledge_distillation(
        teacher=teacher_model,
        student=student_model,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        T=2,
        soft_target_loss_weight=0.25,
        ce_loss_weight=0.75,
        device=device,
        early_stopping=early_stopping,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save the tokenizer
    tokenizer.save_pretrained('/pub/yujeony/student_model_checkpoint')

if __name__ == "__main__":
    main()