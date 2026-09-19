import argparse
import time
import torch
import json
from pathlib import Path
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_PATH = ROOT / "data" / "claims" / "train_formatted.jsonl"
VAL_PATH = ROOT / "data" / "claims" / "val_formatted.jsonl"
# Measured (not guessed): the real max token length across every train+val formatted
# row is 804. 1024 was never actually truncating anything, just padding out the
# sequence-length ceiling further than the data needs, which costs real compute on
# every step (attention scales worse than linearly with sequence length). 896 keeps
# a comfortable buffer above the measured max without carrying 220 unused tokens of
# headroom on every single forward/backward pass.
MAX_SEQ_LENGTH = 896

class ClaimsDataset(Dataset):
    # Method overriding -> inheritance.
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            raw_rows = []

            for line in f:
                # removing empty lines and blank rows to prevent json.loads from crashing.
                # If empty, we skip.
                if line.strip():
                    # Converting to python dict
                    parsed_line = json.loads(line)
                    raw_rows.append(parsed_line)

        # Pre-tokenize every row once here in __init__, instead of doing it inside
        # __getitem__. The DataLoader calls __getitem__ once per example per epoch,
        # so tokenizing on-the-fly meant re-running apply_chat_template on the same
        # 99 rows 3 times over (once per epoch) for no benefit -- the tokenized
        # result never changes between epochs. Doing it once up front trades a small
        # amount of extra time at dataset-construction to save that repeated
        # CPU-side tokenization cost on every subsequent epoch.
        self.examples = [self._build_example(row) for row in raw_rows]

    def _build_example(self, row):
        prompt_msgs = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]}
        ]
        full_msgs = prompt_msgs + [
            {"role": "assistant", "content": row["output"]}
        ]

        # Applying chat template. Generation prompt added to prompt_ids so model knows it is its turn to speak.
        # apply_chat_template(tokenize=True) returns a BatchEncoding (dict-like, keys
        # "input_ids"/"attention_mask") on this transformers version, not a bare
        # list[int] -- pull the actual token list out of it explicitly.
        prompt_ids = self.tokenizer.apply_chat_template(prompt_msgs, tokenize = True, add_generation_prompt = True)["input_ids"]
        full_ids = self.tokenizer.apply_chat_template(full_msgs, tokenize = True, add_generation_prompt = False)["input_ids"]

        # Truncating to ensure memory doesn't run out.
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # So we don't overwrite original input_ids.
        labels = list(full_ids)

        # Replacing prompt tokens in the target labels with -100.
        # So we ignore this position for the loss.
        prompt_len = min(len(prompt_ids), len(full_ids)) # other case only occurs when
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    # Now just returns the tensors computed once in __init__, instead of redoing the
    # chat-template + masking work on every call.
    def __getitem__(self, idx):
        return self.examples[idx]

def setup_lora(model, rank, alpha):
    # peft: parameter-efficient-fine-tuning
    # Creating config object telling PEFT where and how to inject the lora matrices. 
    # r -> rank = inner dimension of decomposed update matrices. 
    # lora_alpha -> scaling factor = scales magnitude of lora updates beore adding them back to frozen model weights
    # target_modules -> specifying which linear project layers inside the transformer attention blocks receive LoRA adapter matrices.
    # CAUSAL_LM -> informs peft it is a autoregressive decoder model.
    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")

    model = get_peft_model(model, config)

    for p in model.parameters():
        # Çasts only the tiny, trainable LoRA adapter weights to full 32-bit float 
        # so optimizer can calculate microscopic gradient updates accurately without rounding them to zero
        # while keeping the massive base model in 16-bit to save memory.
        if p.requires_grad:
            p.data = p.data.float()

    model.print_trainable_parameters()

    return model


# Diagnostic fix for the escalating per-step slowdown (11s -> 51s -> 189s/it) seen
# during training. Every example here has a different token length (batch size 1,
# no padding), so every step hands MPS's memory allocator a differently-shaped
# tensor. MPS's caching allocator is less mature than CUDA's and can fragment badly
# under that pattern -- each new shape may force a fresh allocation instead of
# reusing freed memory, and the fragmentation compounds step over step. Forcing an
# empty_cache() after every step releases whatever MPS was holding onto, so
# fragmentation can't accumulate across steps even though shapes keep varying.
class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.mps.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA Fine-Tuning Script"
    )

    parser.add_argument(
        "--rank", type=int, default=8, help="LoRA attention mechanism (rank)"
    )

    parser.add_argument(
        "--alpha", type=int, default=16, help="LoRA scaling parameter (alpha)"
    )

    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting run with configuration:", args)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        dtype=torch.bfloat16
    ).to("mps")

    # Injecting lora adapters to model. 
    model = setup_lora(model, rank=args.rank, alpha=args.alpha)

    # Creating new instances of claimsdataset. (calls init)
    train_set = ClaimsDataset(TRAIN_PATH, tokenizer, MAX_SEQ_LENGTH)
    val_set = ClaimsDataset(VAL_PATH, tokenizer, MAX_SEQ_LENGTH)

    # Configuring training arguments
    training_args = TrainingArguments(per_device_train_batch_size=1, per_device_eval_batch_size=1, 
                                    gradient_accumulation_steps=8, num_train_epochs=args.epochs,
                                    eval_strategy="epoch", logging_strategy="epoch", save_strategy="no", 
                                    report_to=[], learning_rate=2e-4, output_dir=str(ROOT / "outputs" / "model_out"))

    # Instantiating trainer -> automatically ahndles training loop, gradient accumulation, device management, evaluation and logging.
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
        callbacks=[EmptyCacheCallback()],
    )

    start_time = time.monotonic()
    result = trainer.train()
    end_time = time.monotonic()

    duration = end_time - start_time
    print("Training complete in:", duration, "seconds")

    out_dir = ROOT / "outputs" / ("lora-claims-extractor-r" + str(args.rank))

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved adapter and tokenizer to:", out_dir)

    print("Final Training loss:", result.training_loss)
    print("Log History:")
    for entry in trainer.state.log_history:
        print(entry)


if __name__ == "__main__":
    main()