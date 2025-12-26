import argparse
import math
import os
from typing import List, Optional
import numpy as np
from os.path import join, basename, exists, splitext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field
from torch import nn
import torch
from tqdm import tqdm

from cs336_basics.generate import generate_text
from cs336_basics.loss import CrossEntropy
from cs336_basics.model import TransformerLM, load_checkpoint, save_checkpoint
from cs336_basics.opt import AdamW
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.dataset import get_batch

TRAINING_DIR = "training_runs"
CACHE_DIR = "data/cache"
TOKENIZER_CACHE_DIR = join(CACHE_DIR, "tokenizer")
DATASET_CACHE_DIR = join(CACHE_DIR, "dataset")
TOKENIZER_SPECIAL_TOKENS = ["<|endoftext|>"]

N_VAL = 2000
N_VAL_STEPS = 25
SAVE_MODEL_EVERY_N_ITERS = 25


class TrainConfig(BaseModel):
    vocab_size: int = Field(..., description="Vocabulary size")
    context_length: int = Field(..., description="Context length")
    d_model: int = Field(..., description="Transformer model dimension")
    d_ff: int = Field(..., description="Feed-forward (FFN) dimension")
    rope_theta: float = Field(..., description="Rotary Embedding theta")
    num_layers: int = Field(..., description="Number of transformer layers")
    num_heads: int = Field(..., description="Number of attention heads")
    learning_rate: float = Field(..., description="AdamW base learning rate")
    learning_rate_warmup: int = Field(..., description="Learning rate warmup steps")
    adamw_beta1: float = Field(..., description="AdamW beta1")
    adamw_beta2: float = Field(..., description="AdamW beta2")
    adamw_eps: float = Field(..., description="AdamW epsilon")
    adamw_weight_decay : float = Field(..., description="AdamW weight decay")
    batch_size: int = Field(..., description="Batch size")
    total_tokens_processed: int = Field(..., description="Number of total tokens to process")
    train_path: str = Field(..., description="Path to the training data file")
    val_path: str = Field(..., description="Path to the validation data file")
    training_run: str = Field(..., description="Training run identifier string")
    ckpt_path: Optional[str] = Field(None, description="Optional checkpoint path to load from")
    device: str = Field("cpu", description="Device to run on")

    @classmethod
    def from_args(cls, args):
        return cls(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            d_ff=args.d_ff,
            rope_theta=args.rope_theta,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            learning_rate=args.learning_rate,
            learning_rate_warmup=args.learning_rate_warmup,
            adamw_beta1=args.adamw_beta1,
            adamw_beta2=args.adamw_beta2,
            adamw_eps=args.adamw_eps,
            adamw_weight_decay=args.adamw_weight_decay,
            batch_size=args.batch_size,
            total_tokens_processed=args.total_tokens_processed,
            train_path=args.train_path,
            val_path=args.val_path,
            training_run=args.training_run,
            ckpt_path=args.ckpt_path,
            device=args.device,
        )

def get_tokenizer(cfg : TrainConfig) -> BPETokenizer:
    train_data_name = splitext(basename(cfg.train_path))[0]
    tokenizer_dir = join(TOKENIZER_CACHE_DIR, train_data_name)
    vocab_path = join(tokenizer_dir, "vocab.pkl")
    merges_path = join(tokenizer_dir, "merges.pkl")
    if exists(vocab_path) and exists(merges_path):
        tokenizer = BPETokenizer.from_trained_tokenizer_files(vocab_path, merges_path)
        print(f"Loaded trained tokenizer from directory: {tokenizer_dir}")
    else:
        tokenizer = BPETokenizer(special_tokens=TOKENIZER_SPECIAL_TOKENS)
        print(f"Training the tokenizer...", end="\t")
        tokenizer.train(cfg.train_path, cfg.vocab_size)
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save(vocab_path, merges_path)
        print(f"Saved trained tokenizer to directory: {tokenizer_dir}")
        print(f"Done training tokenizer!")
    
    return tokenizer

def get_datasets(cfg : TrainConfig):
    train_dataset_cache_dir = join(DATASET_CACHE_DIR, splitext(basename(cfg.train_path))[0])
    val_dataset_cache_dir = join(DATASET_CACHE_DIR, splitext(basename(cfg.val_path))[0])
    train_dataset_cache_path = join(train_dataset_cache_dir, 'data.npz')
    val_dataset_cache_path = join(val_dataset_cache_dir, 'data.npz')
    
    if not exists(train_dataset_cache_path) or not exists(val_dataset_cache_path):
        tokenizer : BPETokenizer = get_tokenizer(cfg)
        print(f"Getting datassets...", end="\t")
        def get_tokens(text_path):
            def chunk_reader(f, chunk_size=4*1024*1024): # Read 4 MB at a time
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            print(f"Encoding text from: {text_path} (this will take a while)")
            with open(text_path, 'r') as f:
                ids = []
                for _id in tokenizer.encode_iterable(chunk_reader(f)):
                    ids.append(_id)
            return np.array(ids, dtype=np.int32)

        train_tokens : np.ndarray = get_tokens(cfg.train_path)
        val_tokens : np.ndarray = get_tokens(cfg.val_path)
        os.makedirs(train_dataset_cache_dir, exist_ok=True)
        os.makedirs(val_dataset_cache_dir, exist_ok=True)
        np.savez_compressed(train_dataset_cache_path, data=train_tokens)
        np.savez_compressed(val_dataset_cache_path, data=val_tokens)
        print(f"Saved train dataset to {train_dataset_cache_path}")
        print(f"Saved val dataset to {val_dataset_cache_path}")
    else:
        train_tokens = np.load(train_dataset_cache_path, mmap_mode='r')['data']
        val_tokens = np.load(val_dataset_cache_path, mmap_mode='r')['data']
        print(f"Loaded train dataset from {train_dataset_cache_path}")
        print(f"Loaded val dataset from {val_dataset_cache_path}")

    # return val_tokens, val_tokens
    return train_tokens, val_tokens

def get_argparser():
    parser = argparse.ArgumentParser(description="Training arguments for TransformerLM")

    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size (default: 10000)")
    parser.add_argument("--context_length", type=int, default=256, help="Context length (default: 256)")
    parser.add_argument("--d_model", type=int, default=512, help="Transformer model dimension (default: 512)")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feed-forward (FFN) dimension (default: 1344)")
    parser.add_argument("--rope_theta", type=float, default=10000, help="Rotary Embedding theta (default: 10000)")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="AdamW base learning rate (default: 1e-4)")
    parser.add_argument("--learning_rate_warmup", type=int, default=100, help="Learning rate warmup steps (default: 100)")
    parser.add_argument("--adamw_beta1", type=float, default=0.9, help="AdamW beta1 (default: 0.9)")
    parser.add_argument("--adamw_beta2", type=float, default=0.999, help="AdamW beta2 (default: 0.999)")
    parser.add_argument("--adamw_eps", type=float, default=1e-8, help="AdamW epsilon (default: 1e-8)")
    parser.add_argument("--adamw_weight_decay", type=float, default=1.e-2, help="AdamW weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--total_tokens_processed", type=int, default=327_680_000, help="Number of total tokens to process = batch_size x total_step_count x context_length")
    parser.add_argument("--train_path", type=str, default="data/TinyStoriesV2-GPT4-train.txt", help="Path to the training data file")
    parser.add_argument("--val_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", help="Path to the validation data file")
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default="training_runs/2025_12_14/models/model_649_val_loss_1.239.ckpt", help="Ckpt path to load model + opt from")
    parser.add_argument("-device", "--device", type=str, default="mps:0", help="Device to run on")
    
    parser.add_argument("-name", "--training_run", type=str, required=True, help="Training run identifier string")
    return parser

def validate(model : TransformerLM, dataset : np.ndarray,  loss : CrossEntropy, cfg : TrainConfig, device) -> float:
    loss_val = 0.0
    model.eval()
    with torch.no_grad():
        for step in tqdm(range(N_VAL_STEPS), desc="Val step", leave=False):
            input_tokens, targets = get_batch(dataset, cfg.batch_size, cfg.context_length, device)
            input_tokens, targets = input_tokens.to(device), targets.to(device)
            preds = model(input_tokens)
            loss_output = loss(preds.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss_val += loss_output.item()
    return loss_val / N_VAL_STEPS

def train(model : TransformerLM, opt : AdamW, dataset : np.ndarray, loss : CrossEntropy, cfg : TrainConfig, num_train_steps : int, device) -> float:
    loss_val = 0.0
    for step in tqdm(range(num_train_steps), desc="Train step within iter", leave=False):
        opt.zero_grad()
        input_tokens, targets = get_batch(dataset, cfg.batch_size, cfg.context_length, device)
        preds = model(input_tokens)
        loss_output = loss(preds.reshape(-1, cfg.vocab_size), targets.reshape(-1))
        loss_output.backward()
        opt.step()
        loss_val += loss_output.item()
    return loss_val / num_train_steps

def plot_losses(iter : int, train_losses : List[float], val_losses : List[float]):
    iters = range(iter - len(train_losses) + 1, iter+1)
    plt.figure()
    plt.plot(iters, train_losses, label="train loss")
    plt.plot(iters, val_losses, label="val loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    loss_fig_path = join(training_dir, "loss.png")
    plt.savefig(loss_fig_path)
    plt.close()

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    cfg = TrainConfig.from_args(args)
    device = torch.device(args.device)

    tokenizer = get_tokenizer(cfg)
    train_dataset, val_dataset = get_datasets(cfg)

    model = TransformerLM(cfg.vocab_size, cfg.context_length, cfg.d_model, cfg.num_layers, cfg.num_heads, cfg.d_ff, cfg.rope_theta, device=device)
    opt = AdamW(model.parameters(), cfg.learning_rate, (cfg.adamw_beta1, cfg.adamw_beta2), cfg.adamw_eps, cfg.adamw_weight_decay)
    loss = CrossEntropy()

    if cfg.ckpt_path is not None and exists(cfg.ckpt_path):
        start_iter = load_checkpoint(cfg.ckpt_path, model, opt) + 1
    else:
        start_iter = 1

    n_total_steps = math.ceil(cfg.total_tokens_processed / (cfg.batch_size * cfg.context_length))

    n_train_steps_per_iter = n_total_steps // N_VAL
    n_iters = n_total_steps // n_train_steps_per_iter

    training_dir = join(TRAINING_DIR, cfg.training_run)
    os.makedirs(training_dir, exist_ok=True)
    print(f"Training for n_iters={n_iters} from start_iter={start_iter}, n_train_steps_per_iter={n_train_steps_per_iter}")
    model_dir = join(training_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    val_texts = []
    val_text_gen_temp = 2.0
    val_text_p_sample = 0.01
    val_max_token_gen = 64
    val_starting_toks : list[int] = tokenizer.encode("This is the start of an important message:\n")


    torch.autograd.set_detect_anomaly(True)
    val_text = generate_text(model, tokenizer, val_starting_toks, val_max_token_gen, temperature=val_text_gen_temp, p_sample=val_text_p_sample)
    print(f"[{start_iter-1}] Val generated text (temp={val_text_gen_temp}):\n{val_text}\n")
    val_loss_val = validate(model, val_dataset, loss, cfg, device)
    print(f"[{start_iter-1}] Val loss value: {val_loss_val:.3f}")
    for iter in tqdm(range(start_iter, n_iters), desc="Training iter", leave=True):
        train_loss_val = train(model, opt, train_dataset, loss, cfg, n_train_steps_per_iter, device)
        print(f"[{iter}] Train loss: {train_loss_val:.3f}")
        val_text = generate_text(model, tokenizer, val_starting_toks, val_max_token_gen, temperature=val_text_gen_temp, p_sample=val_text_p_sample)
        print(f"[{iter}] Val generated text (temp={val_text_gen_temp}):\n{val_text}\n")
        val_loss_val = validate(model, val_dataset, loss, cfg, device)
        print(f"[{iter}] Val loss: {val_loss_val:.3f}")
        train_losses.append(train_loss_val)
        val_losses.append(val_loss_val)
        val_texts.append(val_text)
        plot_losses(iter, train_losses, val_losses)

        if iter % SAVE_MODEL_EVERY_N_ITERS == 0:
            out_path = join(model_dir, f"model_{iter}_val_loss_{val_loss_val:.3f}.ckpt")
            save_checkpoint(model, opt, iter, out_path)
            print(f"Saved iter={iter} ckpt to {out_path}")

    # After training completes, save train/val loss curves.
