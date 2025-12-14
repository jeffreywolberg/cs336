import argparse

from pydantic import BaseModel, Field

TOTAL_TOKENS_PROCESSED = 327680000

class TrainHyperparameters(BaseModel):
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
    batch_size: int = Field(..., description="Batch size")
    total_tokens_processed: int = Field(..., description="Number of total tokens to process")
    train_path: str = Field(..., description="Path to the training data file")
    val_path: str = Field(..., description="Path to the validation data file")

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
            batch_size=args.batch_size,
            total_tokens_processed=args.total_tokens_processed,
            train_path=args.train_path,
            val_path=args.val_path,
        )


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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--total_tokens_processed", type=int, default=327_680_000, help="Number of total tokens to process = batch_size x total_step_count x context_length")
    parser.add_argument("--train_path", type=str, default="data/TinyStoriesV2-GPT4-train.txt", help="Path to the training data file")
    parser.add_argument("--val_path", type=str, default="data/TinyStoriesV2-GPT4-valid.txt", help="Path to the validation data file")
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    train_hyperparams = TrainHyperparameters.from_args(args)

    print(train_hyperparams)


