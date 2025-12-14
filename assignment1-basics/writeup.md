## Writeup for Assignment 1

### BPE Tokenizer
#### 2.1
a) chr maps from int->str, ord maps from str->int. chr(0) is the null character

b) Repr prints out its bytes representation, while in printed form it is null

c) You can't see the character

#### 2.2
a) If alphabet of most common languages is quite small, most characters can be represented in 1 byte, while utf-16 can only represent these chars with two bytes. If alphabet is quite large than utf-16 might be correct since first 2^16 letters can be represented in two bytes in this format, while in utf-8 it might take 3 bytes to encode some of the first 2^16 letters (anything from U+0080 to U+07FF takes two byets, anything from U+0800 to U+FFFF will take 3 bytes). 

b) It's decoding the bytes individually, which means it cannot decode any multi-byte character. `"hello! こんにちは!"` fails. UTF-8 expects ASCII to be between 0-127, while 128-255 are reserved for multi-bytes sequences.

c) `bytearray(b'\xe3\x81')` fails since `e3` is a continuation byte for two more bytes for utf-8.

#### 2.3

### TransformerLM

#### 3.6

Embedding = 0
Transformer Block:
    layer norm: d_model 
    MHSA:
        proj of qkv: 3 * [d_model * S * d_model] * 2 (matmul flops = 2x)
        (no matmul) qk rope: num_heads * 2 * (4 * (S * d_k/2) mults + 2 * (S * d_k/2) additions)
        attn: [num_heads * S * S * d_k] * 2 (matmul flops = 2x)
        out_proj : S * d_model * d_ff * 2 (matmul flops = 2x)
    layer norm: d_model
    FFN:
        w1 @ x = S * d_model * d_ff * 2 (matmul flops = 2x)
        w3 @ silu = S * d_model * d_ff * 2 (matmul flops = 2x)
        w2 = d_model * S * d_ff
ln_final: d_model
ln_head: d_model * S * vocab_size

a) 
```
ln = d_model
proj_qkv = 3 * (d_model * d_model)
attn = 0
out_proj = d_model ** 2

ffn_w1_w2_w3 = 3 * d_model * d_ff

embedding = vocab_size * d_model
transformer_block = (ln + (proj_qkv + attn + out_proj) + (ffn_w1_w2_w3 + ln))
ln_final = ln
lm_head = d_model * vocab_size

total = embedding + num_layers * transformer_block + ln_final + lm_head
torch_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {total}")
print(f"Pytorch total: {torch_total}")
```

This leads to a total traininable parameter count of 2,127,057,600. If each param is 32 bit float, that's ~8GB of memory.

b) 
```
ln = d_model
proj_qkv = 3 * (d_model * context_length * d_model) * 2
attn = num_heads * context_length ** 2 * (d_model // num_heads) * 2
out_proj = context_length * d_model * d_model * 2

ffn_w1_w2_w3 = 3 * context_length * d_model * d_ff * 2

embedding = vocab_size * d_model * 2
transformer_block = (ln + (proj_qkv + attn + out_proj) + (ffn_w1_w2_w3 + ln))
ln_final = ln
lm_head = d_model * context_length * vocab_size * 2

print(f"embedding: {embedding}")
print(f"transformer_block: {transformer_block}")
print(f"\tproj_qkv: {proj_qkv}")
print(f"\tattn: {attn}")
print(f"\tout_proj: {out_proj}")
print(f"\tffn_w1_w2_w3: {ffn_w1_w2_w3}")
print(f"transformer_block: {transformer_block}")

total_flops = embedding + num_layers * transformer_block + ln_final + lm_head
print(f"total flops: {total_flops}")
```

```
embedding: 160822400
transformer_block: 87241526400
        proj_qkv: 15728640000
        attn: 3355443200
        out_proj: 5242880000
        ffn_w1_w2_w3: 62914560000
transformer_block: 87241526400
total flops: 4352436228800
```

c) 
Proporitonally, the ffn's require the most FLOPS, around 70% of them, with attn + projections requiring rouhgly the other 30%.

d) 
As model gets bigger, the attn block uses more FLOPS, since d_ff remains the same.

e) When context length increases, the FLOPS of the ffn within the transformer block goes from 72% to 45%, since attn starts to dominate due to its context_length**2 time complexity. Attn uses 40% of FLOPS with context length = 16,384.

4)a) for every param in the model, you need to store 5 things: (param value, param grad, and m, v, t from adamw). The computation for memory usage of activations is separate, and follows 3b.