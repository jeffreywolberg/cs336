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

