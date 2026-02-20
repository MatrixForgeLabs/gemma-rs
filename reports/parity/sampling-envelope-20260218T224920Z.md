# Sampling Compatibility Envelope Report 20260218T224920Z

Settings
- Rust repo: /home/jamie/1tb/dev/rust/gemma-rs
- gemma.cpp repo: /home/jamie/1tb/dev/cpp/gemma.cpp
- SBS: /home/jamie/1tb/dev/rust/gemma-rs/gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs
- prompts: 5
- max tokens: 32
- min exact-match rate: 0.60

## Config: baseline_greedy

- temperature: 1.0
- top_k: 1
- top_p: 1.0
- seed: 0

### Prompt [MISMATCH]

Prompt:
```text
Hello
```

Rust:
```text
HelloThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThis
```

C++:
```text
Hello! How can I help you today?
```

### Prompt [MISMATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Write a one sentence summary of Rust ownership.ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThis
```

C++:
```text
Rust ownership is the ownership of the Rust compiler and its associated libraries.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2?ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThis
```

C++:
```text
2 + 2 = 4
```

### Prompt [MISMATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
List three colors.ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThis
```

C++:
```text
Here are three colors:

*   **Blue**
*   **Green**
*   **Red**
```

### Prompt [MISMATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
Complete: The quick brown foxThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThis
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 0
- mismatches: 5
- exact-match rate: 0.000

## Config: sample_t08_k8_p095_s42

- temperature: 0.8
- top_k: 8
- top_p: 0.95
- seed: 42

### Prompt [MISMATCH]

Prompt:
```text
Hello
```

Rust:
```text
HelloThisThisThisThisThisThisThisThisThisThisThisInThisThisInThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Hello! How can I help you today? 
```

### Prompt [MISMATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Write a one sentence summary of Rust ownership.ThisThisThisThisThisThisThisThisThisThisThisInThisThisInThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Rust ownership grants the owner the ability to modify and manage Rust code, including its memory layout, ownership, and security.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2?ThisThisThisThisThisThisThisThisThisThisThisInThisThisInThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
2 + 2 = 4
```

### Prompt [MISMATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
List three colors.ThisThisThisThisThisThisThisThisThisThisThisInThisThisInThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MISMATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
Complete: The quick brown foxThisThisThisThisThisThisThisThisThisThisThisInThisThisInThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 0
- mismatches: 5
- exact-match rate: 0.000

## Config: sample_t07_k16_p090_s42

- temperature: 0.7
- top_k: 16
- top_p: 0.90
- seed: 42

### Prompt [MISMATCH]

Prompt:
```text
Hello
```

Rust:
```text
HelloThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Hello! How can I help you today? 
```

### Prompt [MISMATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Write a one sentence summary of Rust ownership.ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Rust ownership refers to the ownership of the Rust compiler, library, and runtime environment.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2?ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
2 + 2 = 4
```

### Prompt [MISMATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
List three colors.ThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MISMATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
Complete: The quick brown foxThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisThisInThisThisThisThisThisThisThisThisThis
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 0
- mismatches: 5
- exact-match rate: 0.000

## Config: sample_t10_k32_p080_s123

- temperature: 1.0
- top_k: 32
- top_p: 0.80
- seed: 123

### Prompt [MISMATCH]

Prompt:
```text
Hello
```

Rust:
```text
HelloThisThisThisThisThisTheThisInThisThisThisInInInInTheThisThisThisThisThisThisThisThisInThisThisThisInThisThisThis
```

C++:
```text
Hello! How can I help you today? 
```

### Prompt [MISMATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Write a one sentence summary of Rust ownership.ThisThisThisThisThisTheThisInThisThisThisInInInInTheThisThisThisThisThisThisThisThisInThisThisThisInThisThisThis
```

C++:
```text
Rust ownership ensures the Rust compiler and runtime environment are independent and can be used by different developers.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2?ThisThisThisThisThisTheThisInThisThisThisInInInInTheThisThisThisThisThisThisThisThisInThisThisThisInThisThisThis
```

C++:
```text
2 + 2 = 4
```

### Prompt [MISMATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
List three colors.ThisThisThisThisThisTheThisInThisThisThisInInInInTheThisThisThisThisThisThisThisThisInThisThisThisInThisThisThis
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MISMATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
Complete: The quick brown foxThisThisThisThisThisTheThisInThisThisThisInInInInTheThisThisThisThisThisThisThisThisInThisThisThisInThisThisThis
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 0
- mismatches: 5
- exact-match rate: 0.000

## Config: sample_t06_k4_p100_s7

- temperature: 0.6
- top_k: 4
- top_p: 1.0
- seed: 7

### Prompt [MISMATCH]

Prompt:
```text
Hello
```

Rust:
```text
HelloThisThisThisThisThisThisThisThisInThisTheThisThisTheThisThisThisTheThisThisThisThisTheThisThisThisThisInItThisThisIn
```

C++:
```text
Hello! How can I help you today? 
```

### Prompt [MISMATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Write a one sentence summary of Rust ownership.ThisThisThisThisThisThisThisThisInThisTheThisThisTheThisThisThisTheThisThisThisThisTheThisThisThisThisInItThisThisIn
```

C++:
```text
Rust ownership allows you to manage and control your Rust code and its dependencies, ensuring security and stability.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2?ThisThisThisThisThisThisThisThisInThisTheThisThisTheThisThisThisTheThisThisThisThisTheThisThisThisThisInItThisThisIn
```

C++:
```text
2 + 2 = 4
```

### Prompt [MISMATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
List three colors.ThisThisThisThisThisThisThisThisInThisTheThisThisTheThisThisThisTheThisThisThisThisTheThisThisThisThisInItThisThisIn
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MISMATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
Complete: The quick brown foxThisThisThisThisThisThisThisThisInThisTheThisThisTheThisThisThisTheThisThisThisThisTheThisThisThisThisInItThisThisIn
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 0
- mismatches: 5
- exact-match rate: 0.000

## Overall Summary

- total: 25
- matches: 0
- mismatches: 25
- exact-match rate: 0.000
