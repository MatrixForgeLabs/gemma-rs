# Sampling Compatibility Envelope Report 20260219T020134Z

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

### Prompt [MATCH]

Prompt:
```text
Hello
```

Rust:
```text
Hello! How can I help you today?
```

C++:
```text
Hello! How can I help you today?
```

### Prompt [MATCH]

Prompt:
```text
Write a one sentence summary of Rust ownership.
```

Rust:
```text
Rust ownership is the ownership of the Rust compiler and its associated libraries.
```

C++:
```text
Rust ownership is the ownership of the Rust compiler and its associated libraries.
```

### Prompt [MATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
2 + 2 = 4
```

C++:
```text
2 + 2 = 4
```

### Prompt [MATCH]

Prompt:
```text
List three colors.
```

Rust:
```text
Here are three colors:

*   **Blue**
*   **Green**
*   **Red**
```

C++:
```text
Here are three colors:

*   **Blue**
*   **Green**
*   **Red**
```

### Prompt [MATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
The quick brown fox
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 5
- mismatches: 0
- exact-match rate: 1.000

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
Hello! How can I help you today? 😊
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
Rust ownership allows you to modify the ownership of your Rust code, enabling you to modify the program's behavior without risking your own data or security.
```

C++:
```text
Rust ownership grants the owner the ability to modify and manage Rust code, including its memory layout, ownership, and security.
```

### Prompt [MATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
2 + 2 = 4
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
Here are three colors:

*   **Red**
*   **Blue**
*   **Green**
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
The quick brown fox
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 2
- mismatches: 3
- exact-match rate: 0.400

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
Hello! How can I help you today? 😊
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
Rust ownership allows you to modify the ownership of your Rust code, enabling you to modify the program's behavior without forgetting to own the original code.
```

C++:
```text
Rust ownership refers to the ownership of the Rust compiler, library, and runtime environment.
```

### Prompt [MATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
2 + 2 = 4
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
Here are three colors:

*   **Red**
*   **Blue**
*   **Green**
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
The quick brown fox
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 2
- mismatches: 3
- exact-match rate: 0.400

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
Hello! How can I help you today? 😊
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
Rust ownership grants ownership to libraries, functions, and data structures, allowing developers to control and modify the code they write.
```

C++:
```text
Rust ownership ensures the Rust compiler and runtime environment are independent and can be used by different systems.
```

### Prompt [MATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
2 + 2 = 4
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
Here are three colors:

*   **Red**
*   **Blue**
*   **Green**
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
The quick brown fox
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 2
- mismatches: 3
- exact-match rate: 0.400

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
Hello! How can I help you today? 😊
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
Rust ownership is the ability to own and manage Rust code, enabling developers to use and extend Rust features.
```

C++:
```text
Rust ownership allows you to manage and control your Rust code and its dependencies, ensuring stability, security, and maintainability.
```

### Prompt [MATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
2 + 2 = 4
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
Here are three colors:

*   **Red**
*   **Blue**
*   **Green**
```

C++:
```text
Here are three colors:

*   Red
*   Blue
*   Green
```

### Prompt [MATCH]

Prompt:
```text
Complete: The quick brown fox
```

Rust:
```text
The quick brown fox
```

C++:
```text
The quick brown fox
```

Config summary
- total: 5
- matches: 2
- mismatches: 3
- exact-match rate: 0.400

## Overall Summary

- total: 25
- matches: 13
- mismatches: 12
- exact-match rate: 0.520
