# Sampling Compatibility Envelope Report 20260219T001501Z

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
Hello खान खान दूसरी दूसरी दूसरी दूसरी दूसरी आपल्याला आपल्याला आपल्याला आपल्याला आपल्याला आपल्याला તો તો તો ডিগ্রী ডিগ্রী आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या
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
Write a one sentence summary of Rust ownership.ണ്‍ ডিগ্রীണ്‍ ডিগ্রী आपल्याला आपल्याला ডিগ্রী ডিগ্রী તોണ്‍ प्रॉ आपल्या आपल्या आपल्या आपल्याला आपल्याला चट्ट चट्ट चट्ट चट्ट चट्ट चट्ट प्रॉ प्रॉ दूसरी प्रॉ आपल्याला તો आपल्याला તો તો તો
```

C++:
```text
Rust ownership is the ownership of the Rust compiler and its associated libraries, which is typically managed by the Rust compiler.
```

### Prompt [MISMATCH]

Prompt:
```text
What is 2 + 2?
```

Rust:
```text
What is 2 + 2? अॅ अॅ अॅണ്‍ अॅ अॅ अॅ अॅ अॅ તો તો आपल्या आपल्या आपल्या आपल्या તો તો प्रॉ चट्ट चट्ट चट्ट चट्ट चट्ट दूसरी चट्ट दूसरी प्रॉ તો તો તો તો आपल्याला
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
List three colors. आपल्याला आपल्याला आपल्याला आपल्याला आपल्याला आपल्यालाണ്‍ आपल्यालाണ്‍ आपल्याला आपल्याला आपल्याला आपल्याला आपल्याला आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या आपल्या તો તો
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
Complete: The quick brown fox ডিগ্রীकिंगकिंगकिंग તો તો તો'।ണ്‍ ডিগ্রী તો તો તો તો તો તો તો ডিগ্রী ডিগ্রী તો ডিগ্রী તો ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী ডিগ্রী
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
Hello खान प्रॉ खान दूसरी दूसरी प्रॉिंग आपल्याला अॅ अॅ आपल्याला ডিগ্রী आपल्याला તો'। તો તો તો ডিগ্রী आपल्या आपल्या आपल्या गोष्ट आपल्याला તો पाचन पाचन आपल्या आपल्याला आपल्यालाണ്‍ണ്‍
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
Write a one sentence summary of Rust ownership.ണ്‍ आपल्याला प्रॉണ്‍ ডিগ্রী તો प्रॉ प्रॉ प्रॉ તો प्रॉ चट्ट તો प्रॉ चट्ट प्रॉ प्रॉ चट्ट चट्ट તો ডিগ্রী दूसरी आपल्याला प्रॉ प्रॉ आपल्याला आपल्याला आपल्याला તો आपल्याला તો प्रॉ
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
What is 2 + 2? अॅ'।ണ്‍ अॅचं प्रॉ खान खान आपल्याला अॅ તો अॅ प्रॉ प्रॉ चट्ट प्रॉ प्रॉ प्रॉ प्रॉ તો दूसरी आपल्याला ગ્ર आपल्याला ডিগ্রী ডিগ্রী दूसरी आपल्याला તો आपल्याला पाचन ডিগ্রী
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
List three colors. आपल्यालाണ്‍ ডিগ্রী आपल्याला आपल्यालाണ്‍ प्रॉ चट्ट ডিগ্রী તો आपल्याला चट्ट आपल्याला आपल्याला चट्ट आपल्याला आपल्याला आपल्याला आपल्या आपल्याला आपल्या તો प्रॉ ગ્ર ওজন ગ્ર તો आपल्या आपल्याला आपल्याला आपल्या ગ્ર
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
Complete: The quick brown fox ডিগ্রী दूसरी आपल्यालाണ്‍ണ്‍'। ডিগ্রী'।'।ണ്‍ണ്‍ चट्ट'।'। आपल्या તો તોണ്‍ണ്‍ണ്‍ आपल्याला ডিগ্রীണ്‍ आपल्याला'। आपल्यालाണ്‍ണ്‍ ডিগ্রীണ്‍ दूसरी ডিগ্রী
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
Hello खान সাগর प्रॉ दूसरी दूसरी ডিগ্রীिंग खान अॅ अॅ તો गोष्ट चट्ट તો चट्ट તો તો તો ডিগ্রী अव તો दूसरी अव आपल्या आपल्या आपल्याला पाचन ডিগ্রী दूसरी ডিগ্রীണ്‍ आपल्याला
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
Write a one sentence summary of Rust ownership.ണ്‍ आपल्याला प्रॉണ്‍ ডিগ্রী તો दूसरी प्रॉ चट्ट प्रॉ તો ગ્ર आपल्याला प्रॉ তো प्रॉ चट्ट आपल्याला ডিগ্রী તો ডিগ্রী दूसरीണ്‍ प्रॉ दूसरी दूसरी ডিগ্রী आपल्याला તો आपल्याला તો इमरजेंसी
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
What is 2 + 2? अॅ'।ണ്‍ अॅचं प्रॉ রান্না ডিগ্রী ডিগ্রী आपल्याला તો ডিগ্রী प्रॉ प्रॉ दूसरी आपल्याला प्रॉ प्रॉ चट्ट તો दूसरी ગ્ર प्रॉ आपल्याला ডিগ্রী ডিগ্রী खान दूसरी दूसरी आपल्याला चट्ट पाचन
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
List three colors. आपल्यालाണ്‍ ডিগ্রী आपल्याला आपल्यालाണ്‍ प्रॉ चट्ट ডিগ্রী તો आपल्यालाണ്‍ आपल्याला आपल्याला चट्ट आपल्याला आपल्याला आपल्याला आपल्या आपल्याला आपल्या તો ওজন ગ્ર पाचन चट्ट તો आपल्या आपल्या आपल्याला आपल्या ગ્ર
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
Complete: The quick brown fox ডিগ্রী दूसरी अॅണ്‍ണ്‍ તો अॅ'।'।ണ്‍ണ്‍ खान'। आपल्याला चट्ट आपल्याला તો आपल्याला તો ডিগ্রীണ്‍ണ്‍ সাগর તોണ്‍ണ്‍ ডিগ্রীണ്‍ ডিগ্রীണ്‍'। તો
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
Hello खान अॅ आपल्याला अॅिंग'। आपल्याला'। आपल्याला अॅ दूसरीাত্মক अॅ अॅ आपल्या গত आपल्या आपल्या ডিগ্রী देशा आपल्या चट्ट चट्ट प्रॉ রান্না आपल्याला गोष्ट पाचन तयार आपल्या आपल्या दूसरी
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
Write a one sentence summary of Rust ownership.ണ്‍ সাগর ডিগ্রী आपल्याला તો खान आपल्याला সাগরണ്‍ चट्ट ডিগ্রী खान पाचन दूसरी देशा खान दूसरी ডিগ্রী प्रॉ आपल्याला ডিগ্রীണ്‍ दूसरी दूसरी इमरजेंसीണ്‍ണ്‍ आपल्याला चट्ट आपल्याला आपल्याला आपल्याला
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
What is 2 + 2? अॅ તો खानचं'।चं खान चट्ट अॅ अॅ आपल्यालाकिंगकिंग दूसरीണ്‍ পণ্য आपल्या प्रॉ अॅ ગ્ર તો प्रॉ अॅ ગ્ર সাগর अॅ अॅ आपल्याला पाचन आपल्याला आपल्याला તો
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
List three colors. आपल्याला अॅ ডিগ্রী ডিগ্রীकिंग ज़ണ്‍ दूसरी आपल्याला તોണ്‍ गोष्ट আমাদের আমাদের तयार गोष्ट आपल्या आपल्या আমাদের ગ્ર आपल्याला ગ્ર ওজন प्रॉ আমাদের प्रॉ प्रॉ તો पाचनണ്‍ आपल्याला ગ્ર
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
Complete: The quick brown fox ডিগ্রীिंग खान ডিগ্রী તો गोष्ट चट्ट दूसरीണ്‍ તો ডিগ্রী चट्ट खान अॅ देश গত प्रॉ प्रॉ সাগর दूसरी તો आपल्याला प्रॉ इमरजेंसी সাগর इमरजेंसी इमरजेंसी ডিগ্রী તો ডিগ্রী दूसरी ডিগ্রী
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
Hello खान दूसरी दूसरी दूसरी दूसरी ডিগ্রী दूसरी दूसरी दूसरी आपल्याला चट्ट તો તો'। दूसरी ডিগ্রী તો તો आपल्या ডিগ্রী आपल्याला चट्ट તો आपल्याला आपल्याला पाचन आपल्याला ডিগ্রী आपल्याला ডিগ্রী ডিগ্রী दूसरी
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
Write a one sentence summary of Rust ownership.ണ്‍ ডিগ্রী ডিগ্রী आपल्याला ডিগ্রী ডিগ্রী आपल्याला आपल्याला चट्ट चट्ट चट्ट प्रॉ તો आपल्याला ગ્ર चट्ट चट्ट आपल्यालाണ്‍ ગ્ર चट्ट आपल्याला प्रॉണ്‍ आपल्याला ডিগ্রী चट्ट इमरजेंसी इमरजेंसी તો आपल्याला चट्ट
```

C++:
```text
Rust ownership allows you to manage and control your Rust code and its dependencies, ensuring code integrity and stability.
```

