# GenPy Model Quality Diagnostic

Created: 2026-07-23T09:38:53.344316+00:00

## Conclusion

Primary issue: **A. pretraining**

The strongest supported diagnosis is A. pretraining. This is based only on collected dataset, checkpoint, and benchmark evidence.

Evidence:
- Pretraining corpus is small for an LLM: 10,777,409 tokens.
- Checkpoint validation loss is high: 3.7122 (perplexity 40.94).
- Instruction duplicate-output rate is 32.59%.
- Instruction template inserts exactly one system/user/assistant marker per record.
- Benchmark repetition loops: 1 / 7 prompts.
- Model is small: 35,823,616 parameters.

## Pretraining Dataset

- Total tokens: 10,777,409
- Vocabulary coverage: 91.77% (29,366 / 32,000 token IDs used)
- Duplicate rate: 1.50% (84 duplicates / 5,616 considered)
- Python language token percentage: 100.00%
- Python code percentage: 81.61%
- Natural-language percentage: 18.39%
- Source mix method: Character-weighted over pretraining source files: comments and AST docstrings count as natural language; remaining non-whitespace Python source counts as code.

## Instruction Dataset

- Examples: 56,947
- Average prompt length: 131.36 tokens
- Average response length: 112.94 tokens
- Conversation template: `<|system|>\n{system_prompt}\n\n<|user|>\n{instruction}[\n\n{input}]\n\n<|assistant|>\n{output}`
- Repeated marker records: {}
- Duplicate output rate: 32.59%

## Checkpoint

- Path: `/Users/macbook/Downloads/GenPy/checkpoints/quantized/last_checkpoint_fp16.pt`
- Kind: quantized
- Parameter count: 35,823,616
- Training steps: 2,000
- Final validation loss: 3.7122
- Perplexity: 40.9442
- Optimizer state: {'available': True, 'param_groups': 2, 'state_entries': 76, 'tensor_count': 228, 'tensor_bytes': 286589232}

## Benchmarks

- Device: mps
- Quantization: fp16
- Settings: `{'prompts': (), 'max_new_tokens': 256, 'temperature': 0.7, 'top_k': None, 'top_p': 0.95, 'do_sample': True, 'repetition_penalty': 1.0, 'stop_tokens': ('<eos>',)}`
- Issue counts: {'invalid_python_syntax': 5, 'repetition_loop': 1}

### Hello

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/01_hello.json`
- Tokens generated: 25
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: True
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
def __init__(self, self, x):
        self.from_url.user_name = self.name
```

### 1 + 1 =

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/02_1_1.json`
- Tokens generated: 64
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: True
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
def a():
    with pytest.raises(ValidationError) as exc_raises(ValidationError) as exc_message(ValidationError) as exc_raises(ValidationError) as exc.value:
        exc_message(
            exc_value.value) as exc_value.value:
            return exc_value.value
```

### Write bubble sort

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/03_write_bubble_sort.json`
- Tokens generated: 26
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: True
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
def f():
    print(f"Hello {f"Hello {f" f"Hello {line r"docs }
```

### Reverse a linked list

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/04_reverse_a_linked_list.json`
- Tokens generated: 12
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: False
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
def f():
    print(f"-0")
```

### Explain Python decorators

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/05_explain_python_decorators.json`
- Tokens generated: 17
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: False
- Possible undefined identifiers: ['Base', 'Mapped']
- Unfinished functions: False

Response:

```text
class B(Base):
    id: Mapped[str] = "q"
```

### Fibonacci

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/06_fibonacci.json`
- Tokens generated: 16
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: False
- Invalid Python syntax: True
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
class A(Base):
    __init__(self.TestBase):
        pass
```

### Prime numbers

- Response file: `/Users/macbook/Downloads/GenPy/reports/model_quality/20260723T093810Z/responses/07_prime_numbers.json`
- Tokens generated: 130
- Stopped: True
- Prompt echoing: False
- Repeated markers: False {'<|system|>': 0, '<|user|>': 0, '<|assistant|>': 0}
- Repetition loop: True
- Invalid Python syntax: True
- Possible undefined identifiers: []
- Unfinished functions: False

Response:

```text
def __init__(self, self, cls, self, self, self,self, self, self, self.classes.classes.classes.classes.classes.User, self.User, self.tables.User, self.classes.User, self.classes.classes.classes.mapper_imperatively(self.map_map_registry.tables.classes.classes.User,
            self.mapper_registry.mapper_imperatively(User, self.mapper_imperatively(User,
        )
        sess = fixture_session()
        User, backref="o, addresses.id": relationship(User).all(),
        )
```
