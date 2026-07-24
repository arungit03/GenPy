# GenPy Phase 8 Evaluation Report

- Checkpoint: `/Users/macbook/Downloads/GenPy/checkpoints/lora/last_adapter.pt`
- Checkpoint step: 1
- Device: `mps`
- Evaluated at: 2026-07-23T02:25:49.047115+00:00
- Validation loss: 5.354034
- Perplexity: 211.459697
- Validation tokens: 215
- Generated tokens: 40
- Aggregate generation speed: 62.141 tokens/sec

> Pass/fail uses static syntax and task-term heuristics only. Generated code is not executed.

## 1. Write bubble sort.

**Prompt**

```text
Write bubble sort.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.042443 seconds
- Generated tokens: 2
- Tokens/sec: 47.122
- Pass/Fail: **Fail**
- Check: missing required terms: def, return; missing one of: for, while; missing one of: bubble, swapped; no Python code found

## 2. Write quick sort.

**Prompt**

```text
Write quick sort.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.042267 seconds
- Generated tokens: 2
- Tokens/sec: 47.318
- Pass/Fail: **Fail**
- Check: missing required terms: def, return; missing one of: pivot, partition; missing one of: quick_sort, quicksort; no Python code found

## 3. Reverse a linked list.

**Prompt**

```text
Reverse a linked list.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.020714 seconds
- Generated tokens: 2
- Tokens/sec: 96.552
- Pass/Fail: **Fail**
- Check: missing required terms: def, next, return; missing one of: prev, previous; no Python code found

## 4. Implement binary search.

**Prompt**

```text
Implement binary search.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.021739 seconds
- Generated tokens: 2
- Tokens/sec: 92.001
- Pass/Fail: **Fail**
- Check: missing required terms: def, return; missing one of: while, for; missing one of: mid, middle; no Python code found

## 5. Read a CSV using pandas.

**Prompt**

```text
Read a CSV using pandas.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.021097 seconds
- Generated tokens: 2
- Tokens/sec: 94.799
- Pass/Fail: **Fail**
- Check: missing required terms: pandas, read_csv; no Python code found

## 6. Explain Python decorators.

**Prompt**

```text
Explain Python decorators.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.021212 seconds
- Generated tokens: 2
- Tokens/sec: 94.286
- Pass/Fail: **Fail**
- Check: missing required terms: decorator, function; missing one of: wrapper, wrap; missing one of: @, syntax

## 7. Write a FastAPI CRUD API.

**Prompt**

```text
Write a FastAPI CRUD API.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.050925 seconds
- Generated tokens: 2
- Tokens/sec: 39.273
- Pass/Fail: **Fail**
- Check: missing required terms: fastapi, get, post, put, delete; no Python code found

## 8. Fix the following Python code:

**Prompt**

```text
Fix the following Python code:
def reverse(lst):
for i in range(len(lst)):
return lst[::-1]
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.057267 seconds
- Generated tokens: 2
- Tokens/sec: 34.924
- Pass/Fail: **Fail**
- Check: missing required terms: def reverse, return, [::-1]; no Python code found

## 9. Explain list comprehensions.

**Prompt**

```text
Explain list comprehensions.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.030107 seconds
- Generated tokens: 2
- Tokens/sec: 66.429
- Pass/Fail: **Fail**
- Check: missing required terms: list, for; missing one of: expression, iterable, loop; missing one of: [, bracket

## 10. Write a stack implementation.

**Prompt**

```text
Write a stack implementation.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.020345 seconds
- Generated tokens: 2
- Tokens/sec: 98.306
- Pass/Fail: **Fail**
- Check: missing required terms: class, push, pop; no Python code found

## 11. Write a queue implementation.

**Prompt**

```text
Write a queue implementation.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.020130 seconds
- Generated tokens: 2
- Tokens/sec: 99.356
- Pass/Fail: **Fail**
- Check: missing required terms: class; missing one of: enqueue, append; missing one of: dequeue, popleft; no Python code found

## 12. Explain generators.

**Prompt**

```text
Explain generators.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.035537 seconds
- Generated tokens: 2
- Tokens/sec: 56.280
- Pass/Fail: **Fail**
- Check: missing required terms: yield; missing one of: iterator, iteration; missing one of: lazy, memory

## 13. Explain async/await.

**Prompt**

```text
Explain async/await.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.025735 seconds
- Generated tokens: 2
- Tokens/sec: 77.715
- Pass/Fail: **Fail**
- Check: missing required terms: async, await; missing one of: coroutine, event loop, concurrent

## 14. Write a BFS algorithm.

**Prompt**

```text
Write a BFS algorithm.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.026345 seconds
- Generated tokens: 2
- Tokens/sec: 75.914
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: queue, deque; missing one of: bfs, breadth; no Python code found

## 15. Write a DFS algorithm.

**Prompt**

```text
Write a DFS algorithm.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.025666 seconds
- Generated tokens: 2
- Tokens/sec: 77.923
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: stack, recursive, recursion; missing one of: dfs, depth; no Python code found

## 16. Write Dijkstra's algorithm.

**Prompt**

```text
Write Dijkstra's algorithm.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.024465 seconds
- Generated tokens: 2
- Tokens/sec: 81.751
- Pass/Fail: **Fail**
- Check: missing required terms: distance; missing one of: heap, priority; missing one of: dijkstra, shortest; no Python code found

## 17. Write merge sort.

**Prompt**

```text
Write merge sort.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.034654 seconds
- Generated tokens: 2
- Tokens/sec: 57.713
- Pass/Fail: **Fail**
- Check: missing required terms: def, merge, return; no Python code found

## 18. Explain recursion.

**Prompt**

```text
Explain recursion.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.032950 seconds
- Generated tokens: 2
- Tokens/sec: 60.698
- Pass/Fail: **Fail**
- Check: missing required terms: function, base case; missing one of: itself, recursive call

## 19. Explain time complexity of quick sort.

**Prompt**

```text
Explain time complexity of quick sort.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.036395 seconds
- Generated tokens: 2
- Tokens/sec: 54.952
- Pass/Fail: **Fail**
- Check: missing required terms: o(n log n), o(n^2); missing one of: average, expected; missing one of: worst, worst-case

## 20. Convert a for loop into a list comprehension.

**Prompt**

```text
Convert a for loop into a list comprehension.
```

**Generated answer**

```text
(empty output)
```

- Generation time: 0.053698 seconds
- Generated tokens: 2
- Tokens/sec: 37.245
- Pass/Fail: **Fail**
- Check: missing required terms: [, ], for, in
