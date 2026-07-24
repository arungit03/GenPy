# GenPy Phase 8 Evaluation Report

- Checkpoint: `/Users/macbook/Downloads/GenPy/checkpoints/fine_tuned/last_checkpoint.pt`
- Checkpoint step: 2000
- Device: `mps`
- Evaluated at: 2026-07-23T02:25:44.369957+00:00
- Validation loss: 5.479552
- Perplexity: 239.739230
- Validation tokens: 215
- Generated tokens: 40
- Aggregate generation speed: 65.932 tokens/sec

> Pass/fail uses static syntax and task-term heuristics only. Generated code is not executed.

## 1. Write bubble sort.

**Prompt**

```text
Write bubble sort.
```

**Generated answer**

```text
def __
```

- Generation time: 0.042861 seconds
- Generated tokens: 2
- Tokens/sec: 46.662
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: for, while; missing one of: bubble, swapped; Python syntax error at line 1

## 2. Write quick sort.

**Prompt**

```text
Write quick sort.
```

**Generated answer**

```text
def __
```

- Generation time: 0.044775 seconds
- Generated tokens: 2
- Tokens/sec: 44.667
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: pivot, partition; missing one of: quick_sort, quicksort; Python syntax error at line 1

## 3. Reverse a linked list.

**Prompt**

```text
Reverse a linked list.
```

**Generated answer**

```text
def __
```

- Generation time: 0.026222 seconds
- Generated tokens: 2
- Tokens/sec: 76.273
- Pass/Fail: **Fail**
- Check: missing required terms: next, return; missing one of: prev, previous; Python syntax error at line 1

## 4. Implement binary search.

**Prompt**

```text
Implement binary search.
```

**Generated answer**

```text
def __
```

- Generation time: 0.027848 seconds
- Generated tokens: 2
- Tokens/sec: 71.817
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: while, for; missing one of: mid, middle; Python syntax error at line 1

## 5. Read a CSV using pandas.

**Prompt**

```text
Read a CSV using pandas.
```

**Generated answer**

```text
def __
```

- Generation time: 0.025455 seconds
- Generated tokens: 2
- Tokens/sec: 78.569
- Pass/Fail: **Fail**
- Check: missing required terms: pandas, read_csv; Python syntax error at line 1

## 6. Explain Python decorators.

**Prompt**

```text
Explain Python decorators.
```

**Generated answer**

```text
def __
```

- Generation time: 0.016765 seconds
- Generated tokens: 2
- Tokens/sec: 119.294
- Pass/Fail: **Fail**
- Check: missing required terms: decorator, function; missing one of: wrapper, wrap; missing one of: @, syntax

## 7. Write a FastAPI CRUD API.

**Prompt**

```text
Write a FastAPI CRUD API.
```

**Generated answer**

```text
class A
```

- Generation time: 0.045914 seconds
- Generated tokens: 2
- Tokens/sec: 43.560
- Pass/Fail: **Fail**
- Check: missing required terms: fastapi, get, post, put, delete; Python syntax error at line 1

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
def f
```

- Generation time: 0.048752 seconds
- Generated tokens: 2
- Tokens/sec: 41.024
- Pass/Fail: **Fail**
- Check: missing required terms: def reverse, return, [::-1]; Python syntax error at line 1

## 9. Explain list comprehensions.

**Prompt**

```text
Explain list comprehensions.
```

**Generated answer**

```text
def __
```

- Generation time: 0.028540 seconds
- Generated tokens: 2
- Tokens/sec: 70.078
- Pass/Fail: **Fail**
- Check: missing required terms: list, for; missing one of: expression, iterable, loop; missing one of: [, bracket

## 10. Write a stack implementation.

**Prompt**

```text
Write a stack implementation.
```

**Generated answer**

```text
def f
```

- Generation time: 0.025919 seconds
- Generated tokens: 2
- Tokens/sec: 77.164
- Pass/Fail: **Fail**
- Check: missing required terms: class, push, pop; Python syntax error at line 1

## 11. Write a queue implementation.

**Prompt**

```text
Write a queue implementation.
```

**Generated answer**

```text
def f
```

- Generation time: 0.017136 seconds
- Generated tokens: 2
- Tokens/sec: 116.713
- Pass/Fail: **Fail**
- Check: missing required terms: class; missing one of: enqueue, append; missing one of: dequeue, popleft; Python syntax error at line 1

## 12. Explain generators.

**Prompt**

```text
Explain generators.
```

**Generated answer**

```text
def __
```

- Generation time: 0.034006 seconds
- Generated tokens: 2
- Tokens/sec: 58.813
- Pass/Fail: **Fail**
- Check: missing required terms: yield; missing one of: iterator, iteration; missing one of: lazy, memory

## 13. Explain async/await.

**Prompt**

```text
Explain async/await.
```

**Generated answer**

```text
def __
```

- Generation time: 0.020977 seconds
- Generated tokens: 2
- Tokens/sec: 95.342
- Pass/Fail: **Fail**
- Check: missing required terms: async, await; missing one of: coroutine, event loop, concurrent

## 14. Write a BFS algorithm.

**Prompt**

```text
Write a BFS algorithm.
```

**Generated answer**

```text
def a
```

- Generation time: 0.020349 seconds
- Generated tokens: 2
- Tokens/sec: 98.284
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: queue, deque; missing one of: bfs, breadth; Python syntax error at line 1

## 15. Write a DFS algorithm.

**Prompt**

```text
Write a DFS algorithm.
```

**Generated answer**

```text
def __
```

- Generation time: 0.020386 seconds
- Generated tokens: 2
- Tokens/sec: 98.108
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: stack, recursive, recursion; missing one of: dfs, depth; Python syntax error at line 1

## 16. Write Dijkstra's algorithm.

**Prompt**

```text
Write Dijkstra's algorithm.
```

**Generated answer**

```text
def __
```

- Generation time: 0.023984 seconds
- Generated tokens: 2
- Tokens/sec: 83.389
- Pass/Fail: **Fail**
- Check: missing required terms: distance; missing one of: heap, priority; missing one of: dijkstra, shortest; Python syntax error at line 1

## 17. Write merge sort.

**Prompt**

```text
Write merge sort.
```

**Generated answer**

```text
def __
```

- Generation time: 0.031613 seconds
- Generated tokens: 2
- Tokens/sec: 63.265
- Pass/Fail: **Fail**
- Check: missing required terms: merge, return; Python syntax error at line 1

## 18. Explain recursion.

**Prompt**

```text
Explain recursion.
```

**Generated answer**

```text
def __
```

- Generation time: 0.026422 seconds
- Generated tokens: 2
- Tokens/sec: 75.694
- Pass/Fail: **Fail**
- Check: missing required terms: function, base case; missing one of: itself, recursive call

## 19. Explain time complexity of quick sort.

**Prompt**

```text
Explain time complexity of quick sort.
```

**Generated answer**

```text
def __
```

- Generation time: 0.033251 seconds
- Generated tokens: 2
- Tokens/sec: 60.149
- Pass/Fail: **Fail**
- Check: missing required terms: o(n log n), o(n^2); missing one of: average, expected; missing one of: worst, worst-case

## 20. Convert a for loop into a list comprehension.

**Prompt**

```text
Convert a for loop into a list comprehension.
```

**Generated answer**

```text
def a
```

- Generation time: 0.045506 seconds
- Generated tokens: 2
- Tokens/sec: 43.950
- Pass/Fail: **Fail**
- Check: missing required terms: [, ], for, in
