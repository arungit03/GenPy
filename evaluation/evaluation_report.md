# GenPy Phase 8 Evaluation Report

- Checkpoint: `/Users/macbook/Downloads/GenPy/checkpoints/fine_tuned/last_checkpoint.pt`
- Checkpoint step: 2000
- Device: `mps`
- Evaluated at: 2026-07-23T02:07:22.325430+00:00
- Validation loss: 5.479552
- Perplexity: 239.739230
- Validation tokens: 215
- Generated tokens: 268
- Aggregate generation speed: 55.911 tokens/sec

> Pass/fail uses static syntax and task-term heuristics only. Generated code is not executed.

## 1. Write bubble sort.

**Prompt**

```text
Write bubble sort.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.401514 seconds
- Generated tokens: 16
- Tokens/sec: 39.849
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: for, while; missing one of: bubble, swapped; Python syntax error at line 1

## 2. Write quick sort.

**Prompt**

```text
Write quick sort.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.298577 seconds
- Generated tokens: 16
- Tokens/sec: 53.588
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: pivot, partition; missing one of: quick_sort, quicksort; Python syntax error at line 1

## 3. Reverse a linked list.

**Prompt**

```text
Reverse a linked list.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.273840 seconds
- Generated tokens: 16
- Tokens/sec: 58.428
- Pass/Fail: **Fail**
- Check: missing required terms: next, return; missing one of: prev, previous; Python syntax error at line 1

## 4. Implement binary search.

**Prompt**

```text
Implement binary search.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.278202 seconds
- Generated tokens: 16
- Tokens/sec: 57.512
- Pass/Fail: **Fail**
- Check: missing required terms: return; missing one of: while, for; missing one of: mid, middle; Python syntax error at line 1

## 5. Read a CSV using pandas.

**Prompt**

```text
Read a CSV using pandas.
```

**Generated answer**

```text
def __init__(self, other):
        pass
```

- Generation time: 0.182621 seconds
- Generated tokens: 11
- Tokens/sec: 60.234
- Pass/Fail: **Fail**
- Check: missing required terms: pandas, read_csv

## 6. Explain Python decorators.

**Prompt**

```text
Explain Python decorators.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.271035 seconds
- Generated tokens: 16
- Tokens/sec: 59.033
- Pass/Fail: **Fail**
- Check: missing required terms: decorator, function; missing one of: wrapper, wrap; missing one of: @, syntax

## 7. Write a FastAPI CRUD API.

**Prompt**

```text
Write a FastAPI CRUD API.
```

**Generated answer**

```text
class A(Base):
    __tablename__ = "id"
    id =
```

- Generation time: 0.288783 seconds
- Generated tokens: 16
- Tokens/sec: 55.405
- Pass/Fail: **Fail**
- Check: missing required terms: fastapi, get, post, put, delete; Python syntax error at line 3

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
def f(1):
    return 1
```

- Generation time: 0.233733 seconds
- Generated tokens: 9
- Tokens/sec: 38.506
- Pass/Fail: **Fail**
- Check: missing required terms: def reverse, [::-1]; Python syntax error at line 1

## 9. Explain list comprehensions.

**Prompt**

```text
Explain list comprehensions.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.273092 seconds
- Generated tokens: 16
- Tokens/sec: 58.588
- Pass/Fail: **Fail**
- Check: missing required terms: list, for; missing one of: expression, iterable, loop; missing one of: [, bracket

## 10. Write a stack implementation.

**Prompt**

```text
Write a stack implementation.
```

**Generated answer**

```text
def f():
    pass
```

- Generation time: 0.098339 seconds
- Generated tokens: 6
- Tokens/sec: 61.013
- Pass/Fail: **Fail**
- Check: missing required terms: class, push, pop

## 11. Write a queue implementation.

**Prompt**

```text
Write a queue implementation.
```

**Generated answer**

```text
def f():
    pass
```

- Generation time: 0.083443 seconds
- Generated tokens: 6
- Tokens/sec: 71.905
- Pass/Fail: **Fail**
- Check: missing required terms: class; missing one of: enqueue, append; missing one of: dequeue, popleft

## 12. Explain generators.

**Prompt**

```text
Explain generators.
```

**Generated answer**

```text
def __init__(self, other):
        self.name = self.name
```

- Generation time: 0.284525 seconds
- Generated tokens: 16
- Tokens/sec: 56.234
- Pass/Fail: **Fail**
- Check: missing required terms: yield; missing one of: iterator, iteration; missing one of: lazy, memory

## 13. Explain async/await.

**Prompt**

```text
Explain async/await.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.272412 seconds
- Generated tokens: 16
- Tokens/sec: 58.734
- Pass/Fail: **Fail**
- Check: missing required terms: async, await; missing one of: coroutine, event loop, concurrent

## 14. Write a BFS algorithm.

**Prompt**

```text
Write a BFS algorithm.
```

**Generated answer**

```text
def a():
    pass
```

- Generation time: 0.100241 seconds
- Generated tokens: 6
- Tokens/sec: 59.856
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: queue, deque; missing one of: bfs, breadth

## 15. Write a DFS algorithm.

**Prompt**

```text
Write a DFS algorithm.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.254177 seconds
- Generated tokens: 16
- Tokens/sec: 62.948
- Pass/Fail: **Fail**
- Check: missing required terms: visited; missing one of: stack, recursive, recursion; missing one of: dfs, depth; Python syntax error at line 1

## 16. Write Dijkstra's algorithm.

**Prompt**

```text
Write Dijkstra's algorithm.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.265803 seconds
- Generated tokens: 16
- Tokens/sec: 60.195
- Pass/Fail: **Fail**
- Check: missing required terms: distance; missing one of: heap, priority; missing one of: dijkstra, shortest; Python syntax error at line 1

## 17. Write merge sort.

**Prompt**

```text
Write merge sort.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.273814 seconds
- Generated tokens: 16
- Tokens/sec: 58.434
- Pass/Fail: **Fail**
- Check: missing required terms: merge, return; Python syntax error at line 1

## 18. Explain recursion.

**Prompt**

```text
Explain recursion.
```

**Generated answer**

```text
def __init__(self, other):
        return self.self.self.
```

- Generation time: 0.274228 seconds
- Generated tokens: 16
- Tokens/sec: 58.346
- Pass/Fail: **Fail**
- Check: missing required terms: function, base case; missing one of: itself, recursive call

## 19. Explain time complexity of quick sort.

**Prompt**

```text
Explain time complexity of quick sort.
```

**Generated answer**

```text
def __init__(self, self, self, self, self, self,
```

- Generation time: 0.277996 seconds
- Generated tokens: 16
- Tokens/sec: 57.555
- Pass/Fail: **Fail**
- Check: missing required terms: o(n log n), o(n^2); missing one of: average, expected; missing one of: worst, worst-case

## 20. Convert a for loop into a list comprehension.

**Prompt**

```text
Convert a for loop into a list comprehension.
```

**Generated answer**

```text
def a():
    pass
```

- Generation time: 0.106980 seconds
- Generated tokens: 6
- Tokens/sec: 56.085
- Pass/Fail: **Fail**
- Check: missing required terms: [, ], for, in
