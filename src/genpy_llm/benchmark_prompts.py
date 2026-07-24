"""Built-in prompt and question datasets for the GenPy benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PythonPrompt:
    """One Python-coding benchmark prompt."""

    id: str
    category: str
    prompt: str


@dataclass(frozen=True)
class DocumentationQuestion:
    """One documentation question scored by expected-keyword coverage."""

    id: str
    source: str
    question: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class TextGenerationTask:
    """One instruction-following/formatting task."""

    id: str
    instruction: str
    required_terms: tuple[str, ...]
    wants_markdown: bool
    wants_code_fence: bool


def _python(category: str, prompts: tuple[str, ...]) -> tuple[PythonPrompt, ...]:
    return tuple(
        PythonPrompt(id=f"{category}_{index:02d}", category=category, prompt=prompt)
        for index, prompt in enumerate(prompts, start=1)
    )


PYTHON_BENCHMARK_PROMPTS: tuple[PythonPrompt, ...] = (
    *_python(
        "algorithms",
        (
            "Write a Python function binary_search(items, target) that returns the index "
            "of target in a sorted list or -1.",
            "Write a Python function fibonacci(n) that returns the nth Fibonacci number "
            "iteratively.",
            "Write a Python function quicksort(values) that sorts a list of integers.",
            "Write a Python function is_prime(number) that returns True for prime numbers.",
            "Write a Python function gcd(a, b) using Euclid's algorithm.",
        ),
    ),
    *_python(
        "data_structures",
        (
            "Implement a Python Stack class with push, pop, and peek methods.",
            "Implement a Python Queue class using collections.deque.",
            "Implement a singly linked list Node class and an append function in Python.",
            "Write a Python function to invert a dictionary mapping keys to values.",
            "Implement a min-heap push operation in Python using the heapq module.",
        ),
    ),
    *_python(
        "oop",
        (
            "Define a Python class Rectangle with width, height, and an area method.",
            "Define a Python class Animal and a subclass Dog overriding a speak method.",
            "Define a Python dataclass Point with x and y fields and a distance method.",
            "Define a Python class Counter with a classmethod from_list and a property total.",
            "Define a Python abstract base class Shape with an abstract area method.",
        ),
    ),
    *_python(
        "typing",
        (
            "Write a typed Python function head(items: list[int]) -> int | None.",
            "Define a Python TypedDict called UserRecord with name and age fields.",
            "Write a generic Python function first(items) using TypeVar annotations.",
            "Annotate a Python function that accepts a callback of type Callable[[int], str].",
            "Define a Python Protocol named Sized with a __len__ method.",
        ),
    ),
    *_python(
        "decorators",
        (
            "Write a Python decorator timed(func) that prints how long a call takes.",
            "Write a Python decorator retry(times) that retries a failing function.",
            "Write a Python decorator memoize(func) that caches results in a dict.",
            "Use functools.wraps in a logging decorator for a Python function.",
            "Write a class-based Python decorator CountCalls that counts invocations.",
        ),
    ),
    *_python(
        "generators",
        (
            "Write a Python generator countdown(n) that yields n down to 1.",
            "Write a Python generator that yields lines of a file lazily.",
            "Write a Python generator expression that yields squares of even numbers.",
            "Write a Python generator chunks(items, size) yielding fixed-size chunks.",
            "Write a Python coroutine-style generator that receives values via send().",
        ),
    ),
    *_python(
        "asyncio",
        (
            "Write an async Python function fetch_all(urls) using asyncio.gather.",
            "Write a Python asyncio example that runs two coroutines concurrently.",
            "Write an async Python context manager using __aenter__ and __aexit__.",
            "Write a Python asyncio producer/consumer example using asyncio.Queue.",
            "Write an async Python function that times out using asyncio.wait_for.",
        ),
    ),
    *_python(
        "fastapi",
        (
            "Write a FastAPI app with a GET /health endpoint returning JSON status.",
            "Write a FastAPI POST /items endpoint using a Pydantic model.",
            "Write a FastAPI path parameter endpoint GET /users/{user_id}.",
            "Add a query parameter with a default value to a FastAPI endpoint.",
            "Write a FastAPI dependency with Depends that provides a database session.",
        ),
    ),
    *_python(
        "flask",
        (
            "Write a minimal Flask app with a route / returning Hello World.",
            "Write a Flask route that accepts POST JSON and echoes it back.",
            "Write a Flask route with a URL parameter /users/<int:user_id>.",
            "Use Flask's render_template to serve an HTML page.",
            "Add an error handler for 404 responses in a Flask app.",
        ),
    ),
    *_python(
        "django",
        (
            "Define a Django model Article with title and published_date fields.",
            "Write a Django view that returns JsonResponse with a list of items.",
            "Write a Django URL pattern routing /articles/ to a view.",
            "Write a Django ModelForm for an Article model.",
            "Write a Django queryset filtering Article objects by published year.",
        ),
    ),
    *_python(
        "numpy",
        (
            "Create a NumPy array of zeros with shape (3, 4) and print its dtype.",
            "Compute the mean and standard deviation of a NumPy array.",
            "Reshape a NumPy arange of 12 elements into a 3x4 matrix.",
            "Multiply two NumPy matrices using the @ operator.",
            "Use NumPy boolean masking to select values greater than 5.",
        ),
    ),
    *_python(
        "pandas",
        (
            "Create a pandas DataFrame from a dict and print the first rows.",
            "Filter pandas DataFrame rows where a column value exceeds a threshold.",
            "Group a pandas DataFrame by a column and compute the mean.",
            "Read a CSV file into pandas and select two columns.",
            "Add a new computed column to a pandas DataFrame.",
        ),
    ),
    *_python(
        "pytorch",
        (
            "Create a PyTorch tensor of ones and move it to the available device.",
            "Define a PyTorch nn.Module with one linear layer and a forward method.",
            "Write a PyTorch training step: forward, loss, backward, optimizer step.",
            "Compute gradients of y = x**2 in PyTorch using autograd.",
            "Save and load a PyTorch model state_dict.",
        ),
    ),
    *_python(
        "regex",
        (
            "Write a Python regex to extract all email addresses from text.",
            "Write a Python regex that validates a date in YYYY-MM-DD format.",
            "Use re.sub in Python to collapse repeated whitespace into one space.",
            "Write a Python regex with named groups to parse 'key=value' pairs.",
            "Use re.findall in Python to extract integers from a string.",
        ),
    ),
    *_python(
        "file_handling",
        (
            "Write Python code that reads a text file and counts its lines.",
            "Write Python code that writes a list of strings to a file, one per line.",
            "Use pathlib in Python to list all .py files in a directory tree.",
            "Copy a file in Python using shutil and verify it exists.",
            "Append a line to a log file in Python using a context manager.",
        ),
    ),
    *_python(
        "cli",
        (
            "Write a Python argparse CLI with a required input path and a --verbose flag.",
            "Write a Python CLI that reads stdin and prints line numbers.",
            "Add a subcommand 'build' with its own arguments using argparse.",
            "Parse environment variables with os.environ in a Python CLI tool.",
            "Print a usage message and exit with status 2 on bad CLI arguments.",
        ),
    ),
    *_python(
        "testing",
        (
            "Write a pytest test for a function add(a, b) including an edge case.",
            "Write a pytest fixture that creates a temporary directory.",
            "Use pytest.raises to assert a ValueError is raised.",
            "Parametrize a pytest test over several input/output pairs.",
            "Write a unittest TestCase with setUp and one assertion.",
        ),
    ),
    *_python(
        "logging",
        (
            "Configure Python logging with a level and message format.",
            "Create a named logger in Python and log an info and an error message.",
            "Write Python logging to a file with a FileHandler.",
            "Use logging.exception inside an except block in Python.",
            "Add a rotating file handler using logging.handlers in Python.",
        ),
    ),
    *_python(
        "json",
        (
            "Parse a JSON string in Python and access a nested field.",
            "Serialize a Python dict to pretty-printed JSON.",
            "Read a JSON file in Python and handle a missing-file error.",
            "Write JSON lines (one object per line) to a file in Python.",
            "Convert a Python dataclass to JSON using dataclasses.asdict.",
        ),
    ),
    *_python(
        "csv",
        (
            "Read a CSV file in Python with csv.DictReader and print one column.",
            "Write a list of dicts to a CSV file with csv.DictWriter.",
            "Filter CSV rows by a column value in Python.",
            "Handle a custom delimiter when reading CSV in Python.",
            "Sum a numeric CSV column in Python.",
        ),
    ),
    *_python(
        "sql",
        (
            "Use sqlite3 in Python to create a table and insert one row.",
            "Write a parameterized SELECT query with sqlite3 in Python.",
            "Iterate over sqlite3 query results and print each row in Python.",
            "Use a sqlite3 connection as a context manager in Python.",
            "Create an index on a sqlite3 table column from Python.",
        ),
    ),
)


def _doc(
    source: str,
    items: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[DocumentationQuestion, ...]:
    return tuple(
        DocumentationQuestion(
            id=f"{source}_{index:02d}",
            source=source,
            question=question,
            keywords=keywords,
        )
        for index, (question, keywords) in enumerate(items, start=1)
    )


DOCUMENTATION_QA: tuple[DocumentationQuestion, ...] = (
    *_doc(
        "python_docs",
        (
            ("What does the len() builtin return?", ("length", "items")),
            ("What is a Python list comprehension?", ("list", "expression", "for")),
            ("What does the with statement do in Python?", ("context", "manager")),
            ("How do you open a file for reading in Python?", ("open", "read")),
            ("What is the difference between a list and a tuple?", ("mutable", "immutable")),
            ("What does the dict.get method do?", ("key", "default")),
            ("What is a Python virtual environment?", ("packages", "isolated")),
            ("What does the enumerate builtin do?", ("index", "iterable")),
            ("What is the purpose of __init__ in a class?", ("initialize", "instance")),
            ("What does the yield keyword do?", ("generator", "value")),
            ("How does try/except work in Python?", ("exception", "handle")),
            ("What is the purpose of the self parameter?", ("instance", "method")),
            ("What does the zip builtin do?", ("iterables", "pairs")),
            ("What is a lambda expression?", ("anonymous", "function")),
            ("What does the range function return?", ("sequence", "numbers")),
        ),
    ),
    *_doc(
        "peps",
        (
            ("What is PEP 8?", ("style", "guide")),
            ("What naming style does PEP 8 recommend for functions?", ("lowercase", "underscores")),
            ("What is PEP 20 also known as?", ("zen", "python")),
            ("What does PEP 484 introduce?", ("type", "hints")),
            ("What indentation does PEP 8 recommend?", ("4", "spaces")),
            ("What does PEP 257 cover?", ("docstring", "conventions")),
            ("What did PEP 572 add to Python?", ("walrus", "assignment")),
            ("What does PEP 517 define?", ("build", "backend")),
            ("What line length does PEP 8 suggest?", ("79", "characters")),
            ("What did PEP 585 change about typing?", ("generic", "builtin")),
            ("What is the purpose of a PEP?", ("proposal", "python")),
            ("What did PEP 634 introduce?", ("pattern", "matching")),
            ("What does PEP 498 describe?", ("f-string", "literal")),
            ("What did PEP 3107 introduce?", ("function", "annotations")),
            ("What does PEP 440 standardize?", ("version", "identifiers")),
        ),
    ),
    *_doc(
        "fastapi",
        (
            ("How do you define a GET endpoint in FastAPI?", ("app.get", "decorator")),
            ("What library does FastAPI use for validation?", ("pydantic", "model")),
            ("How does FastAPI generate API docs?", ("openapi", "swagger")),
            ("How do you declare a path parameter in FastAPI?", ("path", "braces")),
            ("How do you declare a query parameter in FastAPI?", ("function", "parameter")),
            ("What is Depends used for in FastAPI?", ("dependency", "injection")),
            ("How do you return a custom status code in FastAPI?", ("status_code", "response")),
            ("What server is commonly used to run FastAPI?", ("uvicorn", "asgi")),
            ("How do you read a request body in FastAPI?", ("pydantic", "body")),
            ("How do you add CORS to FastAPI?", ("middleware", "cors")),
            ("What Python feature powers FastAPI type checking?", ("type", "hints")),
            ("How do you handle form data in FastAPI?", ("form", "multipart")),
            ("How do you serve static files in FastAPI?", ("staticfiles", "mount")),
            ("How do you write an async endpoint in FastAPI?", ("async", "def")),
            ("How do you raise an HTTP error in FastAPI?", ("httpexception", "raise")),
        ),
    ),
    *_doc(
        "numpy",
        (
            ("What is a NumPy ndarray?", ("array", "dimensional")),
            ("What does numpy.arange do?", ("evenly", "values")),
            ("What does the shape attribute describe?", ("dimensions", "array")),
            ("What is NumPy broadcasting?", ("shapes", "arithmetic")),
            ("What does numpy.zeros create?", ("array", "zeros")),
            ("How do you compute a mean in NumPy?", ("mean", "axis")),
            ("What does numpy.reshape do?", ("shape", "data")),
            ("What is the dtype attribute?", ("data", "type")),
            ("What does numpy.dot compute?", ("product", "matrix")),
            ("How do you select elements with a condition?", ("boolean", "mask")),
            ("What does numpy.linspace return?", ("evenly", "interval")),
            ("What does numpy.concatenate do?", ("join", "arrays")),
            ("What is a NumPy view versus a copy?", ("memory", "data")),
            ("What does numpy.random.rand produce?", ("random", "uniform")),
            ("What does numpy.transpose do?", ("axes", "swap")),
        ),
    ),
    *_doc(
        "pandas",
        (
            ("What is a pandas DataFrame?", ("tabular", "columns")),
            ("What is a pandas Series?", ("one-dimensional", "labeled")),
            ("What does read_csv do?", ("csv", "dataframe")),
            ("What does DataFrame.head return?", ("first", "rows")),
            ("How do you select a column in pandas?", ("bracket", "name")),
            ("What does groupby do in pandas?", ("split", "aggregate")),
            ("What does DataFrame.describe show?", ("statistics", "summary")),
            ("How do you handle missing values in pandas?", ("fillna", "dropna")),
            ("What does DataFrame.merge do?", ("join", "keys")),
            ("What is the pandas index?", ("labels", "rows")),
            ("What does DataFrame.sort_values do?", ("sort", "column")),
            ("What does DataFrame.loc select?", ("label", "rows")),
            ("What does DataFrame.iloc select?", ("integer", "position")),
            ("How do you apply a function to a column?", ("apply", "function")),
            ("What does to_csv do?", ("write", "file")),
        ),
    ),
    *_doc(
        "django",
        (
            ("What is a Django model?", ("database", "table")),
            ("What is a Django view?", ("request", "response")),
            ("What does urls.py define?", ("routes", "patterns")),
            ("What is the Django ORM?", ("queries", "objects")),
            ("What does manage.py migrate do?", ("database", "schema")),
            ("What is a Django template?", ("html", "context")),
            ("What is a QuerySet?", ("database", "lazy")),
            ("What does Django admin provide?", ("interface", "manage")),
            ("What is a Django ModelForm?", ("form", "model")),
            ("What does settings.py contain?", ("configuration", "project")),
            ("What is Django middleware?", ("request", "response")),
            ("What does makemigrations do?", ("changes", "migrations")),
            ("How does Django handle static files?", ("static", "collectstatic")),
            ("What is the purpose of Django signals?", ("events", "receivers")),
            ("What does get_object_or_404 do?", ("404", "object")),
        ),
    ),
    *_doc(
        "flask",
        (
            ("What is Flask?", ("micro", "framework")),
            ("What does the route decorator do?", ("url", "function")),
            ("What is the Flask application object?", ("flask", "instance")),
            ("How do you access request data in Flask?", ("request", "object")),
            ("What does render_template do?", ("html", "jinja")),
            ("What is Flask's debug mode?", ("reload", "errors")),
            ("How do you return JSON from Flask?", ("jsonify", "response")),
            ("What are Flask blueprints?", ("organize", "routes")),
            ("How do you handle a 404 in Flask?", ("errorhandler", "404")),
            ("What is the Flask session object?", ("cookies", "data")),
            ("How do you get URL parameters in Flask?", ("args", "request")),
            ("What template engine does Flask use?", ("jinja", "templates")),
            ("How do you redirect in Flask?", ("redirect", "url_for")),
            ("How do you run a Flask development server?", ("run", "app")),
            ("How do you read posted form fields in Flask?", ("form", "request")),
        ),
    ),
)


TEXT_GENERATION_TASKS: tuple[TextGenerationTask, ...] = (
    TextGenerationTask(
        id="format_markdown_list",
        instruction="List three Python data types as a Markdown bullet list.",
        required_terms=("-",),
        wants_markdown=True,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="format_code_fence",
        instruction="Show a hello world Python program inside a Markdown code fence.",
        required_terms=("print",),
        wants_markdown=True,
        wants_code_fence=True,
    ),
    TextGenerationTask(
        id="explain_function",
        instruction="Explain in one paragraph what a Python function is.",
        required_terms=("function",),
        wants_markdown=False,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="write_docstring",
        instruction="Write a Python function with a docstring that adds two numbers.",
        required_terms=("def", '"""'),
        wants_markdown=False,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="markdown_heading",
        instruction="Write a short Markdown section with a heading about Python lists.",
        required_terms=("#",),
        wants_markdown=True,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="steps_list",
        instruction="Describe the steps to install a Python package as a numbered list.",
        required_terms=("pip",),
        wants_markdown=True,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="fenced_example",
        instruction="Give an example of a for loop in Python using a Markdown code fence.",
        required_terms=("for",),
        wants_markdown=True,
        wants_code_fence=True,
    ),
    TextGenerationTask(
        id="compare_concepts",
        instruction="Compare lists and tuples in Python in two sentences.",
        required_terms=("list", "tuple"),
        wants_markdown=False,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="class_example",
        instruction="Define a small Python class Dog with a bark method.",
        required_terms=("class", "def"),
        wants_markdown=False,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="error_explanation",
        instruction="Explain what a Python TypeError is in one sentence.",
        required_terms=("type",),
        wants_markdown=False,
        wants_code_fence=False,
    ),
    TextGenerationTask(
        id="fenced_json",
        instruction="Show a small JSON object inside a Markdown code fence.",
        required_terms=("{",),
        wants_markdown=True,
        wants_code_fence=True,
    ),
    TextGenerationTask(
        id="summarize_topic",
        instruction="Summarize what the Python standard library offers in two sentences.",
        required_terms=("library",),
        wants_markdown=False,
        wants_code_fence=False,
    ),
)


__all__ = [
    "DOCUMENTATION_QA",
    "DocumentationQuestion",
    "PYTHON_BENCHMARK_PROMPTS",
    "PythonPrompt",
    "TEXT_GENERATION_TASKS",
    "TextGenerationTask",
]
