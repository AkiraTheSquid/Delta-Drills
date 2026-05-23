/* ================================================================
   PRACTICE CONFIG — fallback pool + exclusions
   ================================================================ */

// Fallback pool used when the Pyodide engine + question bank are unavailable.
// Every entry must match the schema the UI consumes: question_text, starter_code
// (function stub WITHOUT the answer leaked above it), test_cases, submission_mode,
// function_name, topic, primary_library, task_type. Setup that defines the
// expected value belongs in test_cases[*].setup_code, NOT in starter_code.
const practiceQuestionPool = [
  {
    question_id: "q001",
    topic: "Numpy",
    subtopic: "Array creation",
    difficulty: 50,
    primary_library: "numpy",
    task_type: "stdout_prediction",
    function_name: "solve",
    submission_mode: "function",
    expected_artifact_type: "stdout",
    question_text: "Create a 5x5 matrix with values 1, 2, 3, 4, 5 on the main diagonal (and zeros elsewhere). Use `np.diag`.",
    expected_output: "[[1 0 0 0 0]\n [0 2 0 0 0]\n [0 0 3 0 0]\n [0 0 0 4 0]\n [0 0 0 0 5]]",
    solution_code: "Z = np.diag(1 + np.arange(5))\nprint(Z)",
    starter_code:
      "import numpy as np\n\ndef solve():\n    \"\"\"Return a 5x5 int array with [1, 2, 3, 4, 5] on the main diagonal.\"\"\"\n    raise NotImplementedError()\n\n\nprint(solve())\n",
    test_cases: [
      {
        setup_code: "Z = np.diag(1 + np.arange(5))",
        call: "solve()",
        expected_expr: "Z",
      },
    ],
  },
  {
    question_id: "q002",
    topic: "Numpy",
    subtopic: "Indexing and selection",
    difficulty: 24,
    primary_library: "numpy",
    task_type: "stdout_prediction",
    function_name: "solve",
    submission_mode: "function",
    expected_artifact_type: "stdout",
    question_text: "Create an 8x8 integer matrix filled with a checkerboard pattern of 0s and 1s. The top-left corner should be 0.",
    expected_output: "[[0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]]",
    solution_code:
      "Z = np.zeros((8, 8), dtype=int)\nZ[1::2, ::2] = 1\nZ[::2, 1::2] = 1\nprint(Z)",
    starter_code:
      "import numpy as np\n\ndef solve():\n    \"\"\"Return an 8x8 int ndarray with a checkerboard of 0s and 1s (top-left is 0).\"\"\"\n    raise NotImplementedError()\n\n\nprint(solve())\n",
    test_cases: [
      {
        setup_code:
          "Z = np.zeros((8, 8), dtype=int)\nZ[1::2, ::2] = 1\nZ[::2, 1::2] = 1",
        call: "solve()",
        expected_expr: "Z",
      },
    ],
  },
  {
    question_id: "q003",
    topic: "Numpy",
    subtopic: "Array creation",
    difficulty: 20,
    primary_library: "numpy",
    task_type: "stdout_prediction",
    function_name: "solve",
    submission_mode: "function",
    expected_artifact_type: "stdout",
    question_text: "Return a 3x3 identity matrix using `np.eye`.",
    expected_output: "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]",
    solution_code: "print(np.eye(3))",
    starter_code:
      "import numpy as np\n\ndef solve():\n    \"\"\"Return a 3x3 identity matrix as a float ndarray.\"\"\"\n    raise NotImplementedError()\n\n\nprint(solve())\n",
    test_cases: [
      {
        setup_code: "Z = np.eye(3)",
        call: "solve()",
        expected_expr: "Z",
      },
    ],
  },
  {
    question_id: "q004",
    topic: "Numpy",
    subtopic: "Vectorization and broadcasting",
    difficulty: 24,
    primary_library: "numpy",
    task_type: "stdout_prediction",
    function_name: "solve",
    submission_mode: "function",
    expected_artifact_type: "stdout",
    question_text: "Given a 5x5 matrix of random integers in [0, 10) generated with `np.random.seed(42)` and `np.random.randint(0, 10, (5, 5))`, return the row-wise argmax (a length-5 array of column indices).",
    expected_output: "[0 4 2 0 3]",
    solution_code:
      "np.random.seed(42)\nZ = np.random.randint(0, 10, (5, 5))\nprint(Z.argmax(axis=1))",
    starter_code:
      "import numpy as np\n\ndef solve():\n    \"\"\"Build the matrix described in the prompt and return its row-wise argmax.\"\"\"\n    raise NotImplementedError()\n\n\nprint(solve())\n",
    test_cases: [
      {
        setup_code:
          "np.random.seed(42)\n_Z_q004 = np.random.randint(0, 10, (5, 5))\nexpected = _Z_q004.argmax(axis=1)",
        call: "solve()",
        expected_expr: "expected",
      },
    ],
  },
];

let practiceQuestionIndex = 0;

const curatedExcludedIds = new Set([
  9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226,
]);

const staleGaussianQuestion = (q) =>
  typeof q?.question_text === "string" &&
  q.question_text.startsWith("Generate a generic 2D Gaussian-like array");
