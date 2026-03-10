/* ================================================================
   PRACTICE CONFIG — fallback pool + exclusions
   ================================================================ */

const practiceQuestionPool = [
  {
    question_id: "q001",
    question_text: "Create a 5x5 matrix with values 1,2,3,4,5 on the diagonal. Use np.diag().",
    subtopic: "Array creation",
    difficulty: 50,
    expected_output: "[[1 0 0 0 0]\n [0 2 0 0 0]\n [0 0 3 0 0]\n [0 0 0 4 0]\n [0 0 0 0 5]]",
    solution_code: "Z = np.diag(1+np.arange(5)); print(Z)",
  },
  {
    question_id: "q002",
    question_text: "Create a 8x8 matrix and fill it with a checkerboard pattern of 0s and 1s.",
    subtopic: "Indexing and selection",
    difficulty: 24,
    expected_output: "[[0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]\n [0 1 0 1 0 1 0 1]\n [1 0 1 0 1 0 1 0]]",
    solution_code:
      "Z = np.zeros((8,8),dtype=int)\nZ[1::2,::2] = 1\nZ[::2,1::2] = 1\nprint(Z)",
  },
  {
    question_id: "q003",
    question_text: "Create a 3x3 identity matrix using np.eye().",
    subtopic: "Array creation",
    difficulty: 20,
    expected_output: "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]",
    solution_code: "print(np.eye(3))",
  },
  {
    question_id: "q004",
    question_text: "Find the row-wise argmax of a 5x5 random integer matrix (seed=42, range 0-9).",
    subtopic: "Vectorization and broadcasting",
    difficulty: 24,
    expected_output: "[0 4 2 0 3]",
    solution_code: "np.random.seed(42)\nZ = np.random.randint(0,10,(5,5))\nprint(Z.argmax(axis=1))",
  },
];

let practiceQuestionIndex = 0;

const curatedExcludedIds = new Set([
  9, 20, 21, 33, 39, 44, 45, 57, 88, 161, 188, 203, 221, 222, 223, 226,
]);

const staleGaussianQuestion = (q) =>
  typeof q?.question_text === "string" &&
  q.question_text.startsWith("Generate a generic 2D Gaussian-like array");
