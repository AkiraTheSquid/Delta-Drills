# LeetCode Patterns — concept notes

One section per concept in `patterns.json`, headed by its id. Each is the text a
learner reads when they click the concept on the Knowledge Graph: what the
pattern is, the signals that call for it, and the skeleton to start from.
`scripts/leetcode_course/export_concepts.py` turns this file into
`Local_Deployed_Shared/lessons/leetcode/concepts.json`.

## leetcode.arrays-strings
Most problems start here: walk an array or string once, keep a little state (a counter, a running best, a write index) and update it as you go. There is no named trick. The work is in getting the indices right: off-by-one bounds, reading `i - 1` or `i + 1` safely, and deciding whether to build a new result or overwrite the input in place.

**Reach for it when**
- The statement describes a simulation ("apply these rules to each element").
- One pass with a few variables is enough and nothing needs sorting or lookup.
- You're asked to modify the array in place with O(1) extra space.

```python
def solve(nums):
    best = cur = 0
    for i, x in enumerate(nums):
        cur = cur + 1 if x == 1 else 0   # update the running state
        best = max(best, cur)            # fold it into the answer
    return best
```

Typical cost: O(n) time, O(1) extra space.

## leetcode.hash-maps
A dict or set answers "have I seen this?" or "how many times?" in O(1). That turns a nested loop that compares every pair into a single pass. Store what you need to look up later, keyed by the value you'll search for: seen values, their indices, their counts, or a normalised key such as sorted letters.

**Reach for it when**
- You need a complement (two-sum: "is `target - x` already here?").
- You're counting frequencies, finding duplicates or grouping anagrams.
- A brute force compares all pairs and you want O(n).

```python
def two_sum(nums, target):
    seen = {}                       # value -> index
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```

`collections.Counter` and `defaultdict(list)` cover most counting and grouping.

## leetcode.prefix-sums
Precompute `pre[i]` = the sum of the first `i` elements, and any range sum becomes `pre[r + 1] - pre[l]` in O(1). The stronger form pairs a running prefix with a hash map of prefixes already seen. A subarray ending here sums to `k` exactly when `prefix - k` appeared earlier.

**Reach for it when**
- You're asked about sums (or counts, or XORs) over many subarrays or ranges.
- "Number of subarrays with sum equal to k", including arrays with negative numbers, where a sliding window fails.
- A running balance matters, such as equal 0s and 1s mapped to −1/+1.

```python
def subarray_sum(nums, k):
    count, prefix = 0, 0
    seen = {0: 1}                   # prefix sum -> how many times
    for x in nums:
        prefix += x
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
```

## leetcode.two-pointers
Two indices share one pass over the data. Either they start at both ends and move inward (pair sums on a sorted array, palindromes, container with most water), or both move forward, a slow write pointer behind a fast read pointer (dedupe in place, partitioning). Each step rules out a whole set of candidates, so O(n²) pairs collapse to O(n).

**Reach for it when**
- The input is sorted, or sorting it first is allowed, and you're after pairs or triples.
- You must compact or partition an array in place.
- You're comparing a sequence against its own reverse.

```python
def pair_with_sum(nums, target):     # nums sorted
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target:
            return [lo, hi]
        if s < target:
            lo += 1                  # need bigger
        else:
            hi -= 1                  # need smaller
    return [-1, -1]
```

3Sum = sort, fix one element, then run this on the rest.

## leetcode.sliding-window
Keep a contiguous window `[left, right]`. Extend `right` one step at a time. When the window breaks the constraint (too big a sum, a repeated character, too many distinct letters), advance `left` until it's valid again. Every element enters and leaves the window once, so the whole scan is O(n).

**Reach for it when**
- You're asked for the longest or shortest contiguous subarray or substring under a condition.
- Fixed-size windows: max sum of size k, anagrams of a pattern inside a string.
- The condition is monotone: shrinking the window can only help it hold.

```python
def longest_without_repeat(s):
    last = {}                        # char -> last index
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1      # shrink past the repeat
        last[ch] = right
        best = max(best, right - left + 1)
    return best
```

Negative numbers break "shrink when the sum is too big". Use prefix sums there.

## leetcode.linked-lists
A linked list is nodes joined by `next` pointers, with no random access. Nearly every problem is a careful walk that re-points `next` fields. Two habits prevent most bugs. Put a **dummy head** in front so the real head needs no special case. Save `nxt = cur.next` before you overwrite anything.

**Reach for it when**
- The input is a `ListNode` and you must insert, delete, merge or rearrange nodes.
- You'd reach for index arithmetic and can't have it.

```python
def merge_two(a, b):
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next
```

## leetcode.fast-slow-pointers
Move two pointers through a sequence at different speeds, usually one step and two steps. If there's a cycle, the fast pointer eventually laps the slow one (Floyd's tortoise and hare). If there isn't, the fast pointer reaches the end just as the slow one reaches the middle. It works on anything with a "next" function, including number sequences like happy numbers.

**Reach for it when**
- You must detect a cycle, or find where it starts, in O(1) space.
- You need the middle of a list, e.g. to split it or to check a palindrome.
- A value maps to a next value and you want to know if the sequence repeats.

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False
```

For the cycle start, reset one pointer to `head` after they meet and step both one at a time. They meet again at the entrance.

## leetcode.in-place-reversal
Reverse a linked list, or a piece of one, by turning each `next` pointer around as you walk, with three variables: `prev`, `cur`, `nxt`. Harder variants reverse positions `m..n` or every group of `k` nodes. The core loop is the same. The work is reconnecting the reversed piece to the nodes before and after it.

**Reach for it when**
- You're asked to reverse a list, a sub-list or k-groups without extra memory.
- A palindrome check or reorder needs half the list reversed.

```python
def reverse(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next
        cur.next = prev              # turn the pointer around
        prev, cur = cur, nxt
    return prev                      # new head
```

## leetcode.cyclic-sort
When an array holds numbers from a known range like `1..n` (or `0..n`), each number has a home index. Swap every number into its home until each slot holds its own value or a value that can't go anywhere. One scan afterwards finds the missing numbers, the duplicates or the first missing positive, in O(n) time and O(1) space.

**Reach for it when**
- The values are bounded by the array length: "numbers in range [1, n]".
- You're asked for missing, duplicate or corrupt numbers with O(1) extra space.

```python
def find_missing(nums):              # values in 1..n
    i = 0
    while i < len(nums):
        home = nums[i] - 1
        if 0 <= home < len(nums) and nums[i] != nums[home]:
            nums[i], nums[home] = nums[home], nums[i]
        else:
            i += 1
    return [i + 1 for i, x in enumerate(nums) if x != i + 1]
```

## leetcode.stack
A stack (a Python list with `append`/`pop`) remembers what is still open, most recent first. That matches anything nested: brackets, function calls, directory paths, expressions with parentheses. Push when something opens. Pop and resolve when it closes.

**Reach for it when**
- You're validating or matching brackets and tags.
- You're evaluating an expression (reverse Polish, a basic calculator) or decoding `3[a2[c]]`.
- You're simplifying a path, or undoing the most recent thing (backspace strings).

```python
def valid_brackets(s):
    pair = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
        else:
            stack.append(ch)
    return not stack
```

## leetcode.monotonic-stack
A stack kept in sorted order, all increasing or all decreasing. When a new element breaks the order, pop until it fits. Every popped element has just found its answer: the new element is its next greater (or smaller) one. Each element is pushed and popped once, so the pass is O(n).

**Reach for it when**
- "Next greater element", "days until a warmer temperature", "previous smaller".
- Largest rectangle in a histogram, trapping rain water, stock span.
- Building the smallest number by removing k digits.

```python
def next_greater(nums):
    ans = [-1] * len(nums)
    stack = []                       # indices, values decreasing
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            ans[stack.pop()] = x     # x is its next greater
        stack.append(i)
    return ans
```

Store indices, not values, when you need distances.

## leetcode.recursion
A function solves a problem by calling itself on a smaller instance and combining the result. Two parts, every time: a **base case** that answers directly, and a **recursive step** that shrinks the input toward it. Trust that the smaller call works, and write only how to build this answer from it.

**Reach for it when**
- The problem is defined in terms of itself: power, factorial, reversing, generating sequences.
- The input is recursive: trees, nested lists, divide-and-conquer.

```python
def my_pow(x, n):
    if n == 0:
        return 1.0                   # base case
    if n < 0:
        return 1 / my_pow(x, -n)
    half = my_pow(x, n // 2)         # smaller instance
    return half * half * (x if n % 2 else 1)
```

Python's default recursion limit is about 1000 frames. Deep linear recursion may need a loop instead.

## leetcode.sorting-algorithms
Sometimes the sort *is* the exercise. **Merge sort** splits in half, sorts each half and merges. It's stable, always O(n log n), and the merge step can count inversions as a side effect. **Quick sort / quickselect** partitions around a pivot and recurses on one or both sides. Custom orderings come from a key function (`sorted(xs, key=...)`) or `functools.cmp_to_key`.

**Reach for it when**
- You're asked to implement a sort, or sort without the built-in.
- Counting inversions or "reverse pairs" as part of merging.
- A custom order, e.g. arranging numbers to form the largest number.

```python
def merge_sort(a):
    if len(a) <= 1:
        return a
    mid = len(a) // 2
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1   # inversions += len(left) - i
    return out + left[i:] + right[j:]
```

## leetcode.merge-intervals
Sort intervals by start. Then any overlap is with the most recent interval you kept: if the next one starts before the last kept one ends, extend that one. Otherwise start a new interval. Insert, intersect and "free time" questions are variations of this same sweep.

**Reach for it when**
- The input is a list of `[start, end]` pairs.
- You're asked to merge overlaps, insert an interval, or intersect two interval lists.
- Meeting-room questions: can one person attend all of them, and how many rooms are needed at once.

```python
def merge(intervals):
    intervals.sort(key=lambda iv: iv[0])
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)   # overlap: extend
        else:
            out.append([s, e])
    return out
```

For "minimum rooms", sort starts and ends separately, or keep a heap of end times.

## leetcode.binary-search
Halve a search space whose predicate is monotone: false, false, …, true, true. It can be a sorted array, a rotated one, or an *answer space* like "the smallest speed that finishes in time". Pick the half that must still contain the answer, and keep the loop invariant exact so it ends on the boundary.

**Reach for it when**
- The input is sorted (or rotated sorted) and you want O(log n).
- First or last position of a value, insertion point, peak element.
- Minimise a maximum or maximise a minimum. Guess the answer, check it's feasible, and binary-search the guess.

```python
def first_true(lo, hi, ok):          # smallest x in [lo, hi] with ok(x)
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid                 # mid might be the answer
        else:
            lo = mid + 1
    return lo
```

`bisect.bisect_left` / `bisect_right` do the array version for you.

## leetcode.bit-manipulation
Treat an integer as a row of bits. `x ^ x == 0` and `x ^ 0 == x`, so XOR-ing everything cancels the pairs and leaves the odd one out. Masks test and set bits (`x >> i & 1`, `x | 1 << i`). `x & (x - 1)` clears the lowest set bit. An n-bit mask can also stand for a subset of n items.

**Reach for it when**
- "Every element appears twice except one": XOR them all.
- You're counting set bits, reversing bits, or checking powers of two.
- A small set (n ≤ 20) can be encoded as a bitmask for enumeration or DP.

```python
def single_number(nums):
    acc = 0
    for x in nums:
        acc ^= x                     # pairs cancel
    return acc

def count_bits(x):
    n = 0
    while x:
        x &= x - 1                   # drop lowest set bit
        n += 1
    return n
```

Python ints don't overflow. Mask with `& 0xFFFFFFFF` when a problem assumes 32-bit.

## leetcode.math
Here the key insight is arithmetic, not a data structure. Examples: pulling digits off with `% 10` and `// 10`, gcd and lcm, a sieve of primes, modular arithmetic under `10**9 + 7`, or a closed form that replaces a loop, such as the sum `n(n+1)/2`.

**Reach for it when**
- The problem is about digits, divisibility, primes, powers or bases.
- The brute force is huge but a formula or number-theory fact collapses it.
- The answer must be returned "modulo 1e9+7".

```python
def count_primes(n):                 # primes < n
    if n < 3:
        return 0
    is_p = [True] * n
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = [False] * len(range(i*i, n, i))
    return sum(is_p)
```

`math.gcd`, `math.comb` and `pow(b, e, mod)` are in the standard library.

## leetcode.binary-trees
A `TreeNode` has a value and up to two children. Most tree problems are one traversal plus a question asked at every node. **Preorder** visits node, left, right. **Inorder** visits left, node, right. **Postorder** visits left, right, node. Height, size, symmetry and "same tree" all come down to recursing on both children and combining the results.

**Reach for it when**
- The input is a `TreeNode` root.
- You're traversing, measuring (depth, count) or comparing trees.
- You're building a tree from traversal orders.

```python
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

def inorder(root, out):
    if root:
        inorder(root.left, out); out.append(root.val); inorder(root.right, out)
```

## leetcode.tree-bfs
Breadth-first search visits a tree level by level with a queue. Take the queue's current length at the start of each round, and exactly that level gets processed. That's how you get level lists, zigzag order, level averages, the right-side view, and the minimum depth (the first leaf reached).

**Reach for it when**
- The answer is grouped by depth ("level order", "each level's …").
- You need the shallowest thing (minimum depth, nearest leaf).
- You're connecting nodes to their neighbours on the same level.

```python
from collections import deque
def level_order(root):
    out, q = [], deque([root] if root else [])
    while q:
        level = []
        for _ in range(len(q)):      # exactly this level
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

## leetcode.tree-dfs
Depth-first search follows one root-to-leaf path at a time, and information flows two ways. Pass state **down** as arguments (the remaining sum, the path so far). Combine results **up** as return values (subtree height, best path through this node). Many hard problems return one thing to the parent while updating a global best on the side.

**Reach for it when**
- You're dealing with root-to-leaf paths: path sum, all paths, sum of numbers along paths.
- A property combines both subtrees: diameter, balanced check, max path sum.
- Lowest common ancestor.

```python
def diameter(root):
    best = 0
    def height(node):
        nonlocal best
        if not node:
            return 0
        l, r = height(node.left), height(node.right)
        best = max(best, l + r)      # path through this node
        return 1 + max(l, r)         # what the parent needs
    height(root)
    return best
```

## leetcode.bst
In a binary search tree, everything left of a node is smaller and everything right is larger. So a search goes down one side only, in O(height). An inorder traversal yields the values in sorted order. Validation must carry bounds down the tree. Comparing a node only to its children is the classic mistake.

**Reach for it when**
- The input is described as a BST.
- Search, insert or delete; kth smallest; inorder successor; range sums.
- Validating a BST, or building a balanced one from a sorted array.

```python
def is_valid_bst(root, lo=float("-inf"), hi=float("inf")):
    if not root:
        return True
    if not lo < root.val < hi:
        return False
    return (is_valid_bst(root.left, lo, root.val) and
            is_valid_bst(root.right, root.val, hi))
```

## leetcode.graphs
A graph is nodes plus edges, usually given as an edge list you turn into an adjacency list. Traverse it with BFS (a queue, finds shortest paths in unweighted graphs) or DFS (recursion or an explicit stack). Unlike a tree, a graph can loop back on itself, so keep a `visited` set.

**Reach for it when**
- The input is edges, an adjacency list, or "connections", "friends", "prerequisites".
- You're asked whether a path exists, to count connected components, or to clone a graph.
- A bipartite check: 2-colour the graph with BFS.

```python
from collections import defaultdict
def count_components(n, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    seen, comps = set(), 0
    for start in range(n):
        if start in seen:
            continue
        comps += 1
        stack = [start]; seen.add(start)
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
    return comps
```

## leetcode.islands
A 2-D grid is a graph whose edges are the 4 neighbours of each cell. Scan every cell. When you hit unvisited land, flood-fill it with DFS or BFS and count one island. Multi-source BFS (every rotten orange starts in the queue together) measures how far something spreads, minute by minute.

**Reach for it when**
- The input is a matrix of `0`/`1`, `"L"`/`"W"` or colours.
- Islands, flood fill, enclosed regions, surrounded regions.
- A spread over time (rotting oranges, distance to the nearest 0).

```python
def num_islands(grid):
    R, C = len(grid), len(grid[0])
    def sink(r, c):
        if 0 <= r < R and 0 <= c < C and grid[r][c] == "1":
            grid[r][c] = "0"         # mark visited
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                sink(r + dr, c + dc)
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == "1":
                count += 1; sink(r, c)
    return count
```

## leetcode.topological-sort
A topological order lists the nodes of a directed acyclic graph so that every edge points forward. **Kahn's algorithm** counts in-degrees, starts from the nodes with none, and releases a node once all its prerequisites have been output. If some nodes never reach in-degree 0, the graph has a cycle and no valid order exists.

**Reach for it when**
- There are dependencies: courses with prerequisites, build order, task scheduling.
- You're asked whether all tasks can finish (cycle detection in a directed graph).
- You must recover an order from pairwise comparisons, as in alien dictionary.

```python
from collections import deque
def course_order(n, prereqs):        # [course, needs]
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for c, need in prereqs:
        adj[need].append(c); indeg[c] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []   # [] = cycle
```

## leetcode.union-find
Disjoint-set union keeps a `parent` pointer for each element. `find` walks up to the root that represents a group, and `union` joins two roots. With **path compression**, each operation is effectively O(1). It's the natural tool when connections arrive one at a time and you keep asking "are these connected now?"

**Reach for it when**
- You're counting connected components while edges are added.
- Finding the edge that closes a cycle (redundant connection).
- Merging groups by shared keys (accounts merge, similar strings).

```python
parent = list(range(n))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path compression
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
        return False                    # already connected
    parent[ra] = rb
    return True
```

## leetcode.top-k-elements
A heap gives you the smallest item in O(log n) per push and pop. To keep the **k largest**, hold a min-heap of size k and pop whenever it grows past k. What remains is the top k, and the root is the kth largest. Python's `heapq` is a min-heap, so negate values when you need a max-heap.

**Reach for it when**
- "k largest", "k most frequent", "k closest", "kth largest".
- You repeatedly need the current smallest or largest item while data changes (task scheduler, last stone weight).

```python
import heapq
from collections import Counter
def top_k_frequent(nums, k):
    heap = []
    for val, cnt in Counter(nums).items():
        heapq.heappush(heap, (cnt, val))
        if len(heap) > k:
            heapq.heappop(heap)      # drop the smallest count
    return [val for _, val in heap]
```

`heapq.nlargest(k, xs)` is fine for a one-shot answer.

## leetcode.two-heaps
Split the numbers into a lower half (a max-heap) and an upper half (a min-heap), balanced so their sizes differ by at most one. The median then sits at the top of one or both heaps, and every insert costs O(log n). The same balancing works whenever you must choose among "what I can afford" versus "what's worth the most", as in IPO.

**Reach for it when**
- Running median of a stream, or the median of a sliding window.
- A greedy choice alternates between two ordered pools.

```python
import heapq
class MedianFinder:
    def __init__(self):
        self.lo, self.hi = [], []    # lo = max-heap (negated)
    def addNum(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

## leetcode.k-way-merge
To merge k sorted lists, keep a min-heap holding the current front of each list. Pop the smallest, output it, and push the next item from the same list. The heap never holds more than k items, so merging N items total costs O(N log k). A sorted matrix is just k sorted rows.

**Reach for it when**
- You're merging k sorted lists or arrays.
- Kth smallest in a sorted matrix, or across several sorted arrays.
- The smallest range that covers an element from each list.

```python
import heapq
def merge_k_lists(lists):
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapq.heapify(heap)
    dummy = tail = ListNode()
    while heap:
        _, i, node = heapq.heappop(heap)
        tail.next = tail = node
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

The index `i` in the tuple breaks ties so Python never compares two nodes.

## leetcode.shortest-paths
With weighted edges, plain BFS no longer gives shortest paths. **Dijkstra** pops the closest unsettled node from a heap and relaxes its edges. It's O(E log V) and needs non-negative weights. **Bellman-Ford** relaxes every edge up to V−1 times and handles negative weights or a cap on the number of edges ("at most k stops"). **Floyd-Warshall** gives all pairs in O(V³).

**Reach for it when**
- Edges carry costs or times: network delay, cheapest flight, path with minimum effort.
- "At most k stops" usually means a Bellman-Ford variant limited to k rounds.

```python
import heapq
def dijkstra(adj, src):              # adj[u] = [(v, w), ...]
    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue                 # stale entry
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist
```

## leetcode.mst
A minimum spanning tree connects every node using the least total edge weight. **Kruskal** sorts edges by weight and adds each one unless union-find says it would close a cycle. **Prim** grows one tree from a start node, always adding the cheapest edge that leaves it, using a heap. Kruskal suits edge lists. Prim suits dense graphs like "all pairs of points".

**Reach for it when**
- "Minimum cost to connect all points, cities or houses".
- You need the cheapest set of edges that keeps everything connected.

```python
def kruskal(n, edges):               # edges = [(w, u, v), ...]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    total = used = 0
    for w, u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv; total += w; used += 1
    return total if used == n - 1 else -1
```

## leetcode.subsets
Generate every subset, permutation or combination. One way is breadth-first: start with `[[]]` and, for each new element, add it to a copy of everything you have so far. The other is include/exclude recursion. With duplicates, sort first and skip a repeated value in the same position so you never build the same set twice.

**Reach for it when**
- The words "all subsets", "all permutations", "letter case permutations", "power set".
- The input is small (n ≤ 15 or so), because the output is exponential.

```python
def subsets(nums):
    out = [[]]
    for x in nums:
        out += [s + [x] for s in out]   # each old subset, with x added
    return out

def permutations(nums):
    if not nums:
        return [[]]
    return [[x] + p for i, x in enumerate(nums)
            for p in permutations(nums[:i] + nums[i+1:])]
```

## leetcode.backtracking
Build a candidate one choice at a time. **Choose** an option, **recurse** to make the next choice, then **un-choose** it and try the next option. Prune early: stop a branch as soon as it can't lead to a valid answer. That's the difference between a solution that finishes and one that times out.

**Reach for it when**
- You're searching combinations under constraints: combination sum, N-queens, sudoku.
- Word search on a grid, palindrome partitioning, generating valid parentheses.
- "Return all valid …" where brute force is exponential but pruning cuts it down.

```python
def combination_sum(cands, target):
    out, path = [], []
    def go(start, remain):
        if remain == 0:
            out.append(path[:]); return
        for i in range(start, len(cands)):
            if cands[i] > remain:
                continue             # prune
            path.append(cands[i])    # choose
            go(i, remain - cands[i]) # explore (i: reuse allowed)
            path.pop()               # un-choose
    go(0, target)
    return out
```

## leetcode.trie
A trie (prefix tree) stores words character by character. Each node is a dict of children, plus a flag for "a word ends here". Insert and lookup cost O(length of the word), whatever the dictionary's size. Every word that shares a prefix shares one path, so "any word starting with …" is a single walk.

**Reach for it when**
- You're asked for prefix search or autocomplete (`startsWith`).
- Wildcard search (`.` matches any letter) with DFS through the trie.
- Word search II: many words on one board, where the trie prunes dead prefixes.

```python
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = True             # end of word
    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node
    def search(self, word):
        node = self._walk(word); return bool(node) and "$" in node
    def startsWith(self, prefix):
        return self._walk(prefix) is not None
```

## leetcode.greedy
Make the choice that looks best right now and never revisit it. This only works when you can argue that some optimal answer agrees with that choice, often through an exchange argument. The usual setup is to sort, then sweep and commit. The hard part is spotting which quantity to be greedy about.

**Reach for it when**
- Interval scheduling (keep the one that ends earliest), non-overlapping intervals.
- Jump game (track the farthest reachable index), gas station, assign cookies.
- A DP would work but each state has one obviously dominant choice.

```python
def can_jump(nums):
    reach = 0
    for i, step in enumerate(nums):
        if i > reach:
            return False             # stuck before i
        reach = max(reach, i + step)
    return True
```

If you can't justify the greedy step, try a small counterexample. If one exists, it's DP.

## leetcode.dp-1d
Dynamic programming stores the answers to subproblems so each is solved once. In 1-D, `dp[i]` is the answer for the first `i` items, or for position `i`, and it depends on a few earlier entries. Write the recurrence first. Then either memoise the recursion (`@cache`) or fill the table left to right. Often only the last two values are needed.

**Reach for it when**
- Counting ways (climbing stairs, decode ways) or best total (house robber).
- Coin change, word break: "can the prefix up to `i` be built?"
- A recursive brute force keeps re-solving the same inputs.

```python
def rob(nums):
    take = skip = 0                  # best ending with/without the last house
    for x in nums:
        take, skip = skip + x, max(skip, take)
    return max(take, skip)

def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float("inf") else -1
```

## leetcode.knapsack-dp
Each item is used at most once and either taken or skipped, against a capacity or a target sum. `dp[j]` answers "is sum `j` reachable?" (or counts the ways to reach it, or gives the best value at weight `j`) using the items so far. With a 1-D table, loop capacity **downwards** so an item can't be counted twice.

**Reach for it when**
- Partition into two equal-sum subsets, target sum with +/− signs, last stone weight II.
- "Choose some items so their sizes hit exactly X."

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [True] + [False] * target   # dp[j]: some subset sums to j
    for x in nums:
        for j in range(target, x - 1, -1):   # downwards: each x once
            dp[j] = dp[j] or dp[j - x]
    return dp[target]
```

Loop capacity **upwards** and you get the unbounded version, where each item can repeat (coin change II).

## leetcode.dp-grid
`dp[r][c]` holds the answer for reaching (or starting from) cell `(r, c)`, built from its neighbours, usually the cell above and the cell to the left. Fill the table in an order where those neighbours are already done. Only one row is ever needed at a time, so it compresses to O(columns) space.

**Reach for it when**
- A grid path moves only right and down: unique paths, minimum path sum.
- Maximal square, triangle, dungeon game, cherry pickup.

```python
def min_path_sum(grid):
    R, C = len(grid), len(grid[0])
    dp = [float("inf")] * C
    dp[0] = 0
    for r in range(R):
        for c in range(C):
            left = dp[c - 1] if c else float("inf")
            dp[c] = grid[r][c] + min(dp[c], left)   # dp[c] = from above
    return dp[-1]
```

## leetcode.dp-strings
Two sequences get a 2-D table: `dp[i][j]` answers the question for the first `i` characters of `a` and the first `j` of `b`. When the last characters match, you usually extend `dp[i-1][j-1]`. When they don't, you take the best of dropping one character from either string (or, for edit distance, of the three edits).

**Reach for it when**
- Longest common subsequence, edit distance, distinct subsequences.
- Interleaving strings, wildcard and regex matching.
- Anything that compares or aligns two strings or arrays.

```python
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
```

Row 0 and column 0 stand for the empty prefix. Set them first.

## leetcode.dp-lis
Longest increasing subsequence. The O(n²) DP sets `dp[i]` to the longest run ending at `i`, from any smaller earlier element. The O(n log n) version keeps `tails[k]` = the smallest possible tail of an increasing run of length `k + 1`, and places each number with `bisect_left`. The length of `tails` is the answer.

**Reach for it when**
- You're asked for the longest increasing (or non-decreasing) subsequence.
- Russian doll envelopes: sort by width, height descending on ties, then LIS on height.
- Largest divisible subset, number of LIS, longest chain of pairs.

```python
from bisect import bisect_left
def length_of_lis(nums):
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)          # extends the longest run
        else:
            tails[i] = x             # a smaller tail for length i+1
    return len(tails)
```

## leetcode.dp-intervals
`dp[i][j]` is the best answer for the subarray or substring from `i` to `j`. Compute it by trying every split point `k` between them and combining `dp[i][k]` with `dp[k][j]`, plus the cost of that split. Fill the table by increasing interval **length**, so every smaller range is ready when you need it. That usually costs O(n³).

**Reach for it when**
- The answer for a range depends on how you split it: matrix chain, burst balloons.
- Minimum cost to cut a stick, palindrome partitioning II, strange printer.
- Longest palindromic subsequence, where `dp[i][j]` is built from the inner range.

```python
def min_cut_cost(n, cuts):
    pts = [0] + sorted(cuts) + [n]
    m = len(pts)
    dp = [[0] * m for _ in range(m)]
    for length in range(2, m):
        for i in range(m - length):
            j = i + length
            dp[i][j] = min(dp[i][k] + dp[k][j] for k in range(i + 1, j)) + pts[j] - pts[i]
    return dp[0][m - 1]
```

## leetcode.ordered-set
You sometimes need a collection that stays sorted while you insert into it, so you can ask for the floor, the ceiling or a range at any moment. Python has no built-in balanced tree. A sorted list maintained with `bisect.insort` answers those queries by binary search. Inserts cost O(n) in the worst case, but they're fast in practice at LeetCode sizes.

**Reach for it when**
- Booking or calendar problems: does the new interval overlap its neighbours?
- You need the nearest value above or below x among the items seen so far.
- Sliding windows that need the window's min and max, or order statistics.

```python
from bisect import bisect_left, insort
class MyCalendar:
    def __init__(self):
        self.starts, self.ends = [], []
    def book(self, s, e):
        i = bisect_left(self.starts, s)
        if i > 0 and self.ends[i - 1] > s:
            return False             # overlaps the one before
        if i < len(self.starts) and self.starts[i] < e:
            return False             # overlaps the one after
        self.starts.insert(i, s); self.ends.insert(i, e)
        return True
```

A monotonic deque is often the O(n) alternative for window min and max.
