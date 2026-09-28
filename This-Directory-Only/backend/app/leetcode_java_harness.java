/* Java test harness for the LeetCode drills (driven by leetcode_java.py).

   java -Djava.security.manager=allow -cp <compiled harness> DeltaJavaHarness <payload.json> <report.json>
   payload = {code, mode: "tests"|"run", cases: [{expr, expected}], caseTimeoutMs}
   mode "run" (the cell's ▶) runs a `main` if the code has one, else the cases
   for their printed output only.

   The learner's source is compiled IN MEMORY (javax.tools, no class files on
   disk) with ListNode/TreeNode alongside, loaded by a loader whose parent is
   the PLATFORM loader (so the harness's own classes are invisible to it), and
   given a ProtectionDomain with no permissions. Then a SecurityManager goes
   in: learner code cannot write or read files, open sockets, start processes,
   call System.exit, redirect System.out, or reflect into the harness. Every
   JDK frame keeps its rights, so collections, lambdas, streams and
   String.format behave normally. A JDK that cannot install a SecurityManager
   (24+) makes the harness refuse to run (exit 3): learner code never runs
   unfenced.

   Each case's `expr` is a small tree translated from the bank's Python call:
     {"v": json}                              a literal
     {"call": name, "args": [...]}            new Solution().name(args...)
     {"h": helper, "args": [...]}             list_node, tree_node, sorted…
     {"design": cls, "ops": [...], "args": [[...]]}   LeetCode's design driver
   Arguments are converted to whatever types the learner's method declares
   (int[] or List<Integer>, char[][] or List<List<String>>…), and whatever it
   returns is flattened back to JSON-shaped values and compared with
   `expected` (JSON of the Python literal). Equality mirrors
   code_runner._delta_equal: lists element-wise, integers exact, floats rtol
   1e-5 / atol 1e-6, true == 1.

   Rows go to the REPORT FILE, opened before the sandbox goes up, so nothing
   the learner prints can forge them. The learner's System.out / System.err
   go to stdout, capped at 64 KB. A compile error goes to stderr, exit 1. */
import java.io.*;
import java.lang.reflect.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.*;
import java.security.cert.Certificate;
import java.util.*;
import java.util.regex.*;
import javax.tools.*;

final class DeltaJavaHarness {
    static final double RTOL = 1e-5, ATOL = 1e-6;
    static final int OUTPUT_LIMIT = 64 * 1024;
    static final String IMPORTS = "import java.util.*;import java.util.function.*;import java.util.stream.*;";

    static final String LIST_NODE_SRC =
        "class ListNode {\n int val;\n ListNode next;\n ListNode() {}\n ListNode(int val) { this.val = val; }\n"
        + " ListNode(int val, ListNode next) { this.val = val; this.next = next; }\n}\n";
    static final String TREE_NODE_SRC =
        "class TreeNode {\n int val;\n TreeNode left;\n TreeNode right;\n TreeNode() {}\n TreeNode(int val) { this.val = val; }\n"
        + " TreeNode(int val, TreeNode left, TreeNode right) { this.val = val; this.left = left; this.right = right; }\n}\n";

    static MemLoader loader;
    static String learnerFile;
    static String[] codeLines = new String[0];
    static Set<String> learnerClasses = new HashSet<>();

    public static void main(String[] argv) throws Exception {
        Map<String, Object> payload = asMap(Json.parse(new String(Files.readAllBytes(Paths.get(argv[0])), StandardCharsets.UTF_8)));
        PrintStream report = new PrintStream(new FileOutputStream(argv[1]), true, "UTF-8");
        PrintStream realOut = new PrintStream(new FileOutputStream(FileDescriptor.out), true, "UTF-8");
        PrintStream realErr = new PrintStream(new FileOutputStream(FileDescriptor.err), true, "UTF-8");
        String code = (String) payload.get("code");
        long caseTimeout = ((Number) payload.getOrDefault("caseTimeoutMs", 4000L)).longValue();

        String compileError = compile(code);
        if (compileError != null) {
            realErr.print(compileError);
            realErr.flush();
            Runtime.getRuntime().halt(1);
        }

        PrintStream capped = new PrintStream(new CappedStream(realOut), true, "UTF-8");
        // Everything the harness itself needs later, loaded while it still may.
        for (Class<?> c : new Class<?>[] {Row.class, Json.class, CappedStream.class, Outcome.class}) c.getName();
        if (!sandbox()) {
            realErr.println("This server's Java cannot sandbox answers (it has no SecurityManager); Java answers are refused.");
            realErr.flush();
            Runtime.getRuntime().halt(3);
        }
        System.setOut(capped);
        System.setErr(capped);

        List<Row> rows = new ArrayList<>();
        if ("run".equals(payload.get("mode"))) {
            Method main = findMain();
            if (main != null) {
                Outcome o = timed(() -> { main.invoke(null, (Object) new String[0]); return null; }, caseTimeout * 2);
                capped.flush();
                if (o.error != null) realErr.println(o.error);
                realErr.flush();
                Runtime.getRuntime().halt(o.error == null ? 0 : 1);
            }
            if (asList(payload.get("cases")).isEmpty()) Runtime.getRuntime().halt(0);
        }

        boolean timedOut = false;
        for (Object raw : asList(payload.get("cases"))) {
            Map<String, Object> tc = asMap(raw);
            Object expected = Json.parse((String) tc.get("expected"));
            if (timedOut) {
                rows.add(new Row(false, "", Json.show(expected), "Not run: an earlier case timed out."));
                continue;
            }
            Outcome o = timed(() -> plain(eval(tc.get("expr"))), caseTimeout);
            if (o.timedOut) timedOut = true;
            if (o.error != null) rows.add(new Row(false, "", Json.show(expected), o.error));
            else rows.add(new Row(equal(o.value, expected), Json.show(o.value), Json.show(expected), ""));
        }
        capped.flush();
        if ("run".equals(payload.get("mode"))) {
            // The cell's ▶: what printed is on stdout; say why a case blew up.
            for (int i = 0; i < rows.size(); i++) {
                if (rows.get(i).error.isEmpty()) continue;
                realErr.println("case " + (i + 1) + ": " + rows.get(i).error);
                realErr.flush();
                Runtime.getRuntime().halt(1);
            }
            Runtime.getRuntime().halt(0);
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < rows.size(); i++) {
            if (i > 0) sb.append(',');
            Row r = rows.get(i);
            sb.append("{\"passed\":").append(r.passed).append(",\"actual\":").append(Json.quote(r.actual))
              .append(",\"expected\":").append(Json.quote(r.expected)).append(",\"error\":").append(Json.quote(r.error))
              .append(",\"note\":\"\"}");
        }
        report.print(sb.append(']'));
        report.flush();
        realOut.flush();
        Runtime.getRuntime().halt(0);
    }

    // ------------------------------------------------------------- compile

    static final Pattern PUBLIC_TYPE = Pattern.compile("(?m)^\\s*public\\s+(?:(?:final|abstract|static)\\s+)*(?:class|interface|enum|record)\\s+(\\w+)");

    /** null on success, else the diagnostics as the learner should read them. */
    static String compile(String code) throws IOException {
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        if (javac == null) return "This server has a Java runtime but no compiler (javac).\n";
        codeLines = code.split("\n", -1);
        Matcher m = PUBLIC_TYPE.matcher(code);
        learnerFile = (m.find() ? m.group(1) : "Solution") + ".java";
        List<JavaFileObject> sources = new ArrayList<>();
        // The imports share line 1 with the learner's first line, so every
        // line number javac or a stack trace reports is the learner's own.
        sources.add(new Source(learnerFile, IMPORTS + code));
        // The learner's own ListNode/TreeNode (pasted from LeetCode) wins; the
        // starter's commented-out definition is not one.
        String bare = code.replaceAll("(?s)/\\*.*?\\*/", "").replaceAll("//[^\\n]*", "");
        if (!Pattern.compile("\\b(?:class|record|interface)\\s+ListNode\\b").matcher(bare).find())
            sources.add(new Source("ListNode.java", LIST_NODE_SRC));
        if (!Pattern.compile("\\b(?:class|record|interface)\\s+TreeNode\\b").matcher(bare).find())
            sources.add(new Source("TreeNode.java", TREE_NODE_SRC));
        DiagnosticCollector<JavaFileObject> diags = new DiagnosticCollector<>();
        Map<String, ByteArrayOutputStream> classes = new HashMap<>();
        StandardJavaFileManager std = javac.getStandardFileManager(diags, Locale.ENGLISH, StandardCharsets.UTF_8);
        JavaFileManager fm = new ForwardingJavaFileManager<JavaFileManager>(std) {
            @Override
            public JavaFileObject getJavaFileForOutput(Location loc, String name, JavaFileObject.Kind kind, FileObject sibling) {
                ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                classes.put(name, bytes);
                return new SimpleJavaFileObject(URI.create("mem:///" + name.replace('.', '/') + kind.extension), kind) {
                    @Override public OutputStream openOutputStream() { return bytes; }
                };
            }
        };
        // An empty classpath: the learner compiles against the JDK and nothing of ours.
        List<String> opts = Arrays.asList("-proc:none", "-g", "-Xlint:none", "-nowarn", "-classpath", System.getProperty("java.io.tmpdir") + "/delta-empty-cp");
        StringWriter out = new StringWriter();
        boolean ok = javac.getTask(out, fm, diags, opts, null, sources).call();
        if (!ok) {
            StringBuilder sb = new StringBuilder();
            String[] lines = codeLines;
            int shown = 0;
            for (Diagnostic<? extends JavaFileObject> d : diags.getDiagnostics()) {
                if (d.getKind() != Diagnostic.Kind.ERROR) continue;
                if (++shown > 8) { sb.append("… more errors\n"); break; }
                String file = d.getSource() == null ? "" : new File(d.getSource().getName()).getName();
                long line = d.getLineNumber();
                sb.append(file.equals(learnerFile) || file.isEmpty() ? where(line) : file + " line " + line)
                  .append(": error: ").append(d.getMessage(Locale.ENGLISH)).append('\n');
                if (file.equals(learnerFile) && line >= 1 && line <= lines.length) sb.append("    ").append(lines[(int) line - 1].trim()).append('\n');
            }
            if (shown == 0) sb.append(out);
            return sb.length() == 0 ? "Compilation failed.\n" : sb.toString();
        }
        Map<String, byte[]> bytes = new HashMap<>();
        classes.forEach((k, v) -> bytes.put(k, v.toByteArray()));
        learnerClasses.addAll(bytes.keySet());
        loader = new MemLoader(bytes);
        return null;
    }

    static final class Source extends SimpleJavaFileObject {
        final String text;
        Source(String file, String text) { super(URI.create("string:///" + file), Kind.SOURCE); this.text = text; }
        @Override public CharSequence getCharContent(boolean ignore) { return text; }
    }

    static final class MemLoader extends ClassLoader {
        final Map<String, byte[]> bytes;
        // Reading system properties and nothing else. Static permissions, so
        // the Policy below is never consulted for this domain.
        final ProtectionDomain domain;
        MemLoader(Map<String, byte[]> bytes) {
            super("learner", ClassLoader.getPlatformClassLoader());
            this.bytes = bytes;
            Permissions perms = new Permissions();
            perms.add(new PropertyPermission("*", "read"));
            perms.setReadOnly();
            domain = new ProtectionDomain(new CodeSource(null, (Certificate[]) null), perms);
        }
        @Override protected Class<?> findClass(String name) throws ClassNotFoundException {
            byte[] b = bytes.get(name);
            if (b == null) throw new ClassNotFoundException(name);
            return defineClass(name, b, 0, b.length, domain);
        }
    }

    // ------------------------------------------------------------- sandbox

    @SuppressWarnings("removal")
    static boolean sandbox() {
        PrintStream err = System.err;
        try {
            // JDK 17-23 warn on stderr that the SecurityManager is deprecated; the learner need not see it.
            System.setErr(new PrintStream(OutputStream.nullOutputStream()));
            final ProtectionDomain learner = loader.domain;
            Policy.setPolicy(new Policy() {
                @Override public boolean implies(ProtectionDomain d, Permission p) { return d != learner; }
            });
            System.setSecurityManager(new SecurityManager() {
                /* The stock check guards only the root thread group, so an answer
                   could start threads without limit and exhaust the container's
                   PIDs or native memory. Starting a thread in ANY group needs
                   modifyThreadGroup, which the learner's domain lacks; the
                   harness's own case threads have no learner frame on the stack. */
                @Override public void checkAccess(ThreadGroup g) {
                    checkPermission(new RuntimePermission("modifyThreadGroup"));
                }
            });
            return System.getSecurityManager() != null;
        } catch (Throwable t) {
            return false;
        } finally {
            System.setErr(err);
        }
    }

    // ------------------------------------------------------------- running

    interface Body { Object run() throws Throwable; }

    static final class Outcome { Object value; String error; boolean timedOut; }

    /** Run `body` on its own thread (a deep stack for recursive answers) with a
     *  clock. A thread that overruns cannot be stopped; it is abandoned, the
     *  remaining cases are marked not run, and the process halts at the end. */
    static Outcome timed(Body body, long timeoutMs) throws InterruptedException {
        Outcome o = new Outcome();
        Thread t = new Thread(null, () -> {
            try { o.value = body.run(); }
            catch (Throwable e) { o.error = describe(e); }
        }, "case", 256L << 20);
        t.setDaemon(true);
        // Not the app loader the harness came from: the learner never gets a handle on it.
        t.setContextClassLoader(loader);
        t.start();
        t.join(timeoutMs);
        if (t.isAlive()) {
            o.timedOut = true;
            o.value = null;
            o.error = "Timed out after " + timeoutMs + " ms (an infinite loop?)";
        }
        return o;
    }

    static String describe(Throwable e) {
        while ((e instanceof InvocationTargetException || e instanceof ExceptionInInitializerError) && e.getCause() != null) e = e.getCause();
        if (e instanceof AccessControlException) return "Not allowed here: " + ((AccessControlException) e).getPermission();
        String name = e.getClass().getName().replaceFirst("^java\\.lang\\.", "");
        String msg = e.getMessage() == null ? name : name + ": " + e.getMessage();
        if (e instanceof StackOverflowError) return "StackOverflowError (infinite recursion?)";
        for (StackTraceElement f : e.getStackTrace()) {
            if (learnerClasses.contains(f.getClassName()) && learnerFile.equals(f.getFileName()) && f.getLineNumber() > 0)
                return msg + " (" + where(f.getLineNumber()).toLowerCase(Locale.ROOT) + ", in " + f.getMethodName() + ")";
        }
        return msg;
    }

    static final Pattern CELL_MARK = Pattern.compile("^// --- cell (\\d+) ---$");

    /** A line of the submitted text as the learner sees it in the editor. The
     *  notebook joins its cells under `// --- cell N ---` headers, so line 5
     *  of the submission is "Line 4" of cell 1, or "Cell 2, line 1". */
    static String where(long line) {
        int cells = 0, mark = 0;
        String cell = null;
        for (int i = 0; i < codeLines.length; i++) {
            Matcher m = CELL_MARK.matcher(codeLines[i].trim());
            if (!m.matches()) continue;
            cells++;
            if (i + 1 < line) { cell = m.group(1); mark = i + 1; }
        }
        if (cell == null) return "Line " + line;
        return cells > 1 ? "Cell " + cell + ", line " + (line - mark) : "Line " + (line - mark);
    }

    static Method findMain() {
        for (String name : learnerClasses) {
            try {
                Method m = loader.loadClass(name).getDeclaredMethod("main", String[].class);
                if (Modifier.isStatic(m.getModifiers())) { m.setAccessible(true); return m; }
            } catch (ReflectiveOperationException | LinkageError ignored) { }
        }
        return null;
    }

    // ------------------------------------------------------------- evaluation

    static Object eval(Object node) throws Throwable {
        Map<String, Object> n = asMap(node);
        if (n.containsKey("v")) return n.get("v");
        if (n.containsKey("call")) {
            List<Object> args = new ArrayList<>();
            for (Object a : asList(n.get("args"))) args.add(eval(a));
            Class<?> sol = learnerClass("Solution");
            Object[] conv = new Object[args.size()];
            Method m = (Method) pick(allMethods(sol, (String) n.get("call")), args, conv, "Solution." + n.get("call"));
            Object self = null;
            if (!Modifier.isStatic(m.getModifiers())) {
                Constructor<?> c = sol.getDeclaredConstructor();
                c.setAccessible(true);
                self = c.newInstance();
            }
            Object r = m.invoke(self, conv);
            return m.getReturnType() == void.class ? null : r;
        }
        if (n.containsKey("design")) return design((String) n.get("design"), asList(n.get("ops")), asList(n.get("args")));
        String h = (String) n.get("h");
        List<Object> args = new ArrayList<>();
        for (Object a : asList(n.get("args"))) args.add(eval(a));
        switch (h) {
            case "list_node": return listNode(asList(args.get(0)));
            case "list_node_cycle": {
                List<Object> pair = asList(args.get(0));
                Object head = listNode(asList(pair.get(0)));
                long pos = ((Number) pair.get(1)).longValue();
                if (head != null && pos >= 0) {
                    Field next = learnerClass("ListNode").getDeclaredField("next");
                    next.setAccessible(true);
                    Object tail = head, target = null, cur = head;
                    for (int i = 0; cur != null; i++) { if (i == pos) target = cur; tail = cur; cur = next.get(cur); }
                    next.set(tail, target);
                }
                return head;
            }
            case "tree_node": return treeNode(asList(args.get(0)));
            case "_from_list_node":
            case "_from_tree_node": {
                Object p = plain(args.get(0));
                return p == null ? new ArrayList<>() : p;
            }
            case "is_same_list":
            case "is_same_tree": return strictEqual(plain(args.get(0)), plain(args.get(1)));
            case "sorted": {
                Object p = plain(args.get(0));
                if (!(p instanceof List)) throw new IllegalArgumentException("sorted() needs a list, got " + Json.show(p));
                List<Object> copy = new ArrayList<>(asList(p));
                copy.sort(DeltaJavaHarness::pyCompare);
                return copy;
            }
            default: throw new IllegalStateException("unknown helper " + h);
        }
    }

    static Object design(String cls, List<Object> ops, List<Object> args) throws Throwable {
        Class<?> c = learnerClass(cls);
        List<Object> outs = new ArrayList<>();
        Object obj = null;
        for (int i = 0; i < ops.size(); i++) {
            List<Object> a = asList(args.get(i));
            Object[] conv = new Object[a.size()];
            if (i == 0) {
                Constructor<?> k = (Constructor<?>) pick(Arrays.asList(c.getDeclaredConstructors()), a, conv, cls + " constructor");
                k.setAccessible(true);
                obj = k.newInstance(conv);
                outs.add(null);
                continue;
            }
            String op = (String) ops.get(i);
            Method m = (Method) pick(allMethods(c, op), a, conv, cls + "." + op);
            Object r = m.invoke(Modifier.isStatic(m.getModifiers()) ? null : obj, conv);
            outs.add(m.getReturnType() == void.class ? null : plain(r));
        }
        return outs;
    }

    static Class<?> learnerClass(String name) throws ClassNotFoundException {
        try { return loader.loadClass(name); }
        catch (ClassNotFoundException e) { throw new ClassNotFoundException("no class " + name + " in your code — declare `class " + name + " { ... }`"); }
    }

    static List<Executable> allMethods(Class<?> c, String name) {
        List<Executable> out = new ArrayList<>();
        for (Class<?> k = c; k != null && k != Object.class; k = k.getSuperclass())
            for (Method m : k.getDeclaredMethods()) if (m.getName().equals(name) && !m.isSynthetic()) out.add(m);
        return out;
    }

    /** The overload whose parameters the arguments convert to (by count, then by type). */
    static Executable pick(List<Executable> candidates, List<Object> args, Object[] conv, String what) {
        String why = what + " not found";
        for (Executable e : candidates) {
            if (e.getParameterCount() != args.size()) { why = what + " takes " + e.getParameterCount() + " argument(s), the test passes " + args.size(); continue; }
            try {
                Type[] types = e.getGenericParameterTypes();
                for (int i = 0; i < types.length; i++) conv[i] = convert(args.get(i), types[i]);
                e.setAccessible(true);
                return e;
            } catch (IllegalArgumentException ex) { why = ex.getMessage(); }
        }
        throw new IllegalArgumentException(why);
    }

    // ------------------------------------------------------------- JSON -> Java

    static Class<?> raw(Type t) {
        if (t instanceof Class) return (Class<?>) t;
        if (t instanceof ParameterizedType) return (Class<?>) ((ParameterizedType) t).getRawType();
        if (t instanceof GenericArrayType) return Array.newInstance(raw(((GenericArrayType) t).getGenericComponentType()), 0).getClass();
        if (t instanceof WildcardType) return raw(((WildcardType) t).getUpperBounds()[0]);
        if (t instanceof TypeVariable) return raw(((TypeVariable<?>) t).getBounds()[0]);
        return Object.class;
    }

    static Type elem(Type t) {
        if (t instanceof ParameterizedType) return ((ParameterizedType) t).getActualTypeArguments()[0];
        return Object.class;
    }

    static boolean isJson(Object v) {
        return v == null || v instanceof Long || v instanceof Double || v instanceof String || v instanceof Boolean || v instanceof List;
    }

    static Object convert(Object v, Type t) {
        Class<?> c = raw(t);
        if (v == null) {
            if (c.isPrimitive()) throw new IllegalArgumentException("cannot pass null as " + c.getName());
            return null;
        }
        if (!isJson(v)) {  // a ListNode/TreeNode a helper built
            if (c.isInstance(v)) return v;
            throw new IllegalArgumentException("cannot pass a " + v.getClass().getSimpleName() + " as " + t.getTypeName());
        }
        if (c == int.class || c == Integer.class) return (int) integral(v, Integer.MIN_VALUE, Integer.MAX_VALUE, "int");
        if (c == long.class || c == Long.class) return integral(v, Long.MIN_VALUE, Long.MAX_VALUE, "long");
        if (c == short.class || c == Short.class) return (short) integral(v, Short.MIN_VALUE, Short.MAX_VALUE, "short");
        if (c == byte.class || c == Byte.class) return (byte) integral(v, Byte.MIN_VALUE, Byte.MAX_VALUE, "byte");
        if (c == double.class || c == Double.class) return number(v).doubleValue();
        if (c == float.class || c == Float.class) return number(v).floatValue();
        if (c == boolean.class || c == Boolean.class) {
            if (v instanceof Boolean) return v;
            throw mismatch(v, t);
        }
        if (c == char.class || c == Character.class) {
            if (v instanceof String && ((String) v).length() == 1) return ((String) v).charAt(0);
            throw mismatch(v, t);
        }
        if (c == String.class || c == CharSequence.class) {
            if (v instanceof String) return v;
            throw mismatch(v, t);
        }
        if (c.isArray()) {
            List<Object> l = listOf(v, t);
            Type comp = t instanceof GenericArrayType ? ((GenericArrayType) t).getGenericComponentType() : c.getComponentType();
            Object arr = Array.newInstance(c.getComponentType(), l.size());
            for (int i = 0; i < l.size(); i++) Array.set(arr, i, convert(l.get(i), comp));
            return arr;
        }
        try {
            if (c.getSimpleName().equals("ListNode") && c.getClassLoader() == loader) return listNode(listOf(v, t));
            if (c.getSimpleName().equals("TreeNode") && c.getClassLoader() == loader) return treeNode(listOf(v, t));
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("cannot build a " + c.getSimpleName() + ": " + e);
        }
        if (c == Object.class) return natural(v);
        if (v instanceof List) {
            Collection<Object> coll;
            if (c.isAssignableFrom(ArrayList.class)) coll = new ArrayList<>();
            else if (c.isAssignableFrom(LinkedList.class)) coll = new LinkedList<>();
            else if (c.isAssignableFrom(ArrayDeque.class)) coll = new ArrayDeque<>();
            else if (c.isAssignableFrom(HashSet.class)) coll = new HashSet<>();
            else throw mismatch(v, t);
            for (Object x : (List<?>) v) coll.add(convert(x, elem(t)));
            return coll;
        }
        throw mismatch(v, t);
    }

    static List<Object> listOf(Object v, Type t) {
        if (v instanceof List) return asList(v);
        throw mismatch(v, t);
    }

    static IllegalArgumentException mismatch(Object v, Type t) {
        return new IllegalArgumentException("the test passes " + Json.show(v) + ", which is not a " + t.getTypeName().replace("java.util.", "").replace("java.lang.", ""));
    }

    static long integral(Object v, long lo, long hi, String type) {
        if (v instanceof Boolean) return (Boolean) v ? 1 : 0;
        if (!(v instanceof Long)) throw new IllegalArgumentException("the test passes " + Json.show(v) + ", which is not a " + type);
        long x = (Long) v;
        if (x < lo || x > hi) throw new IllegalArgumentException("the test passes " + x + ", too big for " + type + " (use long)");
        return x;
    }

    static Number number(Object v) {
        if (v instanceof Number) return (Number) v;
        if (v instanceof Boolean) return (Boolean) v ? 1 : 0;
        throw new IllegalArgumentException("the test passes " + Json.show(v) + ", which is not a number");
    }

    /** For an Object parameter: ints as Integer when they fit (what `(int) o` expects). */
    static Object natural(Object v) {
        if (v instanceof Long && (Long) v >= Integer.MIN_VALUE && (Long) v <= Integer.MAX_VALUE) return (int) (long) (Long) v;
        if (v instanceof List) {
            List<Object> out = new ArrayList<>();
            for (Object x : (List<?>) v) out.add(natural(x));
            return out;
        }
        return v;
    }

    static Object listNode(List<Object> values) throws ReflectiveOperationException {
        Class<?> c = learnerClass("ListNode");
        Field val = c.getDeclaredField("val"), next = c.getDeclaredField("next");
        val.setAccessible(true);
        next.setAccessible(true);
        Constructor<?> k = c.getDeclaredConstructor();
        k.setAccessible(true);
        Object head = null, cur = null;
        for (Object v : values) {
            Object n = k.newInstance();
            val.set(n, convert(v, val.getGenericType()));
            if (head == null) head = cur = n;
            else { next.set(cur, n); cur = n; }
        }
        return head;
    }

    static Object treeNode(List<Object> values) throws ReflectiveOperationException {
        if (values.isEmpty() || values.get(0) == null) return null;
        Class<?> c = learnerClass("TreeNode");
        Field val = c.getDeclaredField("val"), left = c.getDeclaredField("left"), right = c.getDeclaredField("right");
        for (Field f : new Field[] {val, left, right}) f.setAccessible(true);
        Constructor<?> k = c.getDeclaredConstructor();
        k.setAccessible(true);
        Object root = k.newInstance();
        val.set(root, convert(values.get(0), val.getGenericType()));
        ArrayDeque<Object> queue = new ArrayDeque<>();
        queue.add(root);
        int i = 1;
        while (!queue.isEmpty() && i < values.size()) {
            Object node = queue.poll();
            for (Field side : new Field[] {left, right}) {
                if (i < values.size() && values.get(i) != null) {
                    Object child = k.newInstance();
                    val.set(child, convert(values.get(i), val.getGenericType()));
                    side.set(node, child);
                    queue.add(child);
                }
                i++;
            }
        }
        return root;
    }

    // ------------------------------------------------------------- Java -> JSON

    /** What the learner returned, as JSON-shaped values: Long, Double, String,
     *  Boolean, List, null. Arrays and collections become lists, a char a
     *  one-letter string, a ListNode its values, a TreeNode its level order. */
    static Object plain(Object o) throws ReflectiveOperationException {
        if (o == null || o instanceof String || o instanceof Boolean) return o;
        if (o instanceof Character) return String.valueOf(o);
        if (o instanceof Double || o instanceof Float) return ((Number) o).doubleValue();
        if (o instanceof Number) return ((Number) o).longValue();
        if (o.getClass().isArray()) {
            if (o instanceof char[]) {
                List<Object> out = new ArrayList<>();
                for (char ch : (char[]) o) out.add(String.valueOf(ch));
                return out;
            }
            List<Object> out = new ArrayList<>();
            for (int i = 0, n = Array.getLength(o); i < n; i++) out.add(plain(Array.get(o, i)));
            return out;
        }
        if (o instanceof Iterable) {
            List<Object> out = new ArrayList<>();
            for (Object x : (Iterable<?>) o) out.add(plain(x));
            return out;
        }
        Class<?> c = o.getClass();
        if (c.getClassLoader() == loader && c.getSimpleName().equals("ListNode")) {
            Field val = c.getDeclaredField("val"), next = c.getDeclaredField("next");
            val.setAccessible(true);
            next.setAccessible(true);
            List<Object> out = new ArrayList<>();
            for (Object n = o; n != null && out.size() < 100000; n = next.get(n)) out.add(plain(val.get(n)));
            return out;
        }
        if (c.getClassLoader() == loader && c.getSimpleName().equals("TreeNode")) {
            Field val = c.getDeclaredField("val"), left = c.getDeclaredField("left"), right = c.getDeclaredField("right");
            for (Field f : new Field[] {val, left, right}) f.setAccessible(true);
            List<Object> out = new ArrayList<>();
            ArrayDeque<Object[]> queue = new ArrayDeque<>();
            queue.add(new Object[] {o});
            while (!queue.isEmpty() && out.size() < 200000) {
                Object n = queue.poll()[0];
                if (n == null) { out.add(null); continue; }
                out.add(plain(val.get(n)));
                queue.add(new Object[] {left.get(n)});
                queue.add(new Object[] {right.get(n)});
            }
            while (!out.isEmpty() && out.get(out.size() - 1) == null) out.remove(out.size() - 1);
            return out;
        }
        return String.valueOf(o);
    }

    static Object numeric(Object x) { return x instanceof Boolean ? (Long) ((Boolean) x ? 1L : 0L) : x; }

    static boolean equal(Object a, Object b) {
        if (a instanceof List || b instanceof List) {
            if (!(a instanceof List) || !(b instanceof List)) return false;
            List<?> la = (List<?>) a, lb = (List<?>) b;
            if (la.size() != lb.size()) return false;
            for (int i = 0; i < la.size(); i++) if (!equal(la.get(i), lb.get(i))) return false;
            return true;
        }
        Object na = numeric(a), nb = numeric(b);
        if (na instanceof Number && nb instanceof Number) {
            if (na instanceof Long && nb instanceof Long) return ((Long) na).longValue() == (Long) nb;
            double x = ((Number) na).doubleValue(), y = ((Number) nb).doubleValue();
            if (Double.isNaN(x) && Double.isNaN(y)) return true;
            return Math.abs(x - y) <= ATOL + RTOL * Math.abs(y);
        }
        return Objects.equals(a, b);
    }

    /** is_same_list / is_same_tree: values exactly equal, shape included. */
    static boolean strictEqual(Object a, Object b) { return Objects.equals(a, b); }

    static int pyCompare(Object a, Object b) {
        if (a instanceof List && b instanceof List) {
            List<?> la = (List<?>) a, lb = (List<?>) b;
            for (int i = 0; i < Math.min(la.size(), lb.size()); i++) {
                int c = pyCompare(la.get(i), lb.get(i));
                if (c != 0) return c;
            }
            return Integer.compare(la.size(), lb.size());
        }
        if (a instanceof String && b instanceof String) return ((String) a).compareTo((String) b);
        Object na = numeric(a), nb = numeric(b);
        if (na instanceof Number && nb instanceof Number) return Double.compare(((Number) na).doubleValue(), ((Number) nb).doubleValue());
        throw new IllegalArgumentException("cannot order " + Json.show(a) + " and " + Json.show(b));
    }

    // ------------------------------------------------------------- plumbing

    static final class Row {
        final boolean passed; final String actual, expected, error;
        Row(boolean passed, String actual, String expected, String error) { this.passed = passed; this.actual = actual; this.expected = expected; this.error = error; }
    }

    static final class CappedStream extends OutputStream {
        final OutputStream out; int written; boolean cut;
        CappedStream(OutputStream out) { this.out = out; }
        @Override public synchronized void write(int b) throws IOException { write(new byte[] {(byte) b}, 0, 1); }
        @Override public synchronized void write(byte[] b, int off, int len) throws IOException {
            if (cut) return;
            if (written + len > OUTPUT_LIMIT) {
                out.write(b, off, Math.max(0, OUTPUT_LIMIT - written));
                out.write("\n… output truncated\n".getBytes(StandardCharsets.UTF_8));
                cut = true;
                return;
            }
            written += len;
            out.write(b, off, len);
        }
        @Override public void flush() throws IOException { out.flush(); }
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> asMap(Object o) { return (Map<String, Object>) o; }

    @SuppressWarnings("unchecked")
    static List<Object> asList(Object o) { return (List<Object>) o; }

    static final class Json {
        final String s; int i;
        Json(String s) { this.s = s; }

        static Object parse(String s) {
            Json p = new Json(s);
            Object v = p.value();
            p.ws();
            if (p.i != s.length()) throw new IllegalArgumentException("trailing JSON at " + p.i);
            return v;
        }

        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        Object value() {
            ws();
            char c = s.charAt(i);
            if (c == '{') {
                Map<String, Object> m = new LinkedHashMap<>();
                i++; ws();
                if (s.charAt(i) == '}') { i++; return m; }
                while (true) {
                    ws();
                    String k = string();
                    ws(); i++; // ':'
                    m.put(k, value());
                    ws();
                    if (s.charAt(i++) == '}') return m;
                }
            }
            if (c == '[') {
                List<Object> l = new ArrayList<>();
                i++; ws();
                if (s.charAt(i) == ']') { i++; return l; }
                while (true) {
                    l.add(value());
                    ws();
                    if (s.charAt(i++) == ']') return l;
                }
            }
            if (c == '"') return string();
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            if (s.startsWith("null", i)) { i += 4; return null; }
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            String num = s.substring(start, i);
            if (num.matches("-?\\d+")) return Long.parseLong(num);
            return Double.parseDouble(num);
        }

        String string() {
            StringBuilder sb = new StringBuilder();
            i++; // opening quote
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                char e = s.charAt(i++);
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16)); i += 4; break;
                    default: sb.append(e);
                }
            }
        }

        static String quote(String s) {
            StringBuilder sb = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"': sb.append("\\\""); break;
                    case '\\': sb.append("\\\\"); break;
                    case '\n': sb.append("\\n"); break;
                    case '\r': sb.append("\\r"); break;
                    case '\t': sb.append("\\t"); break;
                    default:
                        if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                        else sb.append(c);
                }
            }
            return sb.append('"').toString();
        }

        static String show(Object v) {
            if (v == null) return "null";
            if (v instanceof String) return quote((String) v);
            if (v instanceof List) {
                StringBuilder sb = new StringBuilder("[");
                boolean first = true;
                for (Object x : (List<?>) v) { if (!first) sb.append(','); first = false; sb.append(show(x)); }
                return sb.append(']').toString();
            }
            return String.valueOf(v);
        }
    }
}
