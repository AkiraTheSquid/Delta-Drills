/* course-builder/mcp-instructions.js — the "Copy instructions for your AI"
   option under the Courses tab's "Add your own course" row (courses.js).

   A learner pastes this into Claude Code (or any MCP client) and that AI
   builds the course through the repo's content MCP server —
   content-mcp/ (28 tools over the graph, the lesson pages and the drills,
   password-gated writes, the repo's own validators as the gate). The text is
   written FOR the AI: what the server is, how to reach it, the order to build
   in, and the two checks that decide whether the work has landed. It names
   tools by their registered names in content-mcp/content_mcp/ops.py; rename
   one there and update it here.
   🔴 The editing password never goes through the AI: the text has the
   learner log in from their own terminal (`dd-content login`), which writes
   the same session file the MCP server reads — and via `--password -`
   (stdin, from a silent `read`), so the secret is in neither shell history
   nor the process list. */
(function () {
  "use strict";

  const REPO = "https://github.com/AkiraTheSquid/Delta-Drills";

  const TEXT = `I want to add my own course to Delta Drills (https://deltadrills.com), an adaptive practice app. Its course content — the concept graph, the lesson pages and the drills — is edited through the Delta Drills content MCP server, "delta-drills-content". Please build the course with me through that server.

SETUP (once)
1. Clone the repo and start in it: git clone ${REPO} && cd Delta-Drills
2. The repo ships a .mcp.json that registers "delta-drills-content" (command: ./content-mcp/bin/dd-content-mcp). Enable it, and confirm its 28 tools are listed.
3. Reading is open. Writing needs the shared editing password. Do NOT ask me to paste it into this chat: ask me to run this in my own terminal (it prompts silently, keeps the password out of shell history, and starts the same 12-hour session the MCP server uses):
   read -rs -p "Delta Drills editing password: " DD_PW && printf '%s' "$DD_PW" | ./content-mcp/bin/dd-content login --password - ; unset DD_PW
   Then call content_status to confirm writes are unlocked.

HOW TO BUILD THE COURSE
1. Interview me first: the subject, who it is for, what a learner should be able to DO at the end, and what they already know.
2. Call graph_list to see the concepts that exist, and reuse one as a prerequisite wherever it fits instead of writing a duplicate.
3. Propose the concept graph before writing anything: one concept per node, each with its prerequisites, as a short list I can approve. Keep each node to one idea a learner could fail on its own.
4. Once I approve it, add the nodes with graph_add_kc (prerequisites first; the teaching order must stay a linear extension of the prerequisites).
5. Call lesson_authoring_guide, then write each concept's page with lesson_create — four rungs: Lesson, Faded, Solo, Integrated. One worked example per segment; every runnable code fence must assert its own result.
6. Add drills with drill_add. Ids are positional: never delete a drill, retire it with drill_retire.
7. After every concept, run pipeline_check and then pipeline_watchers. A change that has not passed BOTH has not landed — fix what they report before moving on.
8. If something goes badly wrong, backup_status / backup_restore roll the content back to the last snapshot.

Work one concept at a time, show me each page before moving to the next, and never hand-edit questions.json, questions_structured.json or lessons_structured.json — they are generated.`;

  // Clipboard API first; a hidden textarea + execCommand for an insecure
  // origin (plain http on a LAN IP) where navigator.clipboard is undefined.
  const copy = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(TEXT);
        return true;
      }
    } catch (_) { /* fall through to the textarea route */ }
    const ta = document.createElement("textarea");
    ta.value = TEXT;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
    ta.remove();
    return ok;
  };

  window.DDCourseMcpInstructions = { text: () => TEXT, copy };
})();
