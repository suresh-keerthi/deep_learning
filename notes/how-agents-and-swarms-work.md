# How Agents, and Then Swarms of Agents, Actually Work

*A computer-science-first explanation. Written September 2026; product details (tool names, model names, feature flags) change fast, so treat those as a snapshot and the underlying ideas as the durable part.*

---

## 0. The whole thing in one paragraph

An LLM is a function from text to text. A computer program is also just text (source code, shell commands, JSON, config files). So if you put a **loop** around the LLM, let its output text be **parsed into actions** (run this command, edit this file, click here), execute those actions with ordinary software, and **feed the results back in as more text**, you get an **agent**. A **swarm** is just many such loops running at once, with one of them (or a shared file) handing out work and collecting results. There is no new kind of magic at any level. It's a text-in/text-out model, a `while` loop, and a set of tools.

---

## 1. Building blocks (the premises)

1. **An LLM generates text.** Any text format you can describe: prose, Python, SQL, JSON, shell commands, HTML, coordinates like `[512, 742]`.
2. **A computer program is text.** It is either *compiled* (source → machine code, then the CPU runs it) or *interpreted* (an existing program such as `python` or `node` reads the text and executes it).
3. **Humans use computers in two ways:**
   - **Write code** to make the machine do something new (a minority of people).
   - **Operate software someone else wrote** (browsers, spreadsheets, email, ERP systems, admin panels). This is what most people do, and they didn't write any of that software.
4. **An LLM alone can only emit text.** It can't run anything. But *you* can write a small program (the **harness**) that reads the LLM's text, does something real with it, and hands the outcome back.

### What an agent replaces

| Before | With an agent |
|---|---|
| A human writing code to automate or build things | The model writes, runs, and fixes the code |
| A human clicking through existing software, making a judgment call at every step | The model looks at the state, decides, and operates the software |

The reason this works now, and didn't in 2022, is that models became good enough at (a) producing *valid, precise* structured output, (b) noticing when an action failed and adapting, and (c) staying coherent across dozens or hundreds of steps.

---

## 2. Anatomy of an agent

An agent is four things:

```
┌───────────────────────────────────────────────────────────┐
│                        HARNESS (a normal program)         │
│                                                           │
│   context (messages so far) ──►  LLM  ──► output text     │
│          ▲                                   │            │
│          │                                   ▼            │
│    tool results  ◄──── execute ◄──── parse "tool calls"   │
│   (stdout, files,      (shell, file I/O,     (JSON blocks │
│    screenshots, HTTP)   HTTP, mouse, ...)     in output)  │
└───────────────────────────────────────────────────────────┘
```

1. **The model**: fixed weights, stateless. It remembers nothing between calls.
2. **The context**: the list of messages you send every time (system prompt, user request, every previous action and result). This *is* the agent's short-term memory.
3. **Tools**: named functions the model may request, each with a description and parameter schema. Examples: `bash`, `read_file`, `str_replace`, `web_search`, `screenshot`, `left_click`.
4. **The loop** (in the harness): call the model, check whether it asked for tools, run them, append the results, call the model again. Stop when the model answers without asking for a tool.

The entire core of an agent is roughly this:

```python
def run_agent(task, tools, max_steps=50):
    messages = [{"role": "user", "content": task}]
    for _ in range(max_steps):                     # safety cap
        reply = llm(messages, tools=tools)         # 1. model thinks, emits text + tool requests
        messages.append({"role": "assistant", "content": reply.content})

        calls = [b for b in reply.content if b.type == "tool_use"]
        if not calls:                              # 2. no tool requested -> model considers itself done
            return reply.text

        results = []
        for c in calls:                            # 3. run each requested action for real
            try:
                out = TOOLS[c.name](**c.input)     #    e.g. subprocess.run(["bash","-c",cmd])
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": out})
            except Exception as e:
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": results})   # 4. feed reality back in
```

Everything else in the agent world (frameworks, "skills", MCP servers, memory systems, planners) is refinement of this loop: what tools exist, how the context is managed, how safety is enforced.

**Key insight:** the model never *does* anything. It *requests*. Your harness decides whether to comply. That single fact is where all permissions, sandboxing, and human-approval logic lives.

---

## 3. Use case 1: Agents that write code

### 3.1 The workflow, in plain terms

- You prompt → it writes code.
- You point out a problem → it edits the code.
- You repeat until you're happy.

That is the "chat" version. The **agent** version removes you from the inner loop: the model writes code, *runs it*, reads the errors or test output, and fixes its own mistakes before ever showing you anything.

### 3.2 The tools a coding agent typically has

| Tool | What it does | What really happens |
|---|---|---|
| `read_file` / `view` | Load a file or directory listing into context | `open` + `read` syscalls; result text is appended to the conversation |
| `str_replace` / `create_file` | Edit or create files | Read file, replace a unique string, write back |
| `bash` | Run any shell command | A shell process is spawned; stdout/stderr and exit code come back as text |
| `grep` / `glob` / search | Find things in a big repo without loading it all | Runs `ripgrep`/`find`, returns only matches |
| `web_search` / `web_fetch` | Look up docs, errors, APIs | HTTP requests |
| Spawn subagent | Delegate a subtask (see Section 5) | Starts another agent loop with a fresh context |

### 3.3 Under the hood, from a CS perspective

**a) The model's "action" is just a JSON blob.**
The model emits something like:
```json
{"type": "tool_use", "name": "bash", "input": {"command": "pytest -x tests/"}}
```
The API returns that as structured data with `stop_reason: "tool_use"`. The harness parses it. Nothing has executed yet.

**b) The harness executes it with ordinary OS facilities.**
For the `bash` tool, a typical harness does this:

```python
proc = subprocess.Popen(["bash", "-c", cmd], stdout=PIPE, stderr=STDOUT,
                        cwd=workdir, env=env, preexec_fn=limits)
out, _ = proc.communicate(timeout=120)
return f"exit={proc.returncode}\n{out.decode()[-20000:]}"    # truncate; context is precious
```

At the kernel level, that is roughly these system calls:

| Step | Syscalls (Linux) | Meaning |
|---|---|---|
| Create pipes for output | `pipe2` | Kernel buffers connecting parent ↔ child |
| Spawn the child | `clone` / `fork` (or `vfork`/`posix_spawn`) | New process |
| Become `bash` | `execve("/bin/bash", ...)` | Replace the child's memory image with the program |
| `bash` runs `pytest` | another `fork` + `execve` | `pytest` is itself a Python interpreter reading your test files |
| Program reads/writes files | `openat`, `read`, `write`, `close`, `stat` | File access |
| Program prints | `write(1, ...)` / `write(2, ...)` | Goes into the pipe |
| Harness collects | `read` on the pipe, `wait4` | Gets output and exit status |
| Timeout hit | `kill` (SIGTERM/SIGKILL) | Stops runaway commands |
| Done | `exit_group` | Process ends |

You can see this yourself: `strace -f -e trace=process,file bash -c "python -c 'print(1)'"`.

Many agents keep a **persistent shell** (using a pseudo-terminal via `forkpty`/`openpty`) so that `cd`, environment variables, and virtualenv activation survive between commands, the way they would in a real terminal.

**c) The sandbox is also just OS features.**
Well-built harnesses run all this inside a container or VM. On Linux, a "container" is a normal process wrapped with:
- **namespaces** (its own view of PIDs, filesystem, network),
- **cgroups** (CPU/memory limits),
- **seccomp** (a filter blocking dangerous syscalls),
- and network allowlists.

That's why an agent can `rm -rf` inside its sandbox without harming your machine.

**d) Interpreted vs compiled doesn't matter to the agent.**
- *Interpreted* (Python, JS, shell): write file → `python file.py` → read output.
- *Compiled* (C, Rust, Go): write file → `gcc`/`cargo build` → read compiler errors (great feedback!) → run binary → read output.
In both cases the agent's world is just "text in, command, text out."

**e) Context management is the hard engineering problem.**
Every tool result gets appended, so the context grows. Harnesses cope by:
- truncating long outputs,
- searching instead of reading whole repos,
- **compaction** (asking the model to summarize old history and replacing it),
- writing notes/plans to files the agent can re-read (external memory),
- delegating to subagents whose noisy work never enters the main context.

### 3.4 Why verifiable targets are a superpower

If the goal has a measurable check (tests pass, benchmark got faster, loss went down, linter is clean), the agent doesn't need you to judge anything. It can loop by itself:

```
propose change → run checker → read the number → keep or revert → repeat
```

A clean public example is **Karpathy's `autoresearch`** (March 2026). It's deliberately minimal: one file the agent may edit (`train.py`), one fixed evaluator it may not touch (`prepare.py`), one metric (`val_bpb`, validation bits per byte, lower is better), a fixed ~5-minute training budget per experiment, and a `program.md` where a human writes the instructions. The agent edits the code, trains, reads the metric, keeps the change if it improved and reverts (via git) if not, and repeats: roughly 12 experiments an hour, about 100 overnight.

Why this works so well:
- **A trusted, unmodifiable judge.** The agent can't game the metric because it can't edit the evaluator.
- **Fixed budget** so experiments are comparable.
- **Cheap undo** (git) so bad ideas cost nothing.
- **The model's breadth.** It has read a huge amount of ML literature and code, so its "random" ideas are informed ones, and it doesn't get bored or tired.

Caveats worth knowing: a single short run is noisy (people building on this have added re-runs with multiple seeds to avoid keeping lucky flukes), results are hardware-specific, and an unattended agent that reads outputs from files it can influence is a prompt-injection surface.

### 3.5 Where coding agents fail
- Confidently "fixing" symptoms rather than causes.
- Weak or missing tests → the loop optimizes the wrong thing.
- Reward hacking (making a test pass by special-casing it).
- Losing the plot in very long sessions if context isn't managed.
- Large, cross-cutting design decisions still benefit from human review.

---

## 4. Use case 2: Agents that operate existing software ("computer use")

### 4.1 First, what can it do "just by prompt"?

By **text** (typed) or **audio** (spoken, see 4.6), you can hand it goals like:

- "Find the three cheapest flights next Friday and put them in a comparison sheet."
- "Read this folder of invoices and fill in the accounting spreadsheet."
- "Test our web app's checkout flow and tell me what breaks."
- "Log into the admin panel and export last month's orders."
- "Reorganize my downloads folder by project."
- "Book a table for four; ask me before you pay."
- "Summarize my unread email and draft replies."
- "Research X across 30 sources and write a cited brief."

Notice that these all require *decisions in the middle* (which result is best? which button? is this the right form?), which is exactly what traditional scripts and RPA tools are bad at.

### 4.2 A spectrum of ways to operate software

A good agent tries the **cheapest, most reliable interface first** and falls back to the most general one last.

| Rank | Interface | How the agent uses it | Reliability / speed |
|---|---|---|---|
| 1 | **Direct APIs / SDKs / connectors (e.g., MCP servers)** | Calls `create_event(...)`, `search_issues(...)` as tools | Best. Structured, fast, cheap |
| 2 | **Command line & files** | Runs `git`, `curl`, `ffmpeg`, `sqlite3`; edits files directly (e.g., a library that writes `.xlsx`/`.docx` without opening Word or Excel) | Excellent |
| 3 | **Browser via DOM / DevTools protocol** | Reads page structure and acts on elements ("click element #submit") | Good; page-aware, no pixel guessing |
| 4 | **Accessibility tree** (OS-level "screen reader" APIs: AT-SPI on Linux, UI Automation on Windows, AXUIElement on macOS) | Gets a structured list of buttons/fields with names and positions | Good where apps support it |
| 5 | **Pixels: screenshots + mouse + keyboard ("computer use")** | Looks at an image, outputs coordinates and keystrokes | Most general (works on *any* GUI), but slowest and least precise |

This is the answer to "what does the agent replace?": it replaces the human at whichever rung is available, and **computer use (rung 5)** is the universal fallback for software with no API.

### 4.3 How computer use works, step by step

Here is the loop as Anthropic's API describes it (the toolset is named `computer_toolset_20260801` at the time of writing):

1. **You (the developer)** send a request declaring the computer toolset plus your task ("Save a picture of a cat to my desktop").
2. **The model responds** with tool-use blocks, e.g. `screenshot`, `left_click` at `[512, 742]`, `type` "pictures of cats". Often several in one turn (a **batch**), ending with a `screenshot` so it can see the outcome. `stop_reason` is `tool_use`.
3. **Your harness executes them, in order,** against a real (usually virtual) desktop. If one fails, it stops and reports the rest as "not executed."
4. **Your harness sends back results:** a short "OK" for clicks/typing, an **image** for screenshots.
5. **Repeat** until the model replies with plain text and no more tool calls.

The model doesn't connect to the computer. Your code sits in the middle. That means you control what's reachable, what needs approval, and what gets logged.

**The action vocabulary** (17 member tools in the current toolset) covers everything a human hand can do:

| Category | Actions |
|---|---|
| See | `screenshot`, `zoom` (read a small region at full resolution), `cursor_position` |
| Point & click | `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click` (optionally with modifier keys like ctrl/shift) |
| Drag | `left_click_drag`, or the finer `left_mouse_down` / `left_mouse_up` |
| Move / scroll | `mouse_move` (hover), `scroll` (direction + amount) |
| Keyboard | `type` (literal text), `key` (`Return`, `ctrl+s`, `alt+Tab`, with optional repeat), `hold_key` |
| Timing | `wait` |

**How does it know *where* to click?** The model looks at the screenshot (a grid of pixels) and outputs pixel coordinates. Training it to do this accurately was a major research effort. The coordinates are in the pixel space of the screenshot you returned, so if your harness shrinks a big screen to fit image limits, it must scale coordinates back up before clicking. Typical recommended screen sizes are around 1024×768 to 1280×800, and each screenshot costs on the order of a thousand-plus tokens, which is why long GUI sessions get expensive and slow.

### 4.4 Under the hood: what programs run, what system calls happen

Let's trace a single action: the model says `left_click` at `[512, 742]`.

**The harness** (Python, say) receives the block and runs something like:

```python
subprocess.run(["xdotool", "mousemove", "512", "742", "click", "1"])
```

Then to produce the next screenshot:

```python
subprocess.run(["scrot", "-o", "/tmp/shot.png"])     # or ImageMagick's `import -window root`
png = open("/tmp/shot.png", "rb").read()             # then base64 and send back to the model
```

Now what does *`xdotool`* do at the OS level?

```
python harness ── fork/execve ──► xdotool
                                     │  socket(AF_UNIX) + connect("/tmp/.X11-unix/X99")
                                     ▼
                          X server (the display server)
                                     │  XTEST extension: "pretend the mouse moved/clicked"
                                     ▼
                     X server's normal input pipeline
                                     │  sends ButtonPress/ButtonRelease events over
                                     │  each app's own socket to the window under the pointer
                                     ▼
                         Firefox / LibreOffice / any GUI app
                              handles it exactly like a real click,
                              redraws its window into the X server's framebuffer
```

Concretely:
- **`xdotool`** is a small C program that talks the **X11 protocol** over a **Unix domain socket** (`socket`, `connect`, then `sendmsg`/`recvmsg` or `read`/`write`). It uses the **XTEST** extension, which lets a client inject fake input events.
- The **X server** treats these as if they came from a real device, and delivers them to whichever window is under the cursor. To the app, it is indistinguishable from a human.
- **Screenshots** are the reverse: the tool asks the X server for the framebuffer contents (an X `GetImage` request, often via shared memory) and saves them as PNG.
- **Typing** works the same way: `xdotool type "cats"` maps each character to a keycode and injects key-down/key-up events.

On other platforms the mechanism differs but the pattern doesn't: *inject synthetic input events + capture the screen*.

| Platform | Typical injection | Typical capture |
|---|---|---|
| Linux/X11 | XTEST (`xdotool`) | X `GetImage` (`scrot`, `import`) |
| Linux kernel level | write events to `/dev/uinput` (creates a virtual keyboard/mouse device seen by the kernel's input layer) | n/a |
| Linux/Wayland | Portals / libei (Wayland deliberately restricts synthetic input for security, so tooling is more limited) | PipeWire/portal screencast |
| macOS | Quartz Event Services (`CGEventPost`) or `cliclick` | `CGDisplayCreateImage` / `screencapture` |
| Windows | `SendInput` Win32 API | GDI / DXGI screen capture |

### 4.5 "But how does it work without the program appearing on the monitor?"

Because **the monitor is only the last, optional step**. A GUI application never "draws on the monitor." It draws into a **framebuffer**, a block of memory, via the display server. A physical monitor is merely one consumer of that memory. Take the monitor away and everything still works, as long as *something* provides the display server and framebuffer.

There are four common ways this happens:

**1. A virtual display (Xvfb, "X virtual framebuffer").**
`Xvfb :99 -screen 0 1280x800x24 &` starts a complete X server whose "screen" exists only in RAM. Apps launched with `DISPLAY=:99 firefox &` render normally into that memory. `xdotool` and `scrot` connect to `:99` exactly as before. Nobody is watching, but the pixels are real. If a human wants to peek, they can attach a VNC/noVNC viewer to the same framebuffer. Anthropic's reference implementation runs Xvfb plus a lightweight window manager and panel, with Firefox, LibreOffice, a text editor and a file manager, all inside a Docker container.

**2. Headless browsers.**
`chromium --headless` runs the full browser engine with no window at all. It's controlled over the **Chrome DevTools Protocol** (JSON over a WebSocket or pipe): "navigate to this URL," "click the node with this selector," "capture screenshot." The agent gets either DOM text or rendered screenshots.

**3. No GUI needed whatsoever.**
Often the "software" can be operated directly. Instead of opening Excel, a script reads/writes the `.xlsx` file format (a zip of XML). Instead of opening a web dashboard, it calls the service's HTTP API. Instead of opening a mail client, it speaks IMAP/SMTP or a connector. These are all faster and more reliable than clicking pixels.

**4. Your real desktop, in the background.**
In desktop assistants that control your actual machine, the app is on your real screen, and the agent uses the same screenshot + synthetic input approach (or the accessibility APIs) that a remote-desktop tool would. Some tools can even target a specific window without stealing focus (on X11, `xdotool` can send events to a window ID).

### 4.6 A full worked example

**Prompt (typed or spoken):** *"Find the current price of a Raspberry Pi 5 on the manufacturer's site and paste it into notes.txt on the desktop."*

What happens:

```
[harness] starts Xvfb :99, window manager, Firefox; sends task + tools to model

MODEL  → screenshot
HARNESS→ (xdotool/scrot)  returns PNG of empty desktop with a Firefox icon

MODEL  → left_click [40, 700]                    # the Firefox icon
MODEL  → wait 3
MODEL  → screenshot
HARNESS→ PNG: browser open, blank page

MODEL  → left_click [640, 60]                    # address bar
MODEL  → type "raspberrypi.com"
MODEL  → key "Return"
MODEL  → wait 3
MODEL  → screenshot
HARNESS→ PNG: home page loaded

MODEL  → (reads pixels) "Products menu at top" → left_click [410, 110]
    ... a few more click/screenshot rounds ...
MODEL  → zoom [700, 300, 1000, 380]             # small price text was blurry
HARNESS→ full-res crop: "$60.00"

MODEL  → bash  `echo "Raspberry Pi 5 (4GB): $60.00" > ~/Desktop/notes.txt`   # cheaper than GUI editing!
HARNESS→ subprocess.run(["bash","-c", ...]) → "OK"

MODEL  → "Done. The price shown was $60.00; I saved it to notes.txt."
```

Notice the **hybrid**: the model used the GUI only where it had to and switched to a shell for the file write. That is normal: computer use is normally declared *alongside* bash and a text editor tool.

### 4.7 Voice/audio prompts

The agent core is text, so audio is a front-end layer:

```
microphone → speech-to-text (ASR) → text prompt → AGENT LOOP → text answer → text-to-speech → speaker
```

Some newer models process audio natively (no separate transcription step), but the *action* side is unchanged: the result is still tool calls. In practice, voice adds: interruption handling, low-latency streaming, and confirmation prompts spoken back to you ("I'm about to send this email to 40 people. Go ahead?").

### 4.8 Honest limitations of computer use
- **Latency**: each step is a model call plus a screenshot, so it's much slower than a human at simple things. Best for tasks where speed doesn't matter (overnight jobs, background research, QA).
- **Vision errors**: mis-clicking small or crowded UI, misjudging coordinates.
- **Tricky widgets**: dropdowns, scrollbars, spreadsheets; keyboard shortcuts often work better than mouse.
- **Assumes success**: without a "take a screenshot after each step and verify" instruction, models sometimes assume an action worked.
- **Cost**: screenshots are token-heavy; harnesses prune or clear old ones and cache prompts.
- **Security** (see Section 6).

---

## 5. From one agent to a swarm

### 5.1 Why more than one agent?

A single agent has three hard limits:

1. **Context window**: one agent's memory is finite, and every tool result eats it.
2. **Sequential speed**: one loop does one thing at a time.
3. **Focus**: a context stuffed with unrelated material degrades attention.

The fix is the same one used everywhere in computing: **decompose, parallelize, and isolate**.

### 5.2 What a "subagent" actually is

Not a new kind of thing. A subagent is **the same loop, started fresh by a tool call**:

```python
def spawn_subagent(task_description, tools):          # exposed to the parent as a tool
    return run_agent(task_description, tools)          # new messages list = new, clean context
```

The parent gives it a self-contained task; the subagent works in its own context window (which may fill with thousands of tokens of search results and dead ends); it returns a **short summary**; the parent's context only grows by that summary. The parent never sees the mess.

### 5.3 The most common swarm shape: orchestrator-worker

```
                     ┌────────────────────┐
     user ─────────► │  LEAD / ORCHESTRATOR│  plans, splits the task, decides how many workers
                     └───┬──────┬──────┬───┘
            (parallel)   │      │      │
                   ┌─────▼┐ ┌───▼──┐ ┌─▼────┐
                   │ W1   │ │ W2   │ │ W3   │   each: own context, own tools, one narrow job
                   └─────┬┘ └───┬──┘ └─┬────┘
                         └──────┼──────┘   summaries flow back up
                     ┌──────────▼─────────┐
                     │ LEAD synthesizes;   │  may launch another wave if gaps remain
                     └────────────────────┘
```

Anthropic's Research feature is the best-documented example (published June 2025):
- A **lead agent** plans and spawns roughly **3–5 subagents in parallel**; each subagent also calls several tools in parallel. Anthropic reports this cut research time by up to 90% on complex queries.
- Subagents explore independent threads (different sources, different angles) and report back; a **separate citation-checking pass** ties claims to sources.
- On Anthropic's internal research eval, a lead on Claude Opus 4 with Claude Sonnet 4 subagents beat single-agent Opus 4 by **90.2%**.
- Cost reality: agents use about **4×** the tokens of a normal chat, and multi-agent systems about **15×**. That is the tradeoff: you're buying breadth and speed with tokens. It pays off for high-value, breadth-heavy questions, not for simple ones.

**The lead's delegation prompt is everything.** Early versions of the system misbehaved: spawning dozens of subagents for trivial questions, duplicating each other's work, hunting for sources that don't exist. The fixes were mostly prompt engineering: give every subagent an explicit objective, output format, tool guidance, and **boundaries** (what *not* to research because another worker owns it), plus **scaling rules** embedded in the prompt (roughly: simple lookup → 1 agent with a handful of tool calls; comparison → 2–4 subagents; large research → 10+ with clearly divided responsibilities).

### 5.4 Other coordination patterns

| Pattern | How it works | Good for |
|---|---|---|
| **Orchestrator-worker** (above) | Central lead delegates and merges | Research, decomposable tasks |
| **Pipeline / assembly line** | A → B → C, each specialized (planner → coder → reviewer → tester) | Well-defined workflows |
| **Fan-out / fan-in (map-reduce)** | Same instruction applied to N items in parallel, then aggregated | Processing 300 files, 100 documents, many candidates |
| **Debate / ensemble / voting** | Several agents solve the same problem independently, then compare or vote | Reducing errors on hard reasoning |
| **Generator + critic** | One writes, another reviews and sends it back | Code review, writing quality |
| **Shared workspace ("blackboard")** | Agents read/write a common task list or files and pick up work themselves | Long, evolving projects |
| **Parallel experimenters** | N agents each try a different idea against the same metric; keep the winner | Optimization, autoresearch-style loops run in parallel |

### 5.5 "Agent teams": swarms that talk to each other

Simple subagents only report **up** to their parent and can't talk to each other. A newer style is the **agent team** (Claude Code has this as an experimental, off-by-default feature): one session acts as **team lead**; teammates are **separate full sessions**; and coordination uses:

- a **shared task list** (a file/queue on disk) with **dependency tracking**: tasks unblock automatically when prerequisites finish, and idle teammates can **claim** the next available task;
- a **mailbox / messaging system** so teammates can send each other findings directly;
- optionally, file locking for concurrent writes.

Conceptually, the task list is just a **shared data structure on disk**: the same idea as a job queue with a dependency graph (a DAG scheduler), where the workers happen to be LLMs.

Other vendors ship the same idea at larger scale. For example, Moonshot's Kimi advertises an "Agent Swarm" that decomposes a task and dispatches up to about 300 subagents in parallel.

### 5.6 The real engineering problems in swarms

**1. Isolation vs. sharing.**
The big design question is what each agent needs to know about the others. Anthropic's research design bets on "almost nothing" (independent threads). The opposing argument (notably Cognition's "Don't Build Multi-Agents") is that parallel agents make independent decisions and *independent decisions on the same problem can conflict*. Rule of thumb: **parallelize reading and exploring; be careful about parallel writing and deciding.**

**2. File collisions.**
Two agents editing the same file will clobber each other. Standard fix: give each coder its own **git worktree** (a separate checkout on its own branch) so parallel edits can't collide, then merge branches afterward. When worktrees aren't used (as in current agent teams), you must **partition ownership**: each teammate owns different files.

**3. Coordination overhead.**
More agents means more messages, more duplicate discovery, and more tokens. Past a point, adding agents slows things down.

**4. Compounding errors.**
If Worker 2 builds on Worker 1's *wrong* summary, the mistake propagates. Mitigations: verification agents, tests as ground truth, requiring evidence (file paths, citations) in summaries, and a lead that spot-checks.

**5. Runaway cost and loops.**
Always set budgets: max steps, max agents, max dollars, wall-clock timeouts.

**6. Evaluation is hard.**
Agents can take different valid paths to the same result, so you judge **outcomes** (and sample transcripts), not exact step sequences. Small evals (about 20 realistic queries) plus an LLM judge plus human review catch a lot early.

**7. State that survives context limits.**
Long-running swarms write plans and progress to files or a memory store so an agent that runs out of context can resume rather than restart.

### 5.7 When to use what

```
Simple question or small edit           →  single call / no agent
Multi-step task, one coherent thread    →  single agent with tools
Need to explore several independent     →  orchestrator + parallel subagents
   directions / read a lot                 
Big project, parallel workstreams,      →  agent team (shared task list) + worktrees
   workers must coordinate
Clear metric, cheap experiments         →  optimization loop, optionally N in parallel
```

Start simple. Move up only when a single agent demonstrably hits a wall (context, speed, or focus). Multi-agent is a cost-and-complexity choice, not a status upgrade.

---

## 6. Safety and security (short but critical)

- **Prompt injection.** The agent reads untrusted text (web pages, emails, PDFs, screenshots), and the model can mistake instructions embedded in that data for instructions from you. Defenses: isolate the agent from secrets, allowlist network domains, run in a VM/container with least privilege, and use classifiers plus training to resist injections. None of these are perfect alone.
- **Human-in-the-loop for consequences.** Require approval for payments, sending messages, deleting data, accepting terms. The harness is where to enforce this, ideally checked *before each action runs* (batches can complete multi-step actions in one turn).
- **Least privilege.** Give an agent only the tools and credentials the task needs. In swarms, this matters more: one compromised worker shouldn't reach everything.
- **Auditability.** Log every tool call and result. You'll need it to debug and to trust the system.
- **Unattended loops** (overnight optimization, background swarms) magnify all of the above; scope them narrowly and cap them.

---

## 7. Cheat sheet: the mental model

| Concept | One-line definition | Real implementation |
|---|---|---|
| LLM | Text → text function, stateless | An API call |
| Tool | A named action the model can request | A function in your harness |
| Agent | LLM + tools + loop | `while` loop calling the API and running requested tools |
| Context | The agent's working memory | The `messages` list resent each call |
| Coding agent | Agent whose tools are file edit + shell | `fork`/`execve`, file syscalls, pipes |
| Computer-use agent | Agent whose tools are screenshot + mouse + keyboard | Screenshot capture + synthetic input (XTEST/`xdotool`, `uinput`, OS APIs) |
| Headless | GUI apps without a physical monitor | Virtual framebuffer (Xvfb) or headless browser (DevTools Protocol) |
| Subagent | Agent started by another agent's tool call | Fresh `messages` list; returns a summary |
| Swarm | Many agents working concurrently | Orchestrator + workers, or shared task queue |
| Worktree | Isolated git checkout per agent | `git worktree add` |
| Harness | The program around the model | Where safety, budgets and approvals live |

**The one-sentence version:** *An agent is a language model in a loop that turns its text into actions and the world's response back into text; a swarm is many of those loops splitting up a job, with the hard parts being isolation, coordination, verification, and cost.*

---

## 8. Sources and further reading

- Anthropic Engineering, *How we built our multi-agent research system*: https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic Engineering, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents
- Claude Platform Docs, *Computer use tool* (actions, agent loop, environment, limitations): https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
- Anthropic reference implementation (Docker + Xvfb + tools + loop): https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo
- Claude Code Docs, *Run agents in parallel* (subagents, agent teams, worktrees): https://code.claude.com/docs/en/agents
- Claude Code Docs, *Orchestrate agent teams*: https://code.claude.com/docs/en/agent-teams
- Karpathy, `autoresearch`: https://github.com/karpathy/autoresearch
- Anthropic Engineering, *Effective harnesses for long-running agents*: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- Cognition, *Don't Build Multi-Agents* (the counter-argument on parallel decision-making)

*Notes on confidence: the OS-level details in Section 4.4 (XTEST, Unix sockets, `GetImage`, `uinput`, syscall names) are standard Linux/X11 behavior and are how the common open-source reference tools work; a specific product's internals may use different backends (e.g., accessibility APIs or native OS event APIs). Vendor claims about performance (e.g., 90.2%, 15× tokens) come from Anthropic's own published evaluations.*
