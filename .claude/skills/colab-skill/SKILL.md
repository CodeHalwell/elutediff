# Skill: Colab Session Operator

Operate Google Colab environments via the `colab` CLI: provision GPU/TPU sessions, run Python/shell on the VM, sync files, and capture work as notebooks.

## When to activate
- Creating or managing TPU/GPU sessions.
- Running Python or shell on a remote Colab VM.
- Syncing files between local and remote.
- Automating environment setup (packages, auth, Drive).
- Exporting session history as a Jupyter notebook.

## Mental model (read this first)
- **A session == a live Jupyter kernel on a rented VM.** `colab new` allocates a billable VM; `colab stop` releases it. Nothing reclaims it automatically except a 24h keep-alive cap, so an unstopped session burns compute units indefinitely.
- **Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same session.** Each invocation reattaches to the *same* kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does **not** shut the kernel down. So imports, variables, and defined functions survive between separate `colab exec` commands. Build up state incrementally; don't re-import everything each call. (`colab stop` and `colab restart-kernel` are what actually reset it.)
- **Default working directory is `/content`.** Every `exec`/`repl`/`run` `cd`s there first; prefer absolute paths (`/content/...`) for file work. For `colab ls/rm/upload/download`, absolute `/content/...` paths work and the default `ls` path is `content` (VM root).
- **`colab` is fire-and-forget.** Each command authenticates, does one thing, and exits. A detached background daemon (spawned by `colab new`) handles keep-alive; you don't manage it.

## Authentication (the #1 thing that blocks agents)
- The global flag is `--auth={adc,oauth2}` and the **default is `adc`** (Application Default Credentials). It must come *before* the subcommand: `colab --auth=adc new -s x`.
- **ADC setup** (most reliable for headless/agent use). The Colab backends need a specific scope set, so re-mint ADC with all four scopes:
  ```bash
  gcloud auth application-default login \
    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
  ```
  (Keep the comma-separated scopes on a single line — splitting them across
  backslash-continued lines folds the leading indentation into the scope values
  and silently corrupts the list.)
  Why all four: `userinfo.email` (session backend `colab.research.google.com`, else 401), `colaboratory` (RuntimeService `colab.pa.googleapis.com` keep-alive, else 403), `openid`+`cloud-platform` (mandated by gcloud itself; it rejects scope lists missing `cloud-platform`).
- **oauth2 setup**: `colab --auth=oauth2 <anything>` triggers a browser consent flow on first use (token cached at `~/.config/colab-cli/token.json`). Requires a client config at `~/.colab-cli-oauth-config.json` (or `-c PATH`). The browser step means it usually needs a human; prefer ADC for agents.
- **Verify auth in one shot**: `colab sessions` (read-only, lists server assignments) or `colab whoami` (hidden debug command: prints the active email, scopes, audience, and expiry). When any call 403s against `colab.pa.googleapis.com`, the cause is almost always a missing scope — `colab whoami` shows it instantly.
- **`colab new` pre-flights the keep-alive RPC** right after allocating. If your token lacks the `colaboratory` scope it unassigns the fresh VM (so you don't leak a billable assignment) and prints the exact remediation. Follow that message rather than retrying blindly.
- **Do NOT confuse `colab auth` with CLI authentication.** `colab auth` injects *VM-side* GCP credentials into the running kernel (so notebook code can call BigQuery/GCS); it is orthogonal to how the CLI itself authenticates. Never suggest "run `colab auth`" to fix a CLI 401/403 — that's a scope/identity problem fixed via the `gcloud` command above.

## Workflow

### Provision
- `colab new -s <name>` (CPU). Add `--gpu A100` or `--tpu v6e1` for accelerators. **Always pass `-s <name>`** — an omitted name is auto-generated as a random 6-hex string, which makes later commands ambiguous.
- Supported `--gpu`: `T4`, `L4`, `G4`, `H100`, `A100`. Supported `--tpu`: `v5e1`, `v6e1`.
- **Gotcha**: an unrecognized `--gpu` value silently falls back to **A100** (which then usually fails the next step). A `400` on `colab new` with an accelerator means no quota/entitlement for it on this account — fall back to `--gpu T4` or omit the flag for CPU.
- Accelerator availability is tier-gated; most accounts can only get CPU. Don't assume a GPU/TPU will allocate.
- **The `--gpu` flag picks a *family*, not an exact card — always verify the real hardware after `new`.** Observed (June 2026): `--gpu A100` allocated a **40 GB** A100-SXM4 (not the 80 GB variant), while `--gpu G4` allocated an **RTX PRO 6000 Blackwell with ~96 GB VRAM, ~177 GB system RAM, 48 vCPU** — effectively the "high-RAM" shape, and a bigger card than the A100 you'd reach for. There is **no CLI flag for "high-RAM" or for the 80 GB A100**; the shape is tied to the GPU family + subscription tier. For a model that needs >40 GB VRAM, `G4` may be the right pick over `A100`. Confirm with one probe right after `new`:
  ```bash
  echo 'import subprocess; print(subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],capture_output=True,text=True).stdout); print("RAM GB:", round(int(open("/proc/meminfo").readline().split()[1])/1024/1024,1))' | colab exec -s <name>
  ```
  (Keep the probe on one line — multi-line snippets piped via `echo` reach Python
  with leading indentation and raise `IndentationError`.)
  A failed allocation is unassigned (not billed), so probing entitlement is safe.

### Execute
- **Preferred**: `colab exec -s <name> -f <script.py>` runs a local script on the remote VM (read locally, sent to the kernel — no manual upload needed).
- **Piped code**: `echo "print(1)" | colab exec -s <name>` or `cat script.py | colab exec -s <name>`.
- **Notebooks**: `colab exec -s <name> -f nb.ipynb` runs each code cell and writes results to `<basename>_output.ipynb` next to the input. A `# @title Foo` first line labels the cell in progress output.
- **CRITICAL: `colab exec -f notebook.ipynb` exits 0 even when cells raise.** It runs *every* cell regardless of errors and saves their tracebacks into the output notebook — a failed cell does **not** abort the run or set a non-zero exit code (unlike `colab run script.py`, which propagates exit codes). Never trust the exit status of a notebook `exec`; **grep the captured output/log for `Traceback`/`Error`/`ModuleNotFoundError`** and for your own success markers before declaring it worked. A cascading `ModuleNotFoundError` across every cell usually means an early install cell failed (e.g. a `pip install ... @ git+...@<branch>` pointing at a deleted/renamed branch — `git checkout` fails silently inside pip).
- **Watch long runs by tailing, not by exit code.** For multi-minute jobs (model downloads, training), background the `exec` (redirect to a log) and poll the log for the next checkpoint marker; `colab status -s <name>` shows `BUSY (exec(...))` with the live cell id.
- **Plots/images**: PNG/JPEG outputs are intercepted. Use `--output-image <path>` on `exec`/`repl` to save to a known location (otherwise a temp path is printed). Inline terminal-image escapes are auto-suppressed when stdout isn't a TTY, so piped/captured output stays clean.
- **Shell**: `echo "cmd" | colab console -s <name>` for batch shell. Console wraps bash in tmux, so even piped output contains terminal-control bytes — filter with `grep -a` for a specific line. `exec` is faster when you don't need a real shell.
- **Never run `colab repl`, `colab console`, `colab auth`, or `colab drivemount` interactively from an agent** — they expect a TTY and will hang. `repl`/`console` accept piped stdin and exit on EOF; `auth`/`drivemount` genuinely require a human at the terminal.

### Ephemeral one-shot jobs (`colab run`)
- `colab run [--gpu T4] [--tpu v6e1] [--keep] [-s NAME] script.py [args...]` = `new` + `exec` + `stop` in one command. It provisions a fresh VM, runs the script with `sys.argv` and `__name__ == "__main__"` set like native `python script.py args`, then tears the VM down (unless `--keep`).
- **Exit codes propagate**: an uncaught exception or `sys.exit(N)` in the script makes `colab run` exit non-zero (CPython semantics: `sys.exit()`/`sys.exit(0)` → 0, `sys.exit(N)` → N, `sys.exit("msg")` → 1).
- **Stream separation**: `colab run` writes its own `[colab] ...` chatter to **stderr** and the script's output to **stdout** — so `colab run job.py > out.txt` captures only the script's stdout. (`colab exec` streams the script's stdout/stderr live to your stdout/stderr.)
- Works as a shebang: `#!/usr/bin/env -S colab run --gpu T4` makes a `chmod +x`'d `.py` a self-contained "rent a GPU, run, clean up" script. After editing CLI behavior, reinstall before testing shebangs — they resolve `colab` via `$PATH`, not the editable install.
- A nonexistent script path exits non-zero **before** allocating a VM (no wasted compute).

### Automate
- `colab auth -s <name>` — VM-side GCP creds, needed before in-VM GCS/BigQuery calls (interactive; not agent-runnable).
- `colab drivemount -s <name> [PATH]` — mounts Drive at `/content/drive` by default (interactive; not agent-runnable).
- `colab install -s <name> pkg1 pkg2` — installs via `uv pip install --system`, falling back to `pip`. Also `colab install -s <name> -r requirements.txt`.

### Inspect & report
- `colab help` (or `colab help <cmd>`) lists/explains commands; the listing is alphabetical.
- `colab sessions` lists server-side assignments and auto-prunes stale local entries. Orphans with no local record show as `[?]`.
- `colab status [-s <name>]` shows hardware, IDLE/BUSY, and last execution.
- `colab log -s <name> [-n 20] [-t TYPE]` shows recent structured events; invaluable when a task fails (keep-alive errors carry the raw `response_body`).
- `colab log -s <name> -o summary.ipynb` exports the session as a notebook (also `.md`, `.txt`, `.jsonl` by suffix).
- `colab url -s <name>` prints a browser URL that attaches the Colab web UI to your existing CLI session instead of allocating a new VM (add `--open` to launch it).
- `colab skill` / `colab readme` print this skill and the README (handy for self-discovery).

## Safety
- **Always `colab stop -s <name>` when done** — idle VMs burn compute units. `colab run` (without `--keep`) self-cleans even if the script errors.
- Local state lives in `~/.config/colab-cli/sessions.json` (settings in `settings.json`, history in `history/*.jsonl`). Don't edit by hand.
- **Isolate parallel/agent runs** with the global `--config <path>` flag to point session state at a scratch file (e.g. `colab --config /tmp/agent.json new -s job`). The keep-alive daemon inherits `--auth` and `--config` automatically.

## Recovery
- "Session not found" / 404 / 401 on exec: the backend pruned the VM. `colab exec`/`repl` detect this and clean up local state automatically — run `colab sessions` and re-create with `colab new`.
- Execution timeout or wedged kernel: `colab restart-kernel -s <name>` (keeps the VM, resets the kernel), or `colab stop` then `colab new`.
- Keep-alive daemon died (`colab log` shows `keep_alive_stopped reason=consecutive_4xx_errors`): almost always the missing `colaboratory` scope — re-auth per the Authentication section.
- `RuntimeError: Connection was lost.` from an in-flight `exec`: expected when you `colab stop` a VM mid-execution (e.g. you've seen the checkpoint you needed and tear down early to save units). It's the websocket dropping, **not** a code/kernel error — don't chase it as a bug.

