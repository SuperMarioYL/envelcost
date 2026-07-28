# envelcost demo tape — the 10-minute happy path from install to first result.
# Rendered to assets/demo.gif by .github/workflows/demo.yml (manual dispatch).
# Run locally: `vhs docs/demo.tape` (needs vhs + ttyd + ffmpeg installed).

Output assets/demo.gif
Set FontSize 13
Set Width 1200
Set Height 620
Set Padding 20
Set Theme Dracula

# --- install (the CI job has already done `pip install -e .`, so envelcost is on PATH)
Type "envelcost --version"
Enter
Sleep 2s

# --- run the 5-task × 2-envelope benchmark; asserts the m1 variance gate
Type "envelcost run"
Enter
Sleep 3s

# --- render the per-envelope token-cost report
Type "envelcost report"
Enter
Sleep 2s

# --- project onto an 8xH100 / 50-seat cluster: native fits, openai-shape does not
Type "envelcost project --gpus 8xH100 --seats 50"
Enter
Sleep 3s

# --- a 3-envelope sweep including the claude-code-cliproxy harness
Type "envelcost run --harnesses deepseek-native,openai-shape,claude-code-cliproxy"
Enter
Sleep 3s
