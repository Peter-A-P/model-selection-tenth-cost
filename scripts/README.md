# scripts

Two shell drivers. Neither is a convenience wrapper: both encode the exact sequence, item counts
and flags that make one model's numbers comparable with another's, and that sequence is part of
the method rather than part of the tooling.

| script | what it is for |
|---|---|
| `full-participation.sh` | Every call one panel member has to make to appear in all four findings, not just in the suite. |
| `framing-second-administration.sh` | The repeat administration that turns finding 8's noise correction from an upper bound into an estimate. |

Both are resumable, because every call is keyed by request content hash. Re-running either after
an interruption asks only what is outstanding, so there is no such thing as restarting from the
top. The one exception is `--repeat 2`, which uses its own cache namespace on purpose: a second
administration that read the first one's answers back would measure nothing and look like it had
worked.

`mselect run` refuses to send anything without `--yes`, and the gateway enforces this project's
spend caps before a request leaves. Both scripts pass `--yes`, so read
[`mselect/config/caps.yaml`](../mselect/config/caps.yaml) before running one against the hosted
panel. The laptop aliases cost nothing and need no key.

On Windows, launch a long run detached rather than from a terminal or tool that owns the process.
A nineteen-hour run died three times because its parent went away:

    Start-Process -FilePath "C:\Program Files\Git\bin\bash.exe" `
      -ArgumentList "scripts/full-participation.sh" -WindowStyle Hidden

Keep the machine on AC while one is running. The local models pin every core, and a laptop on
battery will sleep in the middle of an arm; that costs an hour of wall clock and no data.
