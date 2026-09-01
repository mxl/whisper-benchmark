# Local isolated environments

`.venvs/` contains disposable, local-only virtual environments. It is not an
environment manager or a project lockfile; each environment is created from
the instructions for its worker.

The generated environment contents remain untracked; `.gitignore` keeps only
this README. See [`environments/gigaam/README.md`](../environments/gigaam/README.md)
for the official GigaAM worker environment.
