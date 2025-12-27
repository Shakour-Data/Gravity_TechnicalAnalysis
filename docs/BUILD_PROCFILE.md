# Procfile generation and testing

This project includes a dedicated script to create a `Procfile` at build time to avoid brittle long inline `bash -c` commands.

Files added:
- `scripts/create-procfile.sh` — creates `Procfile` based on detected entrypoint (start_server.sh, start.sh, run.sh, main.py, app.py, etc.)
- `cloudbuild.yaml` — includes a `create-procfile` step which runs the script using the `ubuntu` image

Testing locally
- On Windows: use WSL or Git Bash, then run from repo root:
  - `bash scripts/create-procfile.sh`
  - `cat Procfile` to inspect the result
- In Docker: run an ubuntu container with the workspace mounted and execute the script.

Cloud Build
- The `cloudbuild.yaml` provided will run `chmod +x scripts/create-procfile.sh && ./scripts/create-procfile.sh` in the `ubuntu` image as a build step.

Notes
- The step intentionally sets `chmod +x` to ensure the script runs even on build hosts which don't preserve executable bit.
- If you have a custom build pipeline, replace the original inline `bash -c` step with the `create-procfile` step in your pipeline config.
