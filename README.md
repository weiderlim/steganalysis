# steganalysis
Kaggle Steganalysis Competition 

## Setup (First time only)

### Make scripts executable
```bash
chmod u+x scripts/*
```

### Prepare gcloud instance
Based on GCP setup guide from [fast.ai](https://course.fast.ai/start_gcp.html).
```bash
./scripts/create_instance.sh
./scripts/ssh_instance.sh
```

### Get the dataset
Place your [`kaggle.json`](https://github.com/Kaggle/kaggle-api#api-credentials) into the top-level of this repo, then run:
```bash
./scripts/get_dataset.sh
```

### Remote sessions
We use `tmux` to manage remote sessions (guides [here](https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) and [here](https://www.ostechnix.com/tmux-command-examples-to-manage-multiple-terminal-sessions/)). Cheatsheet:
```bash
tmux ls # List all sessions
tmux # Start tmux and create a session
tmux rename-session -t 0 database # Rename current session
tmux new -s webserver # Create named session
tmux kill-session -t ostechnix # Kill session
tmux attach -t ostechnix # Attach to session
# Detach from session: Ctrl+b d
```