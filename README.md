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

### Restarting in the event of preemption
The cost of running a preemptible instance is much lower than one without. Follow [guide](https://medium.com/martinomburajr/using-cloud-scheduler-to-resurrect-preempted-virtual-machines-c637c6d7f098) for the general steps to take to activate a Cloud Function and Cloud Scheduler to restart the instance after preemption. 

However, there are some things left out in the guide:
1. When creating a Cloud Function, make sure to 'Allow Unautenticated Invocations'. If you are worried about security, there is an option to not allow this and give permissions via IAM, but it has not been explored in detail.
2. For some reason when copy and pasting the URL from the Cloud Function into the Cloud Scheduler, the URL is messed up after setting the Scheduler up. Edit the Scheduler after it is set up and make sure the URL matches the one from the Cloud Function. 

