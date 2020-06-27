# Kaggle Steganalysis Competition 

## Setup (First time only) - Local Machine

#### Make scripts executable
```bash
chmod u+x scripts/*
```

### Prepare gcloud instance
Go to the GCP console after setting up your account (or using a current one), accept terms of services, activate $300 credit when prompted at the top of your screen, enter personal and credit card details. 

The following steps are based on the GCP setup guide from [fast.ai](https://course.fast.ai/start_gcp.html).

Make sure to run the .sh files in the same bash instance in order to not lose env variables.

You can edit the default names of your project name, service account, cloud function and cloud scheduler in the source_env.sh file. Enter your Email Address when prompted. Answer Yes to any question prompts.

Creating the project, service account, enabling billing and API services in the project.

```bash
. /scripts/source_env.sh
. /scripts/setup_part1.sh
```
### Adjust your GPU quotas.
Go to Google Cloud Quotas Page. Navigate to the project you just created (top left)

If you signed up with a free tier account, you first need to upgrade to a paid account; do so by clicking the “Upgrade account” button at the top right of the page. Be careful as you will be charged from here on out if you exceed the quota limit.

In the “Metrics” dropdown, select “GPUs (all regions)” and under “Locations” select “Global” (or “All locations”).

Click edit quotas and select the quota to edit (GPUs All Regions). Set the new quota limit to 1 or more. Your request may require confirmation, which Google claims typically takes two business days to get.

Not confirmed, but Google may reject your application if you use the same credit card details for different requests, even with different accounts. 

### Completing automated instance setup.
Creating VM instance, Cloud Function and Scheduler, and scripts to continue training the model when the instance is preempted. 

Note: The Cloud Function allows unauthenticated invocations (meaning anyone could call the function). If you are worried about security, there is an option to not allow this and give permissions via IAM, but it has not been explored in detail.

```bash
. /scripts/setup_part2.sh
```

#### SSH into the remote machine (if not already SSH-ed in)
```bash
. /scripts/ssh_instance.sh
```
Following commands are to be run in the remote machine. 

## Setup (First time only) - Remote Machine

### Clone this repo 
```bash
cd /home/jupyter/repos/
git clone https://github.com/weiderlim/steganalysis.git
cd /steganalysis/
```

### Install dependencies
```bash
. /scripts/provision.sh
```

### Get the dataset
Place your [`kaggle.json`](https://github.com/Kaggle/kaggle-api#api-credentials) into the top-level of this repo, then run:
```bash
. /scripts/get_dataset.sh
```

### Running the model
```bash
python baseline.py
```
And the model should start training!

## Additional Utility Commands (Local Machine)

### Stopping all processes
Command to stop both the instance and the scheduler to prevent further charging of credits.
```bash
. /scripts/stop_instance_scheduler.sh
```

### Restarting all processes
Command to restart the instance and the scheduler.
```bash
. /scripts/start_instance_scheduler.sh
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

### Killing the instance
If you don't want your account to be charged when your credit is up, this command kills the instance.
```bash
. /scripts/kill_instance.sh
```
