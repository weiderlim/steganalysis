# steganalysis
Kaggle Steganalysis Competition 

## Setup (First time only)

### Make scripts executable
```
chmod u+x scripts/*
```

### Prepare gcloud instance
Based on GCP setup guide from [fast.ai](https://course.fast.ai/start_gcp.html).
```
./scripts/create_instance.sh
./scripts/ssh_instance.sh
```

### Get the dataset
Place your [`kaggle.json`](https://github.com/Kaggle/kaggle-api#api-credentials) into the top-level of this repo, then run:
```
./scripts/get_dataset.sh
```