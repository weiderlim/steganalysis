export ZONE="us-west1-b"
export INSTANCE_NAME="my-fastai-instance" # or whatever name if not default

gcloud compute ssh --zone=$ZONE jupyter@$INSTANCE_NAME -- -L 8080:localhost:8080