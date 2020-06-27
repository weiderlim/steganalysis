# see current status of instances, more accurate than the GUI
# gcloud compute instances list

# see status of schedulers
# gcloud beta scheduler jobs list

# start instance
gcloud compute instances start $INSTANCE_NAME

# start scheduler
gcloud beta scheduler jobs resume $SCHEDULER_NAME