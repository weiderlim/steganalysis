# simple command line to replace GUI

# export SCHEDULER_NAME=''
# get scheduler names
# gcloud scheduler jobs list

# echo y for installing gcloud beta to run this command
echo 'y' | gcloud beta scheduler jobs pause $SCHEDULER_NAME

# export INSTANCE_NAME=''
# get instance names
# gcloud compute instances list

# the user and project needs to match (allowed permission)
gcloud compute instances stop $INSTANCE_NAME
