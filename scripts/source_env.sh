echo -n 'Enter Email Address :';
read;

# email address
export ACCOUNT_NAME=${REPLY}
# whatever you wanna call your new project, it cannot be too generic like my-project etc, needs to be pretty unique
export PROJECT_ID='my-fastai-project' # eg. hellothereimcomputer
# name your service account, which is the account that manages your VM instances
# if your service account is "123456-compute@developer.gserviceaccount.com", just enter "123456-compute"
export SA_NAME='my-service-account' # eg. 123456-compute
# name your vm instance
export INSTANCE_NAME='my-fastai-instance' # eg. my-fastai-instance
# name the cloud function
export FUNCTION_NAME='resurrection-manager' # ressurection-function
# name the cloud scheduler
export SCHEDULER_NAME='resurrection-scheduler' # ressurection-scheduler