# Part 1 of setup, break in order to adjust GPU quotas after creating project. Make sure to run both in the same bash instance in order to not lose env variables needed in part 2.

# 1. Go to GCP console with your new account, Accept terms of services, activate $300 credit, enter personal and credit card details. 

# Fill in the details below:
export ACCOUNT_NAME=''
# whatever you wanna call your new project, it cannot be too generic like my-project etc, needs to be pretty unique
export PROJECT_ID=''
# name your service account, which is the account that manages your VM instances
export SA_NAME=''
# name your vm instance
export INSTANCE_NAME=''
# name the cloud function
export FUNCTION_NAME=''
# name the cloud scheduler
export SCHEDULER_NAME=''

# 2. See which accounts are active from multiple google accounts, use these for setting up and navigating between several accounts and projects
gcloud auth list

# 3. Log in and authenticate Google SDK for your account when prompted
gcloud auth login $ACCOUNT_NAME

# 4. Creating a new project and setting it as the current project
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# 5. Creating service account
gcloud iam service-accounts create $SA_NAME --description="Manages VM instance" --display-name=$SA_NAME

# 6. Enable billing for the project
y | gcloud alpha billing accounts list
# use this if you only have one billing account, otherwise check documentation of gcloud alpha
export ACCOUNT_ID=$(gcloud alpha billing accounts list --format 'value(ACCOUNT_ID)')
gcloud alpha billing projects link $PROJECT_ID --billing-account $ACCOUNT_ID

# 7. Enable apis for the VM instance, this takes awhile
gcloud services enable compute.googleapis.com

