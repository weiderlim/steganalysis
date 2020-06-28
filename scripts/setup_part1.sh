# Part 1 of setup, seperated in order to adjust GPU quotas after creating project. 

# 1. Go to GCP console with your new account, Accept terms of services, activate $300 credit, enter personal and credit card details. 

# 2. Install gcloud
# Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"
# Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

# 3. Log in and authenticate Google SDK for your account when prompted
# See which accounts are active from multiple google accounts, use these for setting up and navigating between several accounts and projects
# gcloud auth list
gcloud auth login $ACCOUNT_NAME

# 4. Creating a new project and setting it as the current project
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# 5. Creating service account
gcloud iam service-accounts create $SA_NAME --description="Manages VM instance" --display-name=$SA_NAME

# 6. Enable billing for the project
gcloud alpha billing accounts list
# use this if you only have one billing account, otherwise check documentation of gcloud alpha
export ACCOUNT_ID=$(gcloud alpha billing accounts list --format 'value(ACCOUNT_ID)')
gcloud alpha billing projects link $PROJECT_ID --billing-account $ACCOUNT_ID

# 7. Enable apis for the gcloud, this takes awhile
gcloud services enable compute.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable appengine.googleapis.com

# 8. adjust your GPU quotas.
# Go to Google Cloud Quotas Page. Navigate to the project you just created (top left)
# If you signed up with a free tier account, you first need to upgrade to a paid account; do so by clicking the “Upgrade account” button at the top right of the page. Be careful as you will be charged from here on out if you exceed the quota limit.
# In the “Metrics” dropdown, select “GPUs (all regions)” and under “Locations” select “Global” (or “All locations”).
# Click edit quotas and select the quota to edit (GPUs All Regions). Set the new quota limit to 1 or more. Your request may require confirmation, which Google claims typically takes two business days to get.