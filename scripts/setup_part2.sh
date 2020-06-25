# 8. adjust your GPU quotas.
# Go to Google Cloud Quotas Page. Navigate to the project you just created (top left)
# If you signed up with a free tier account, you first need to upgrade to a paid account; do so by clicking the “Upgrade account” button at the top right of the page. Be careful as you will be charged from here on out if you exceed the quota limit.
# In the “Metrics” dropdown, select “GPUs (all regions)” and under “Locations” select “Global” (or “All locations”).
# Click edit quotas and select the quota to edit (GPUs All Regions). Set the new quota limit to 1 or more. Your request may require confirmation, which Google claims typically takes two business days to get.

# 9. Creating the VM instance
export IMAGE_FAMILY="pytorch-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
export ZONE="us-west1-b"
export INSTANCE_TYPE="n1-highmem-8"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible

# 10. Cloud Function setup
export REGION1='us-central1'
export RUNTIME='go111'
export SERVICE_ACCOUNT=$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com
export SOURCE='./cloud_function/'
export REGION2='us-west1'
export ENTRY_POINT='PollInstance'

# name is given first, parameters are added with -- 
gcloud functions deploy $FUNCTION_NAME --region=$REGION1 --allow-unauthenticated --runtime=$RUNTIME --service-account=$SERVICE_ACCOUNT --source=$SOURCE --entry-point=$ENTRY_POINT --trigger-http --set-env-vars=ZONE=$ZONE,REGION=$REGION2,PROJECT_ID=$PROJECT_ID,INSTANCE_NAME=$INSTANCE_NAME 

# 11. Cloud Scheduler setup
export URI=$(gcloud functions describe $FUNCTION_NAME --format 'value(httpsTrigger.url)')

# checks every 10 mins
gcloud scheduler jobs create http $SCHEDULER_NAME --schedule='*/10 * * * *' --uri=$URI --http-method=GET --time-zone='Asia/Kuala_Lumpur'

# 12. Navigate to the Cloud Scheduler page, click 'RUN NOW' on the scheduler in order to activate it.