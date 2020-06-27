# If you haven't completed steps 1-8, go to setup_part1.sh

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
gcloud functions deploy $FUNCTION_NAME --region=$REGION1 --allow-unauthenticated --runtime=$RUNTIME --source=$SOURCE --entry-point=$ENTRY_POINT --trigger-http --set-env-vars=ZONE=$ZONE,REGION=$REGION2,PROJECT_ID=$PROJECT_ID,INSTANCE_NAME=$INSTANCE_NAME 

# 11. Cloud Scheduler setup
export URI=$(gcloud functions describe $FUNCTION_NAME --format 'value(httpsTrigger.url)')

# checks every 10 mins
gcloud scheduler jobs create http $SCHEDULER_NAME --schedule='*/10 * * * *' --uri=$URI --http-method=GET --time-zone='Asia/Kuala_Lumpur'
# Select yes for all prompts, choose us-central for region of App Engine if prompted

# 12. Start the scheduler
gcloud beta scheduler jobs resume $SCHEDULER_NAME