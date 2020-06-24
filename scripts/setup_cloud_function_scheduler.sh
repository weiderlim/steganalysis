# currently we have setting up instance and ssh-ing into the instance. To fully automate the process we need to setup scripts for project creation, service accounts (not very needed atm since we don't really care about security, but i am sure there are also ways to automate that if needed), cloud function and scheduler. I think the create instance can be pulled here, only the ssh.instance is different.

# project creation 

# service account setup

# create instance 

# cloud function setup
export NAME='steg-manager-2'
export REGION1='us-central1'
export RUNTIME='go111'
export SERVICE_ACCOUNT='steganalysis-instance-manager@fast-ai-gpu-280616.iam.gserviceaccount.com'
export SOURCE='./cloud_function/'
export PROJECT_ID='fast-ai-gpu-280616'
export ZONE='us-west1-b'
export INSTANCE_NAME='my-fastai-instance'
export REGION2='us-west1'
export ENTRY_POINT='PollInstance'

# name is given first, parameters are added with -- 
gcloud functions deploy $NAME --region=$REGION1 --allow-unauthenticated --runtime=$RUNTIME --service-account=$SERVICE_ACCOUNT --source=$SOURCE --entry-point=$ENTRY_POINT --trigger-http --set-env-vars=ZONE=$ZONE,REGION=$REGION2,PROJECT_ID=$PROJECT_ID,INSTANCE_NAME=$INSTANCE_NAME

# gcloud describe i think to view the URL created at the endpoint from trigger-http, but dunno how to extract it, so for now using this brute force method
export URI='https://'$REGION1'-'$PROJECT_ID'.cloudfunctions.net/'$NAME
export SCHEDULER_NAME='steg-reawakener-2'
# checks every 10 minutes, not inputting into export because bash * means something else
# export INTERVAL='*/10 * * *'

# cloud scheduler setup
gcloud scheduler jobs create http $SCHEDULER_NAME --schedule='*/10 * * * *' --uri=$URI --http-method=GET --time-zone='Asia/Kuala_Lumpur'