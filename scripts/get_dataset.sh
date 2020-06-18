export KAGGLE_CONFIG_DIR=".."

pip install --upgrade pip
pip install kaggle==1.5.6

# !kaggle competitions list
kaggle competitions download -c alaska2-image-steganalysis -p alaska2-image-steganalysis

# Unzip images
# https://stackoverflow.com/questions/61110400/unzipping-image-directory-in-google-colab-doesnt-unzip-entire-contents/61113374#61113374
cd alaska2-image-steganalysis
7z x alaska2-image-steganalysis.zip
rm alaska2-image-steganalysis.zip