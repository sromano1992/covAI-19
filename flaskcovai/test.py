# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 06:59:10 2020

@author: SimoneRomano
"""

#load data from Google Cloud Storage - GCS
from google.cloud import storage

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from io import BytesIO

# Explicitly use service account credentials by specifying the private key
# file.
storage_client = storage.Client.from_service_account_json(
    'security/covid19-270908-fe79e25b6fb0.json')

# Make an authenticated API request
buckets = list(storage_client.list_buckets())
print(buckets)

# Note: Client.list_blobs requires at least package version 1.17.0.
blobs = storage_client.list_blobs('covai19_dataset')

for blob in blobs:
    print(blob.time_created)
    
bucket = storage_client.bucket('covai19_dataset')
blob = bucket.blob('newfile.csv')
blob.upload_from_string("ciaociao", content_type='text/plain')



#dataset_bucket = storage_client.get_bucket("covai19_dataset")
#blob = dataset_bucket.get_blob('covid_19_data.csv')
## download as string
#json_data = blob.download_as_string()
#
#
#train = pd.read_csv(BytesIO(json_data), encoding='utf8', sep=',')
#
#
#dataset_bucket = storage_client.get_bucket("covai19_models")
#blob = dataset_bucket.get_blob('clf_confirmed_SVR.p')
#import pickle
#clf_recovered = pickle.loads(blob.download_as_string())