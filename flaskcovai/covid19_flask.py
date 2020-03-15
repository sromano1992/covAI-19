# coding: utf-8

#  # Data import
# 

# In[128]:

from flask import Flask, request

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from io import BytesIO

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Welcome to covAI-19 study!")

#load data from Google Cloud Storage - GCS

# Explicitly use service account credentials by specifying the private key
# file.

from flask_cors import CORS
app = Flask('covAI-19')
cors = CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    
    country = request.form['country']

    # Make an authenticated API request
    from google.cloud import storage
    storage_client = storage.Client.from_service_account_json('security/covid19-270908-fe79e25b6fb0.json')
    buckets = list(storage_client.list_buckets())
    print(buckets)
    
    #load data from Google Cloud Storage - GCS
    from google.cloud import storage
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from io import BytesIO
    from flask import jsonify
    
    dataset_bucket = storage_client.get_bucket("covai19_dataset")
    
    covid_19_data_blob = dataset_bucket.get_blob('covid_19_data.csv')
    covid_19_line_list_data_blob = dataset_bucket.get_blob('COVID19_line_list_data.csv')
    covid_19_open_line_list_blob = dataset_bucket.get_blob('COVID19_open_line_list.csv')
    
    dataset_time_created = covid_19_data_blob.time_created
    dataset_time_created_string = dataset_time_created.strftime("%Y_%m_%d")
    #check if, for the last dataset, we have already created the json
    filename_to_check = 'json_' + country + '_' + dataset_time_created_string + '.csv'
    data = dataset_bucket.get_blob(filename_to_check)
    if (data is not None):
        logger.info('data already calculated...returing...')
        return jsonify(data.download_as_string().decode("utf-8"))
        
    
    # download as string
    json_data = covid_19_data_blob.download_as_string()
    covid_19_data = pd.read_csv(BytesIO(json_data), encoding='utf8', sep=',', header=0, index_col=0, parse_dates=['ObservationDate'], squeeze=True)
    json_data = covid_19_line_list_data_blob.download_as_string()
    covid_19_line_list_data = pd.read_csv(BytesIO(json_data), encoding='utf8', sep=',', header=0, index_col=0, parse_dates=['reporting date'], squeeze=True)
    json_data = covid_19_open_line_list_blob.download_as_string()
    covid_19_open_line_list = pd.read_csv(BytesIO(json_data), encoding='utf8', sep=',', header=0, index_col=0, parse_dates=['date_confirmation'], squeeze=True)
    
    
    # # Data country filter
    # In[130]:
    
    
    covid_19_data_countryFiltered = None
    covid_19_line_list_data_countryFiltered = None
    covid_19_open_line_list_countryFiltered = None
    
    def loadData(_COUNTRY_covid_19_data = 'Mainland China', _COUTNRY_covid_19_line_list_data = 'China', _COUTNRY_covid_19_open_line_list = 'China'):
        global covid_19_data_countryFiltered
        covid_19_data_countryFiltered = covid_19_data[covid_19_data['Country/Region'].eq(_COUNTRY_covid_19_data)]
        if (_COUNTRY_covid_19_data == 'Mainland China'):
            #for Mainland China we have the data grouped by cities; here cumulative data are grouped per Country (as for Italy and others Countries)
            #Ex. 
            #806,02/05/2020,,France,2020-02-01T01:52:40,6.0,0.0,0.0
            #815,02/05/2020,,Italy,2020-01-31T08:15:53,2.0,0.0,0.0
            #826,02/05/2020,Tibet,Mainland China,2020-02-01T01:52:40,1.0,0.0,0.0
            #837,02/06/2020,Hubei,Mainland China,2020-02-06T23:23:02,22112.0,618.0,817.0
            aggregate = covid_19_data_countryFiltered.groupby('ObservationDate').sum()
            aggregate = aggregate.reset_index()
            covid_19_data_countryFiltered = aggregate
        global covid_19_line_list_data_countryFiltered
        covid_19_line_list_data_countryFiltered = covid_19_line_list_data[covid_19_line_list_data['country'].eq(_COUTNRY_covid_19_line_list_data)]
        global covid_19_open_line_list_countryFiltered
        covid_19_open_line_list_countryFiltered = covid_19_open_line_list[covid_19_open_line_list['country'].eq(_COUTNRY_covid_19_open_line_list)]
        return covid_19_data_countryFiltered, covid_19_open_line_list_countryFiltered
    
    
    # In[3]:
    
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    #get_ipython().run_line_magic('matplotlib', 'notebook')
    
    def plotChart(covid_19_data_countryFiltered, countryName):
        # Create figure and plot space
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # Add x-axis and y-axis
        confirmedLine, = ax.plot(covid_19_data_countryFiltered['ObservationDate'],
                            covid_19_data_countryFiltered['Confirmed'],
                            color='purple')
        confirmedLine.set_label('confirmed')
    
        deathsLine, = ax.plot(covid_19_data_countryFiltered['ObservationDate'],
                            covid_19_data_countryFiltered['Deaths'],
                            color='red')
        deathsLine.set_label('deaths')
    
        recoveredLine, = ax.plot(covid_19_data_countryFiltered['ObservationDate'],
                            covid_19_data_countryFiltered['Recovered'],
                            color='yellow')
        recoveredLine.set_label('recovered')
    
        # Set title and labels for axes
        ax.set(xlabel="Date",
               title=countryName + " Confirmed cases COVID 19")
    
        ax.legend()
        ax.grid(True)
    
        plt.setp(ax.get_xticklabels(), rotation=45)
    
        plt.show()
    
    
    # # Features extraction
    
    # In[131]:
    
    
    from datetime import datetime, timedelta
    import math
    #this functoin extracts the features related to the targetData using the last numberOfDays data
    #example:
    #target date: 13/03/2020
    #numberOfDays: 7
    #data to take in consideration: 06-12/03/2020
    #
    #dataTypes:
    #    targetDate --> datetime
    def featuresExtraction(covid_19_open_line_list_countryFiltered, covid_19_data_countryFiltered, targetDate, numberOfDays = 7):
        logging.info("features extraction start...")
        
        oldestDate = covid_19_open_line_list_countryFiltered['date_confirmation'].iloc[0]
        oldestDate_day = oldestDate[0:2]
        oldestDate_month = oldestDate[3:5]
        oldestDate_year = oldestDate[6:10]
        d_old = datetime(int(oldestDate_year), int(oldestDate_month), int(oldestDate_day))
        daysDifference = (targetDate - d_old).days
        
        logger.debug("targetDate: " + str(targetDate) + " oldestDate (from dataset): " + str(d_old))
        logger.debug("days between dates: " + str (daysDifference))
        
        if (daysDifference >= numberOfDays):
            dateLowerBound = targetDate - timedelta(days=numberOfDays)
            dateUpperBound = targetDate - timedelta(days=1)
            logger.info("We have enough data to calucalte features...let me go ahead...")
            logger.info("The interval that will be used: [" + str(dateLowerBound) + ", " + str(dateUpperBound) + "]")
        else:
            logger.info("Not enough data to caluclate features...exiting...")
            return None
        
        features = []
        features.append("[" + str(dateLowerBound) + ", " + str(dateUpperBound) + "]")
    
        #dataset covid_19_data_countryFiltered
        #increment_confirmed_previous_n_days
        yesterdayData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=1))]['Confirmed'].values[0]      
        nDaysAgoData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=numberOfDays))]['Confirmed'].values[0]       
        value = yesterdayData - nDaysAgoData 
        features.append(value)
        
        #increment_deaths_previous_n_days
        yesterdayData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=1))]['Deaths'].values[0]      
        nDaysAgoData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=numberOfDays))]['Deaths'].values[0]       
        value = yesterdayData - nDaysAgoData 
        features.append(value)
        
        #increment_recovered_previous_n_days
        yesterdayData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=1))]['Recovered'].values[0]      
        nDaysAgoData = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(targetDate - timedelta(days=numberOfDays))]['Recovered'].values[0]       
        value = yesterdayData - nDaysAgoData 
        features.append(value)
        
        #dataset covid_19_open_line_list_countryFiltered    
        #creating dates in format 22.01.2020 to interact with this dataset
        dates = []
        for i in range (7):
            currentDate = targetDate - timedelta(days=i+1)
            dates.append(currentDate.strftime("%d.%m.%Y"))
        logger.debug(str(dates))
        logger.debug(covid_19_open_line_list_countryFiltered.shape)
        covid_19_open_line_list_countryDateFiltered = covid_19_open_line_list_countryFiltered[covid_19_open_line_list_countryFiltered['date_confirmation'].isin(dates)]
        logger.debug(covid_19_open_line_list_countryDateFiltered.shape)
        
        #min_age_previous_n_days
        value = pd.to_numeric(covid_19_open_line_list_countryDateFiltered['age'], errors='coerce').min()
        features.append(value)
    
        #max_age_previous_n_days
        value = pd.to_numeric(covid_19_open_line_list_countryDateFiltered['age'], errors='coerce').max()
        features.append(value)
        
        #mean_age_previous_n_days
        value = pd.to_numeric(covid_19_open_line_list_countryDateFiltered['age'], errors='coerce').mean()
        features.append(value)
        
        #std_age_previous_n_days
        value = pd.to_numeric(covid_19_open_line_list_countryDateFiltered['age'], errors='coerce').std()
        features.append(value)
        
        #male_count_previous_n_days
        value = covid_19_open_line_list_countryDateFiltered[covid_19_open_line_list_countryDateFiltered['sex'].eq('male')].shape[0]
        features.append(value)
        
        #female_count_previous_n_days
        value = covid_19_open_line_list_countryDateFiltered[covid_19_open_line_list_countryDateFiltered['sex'].eq('female')].shape[0]
        features.append(value)
        
        #different_cities_count_previous_n_days
        value = covid_19_open_line_list_countryDateFiltered.city.unique().size
        features.append(value)
        
        #different_province_count_previous_n_days
        value = covid_19_open_line_list_countryDateFiltered.province.unique().size
        features.append(value)
        
        #gps_coordinates_distribution_previous_n_days
        logger.warning("graph analysis not implemented yet...")
        
        #mean_num_of_days_from_onset_symptoms_previous_n_days
        #mean across all cases in the intercal of the number of days from onset sympton to the confirmation
        num_of_days_from_onset_symptoms_previous_n_days = []
        for item in range(covid_19_open_line_list_countryDateFiltered.shape[0]):
            try:
                dateOnsetSymtps = covid_19_open_line_list_countryDateFiltered.iloc[item].date_onset_symptoms 
                if isinstance(dateOnsetSymtps,float) and math.isnan(dateOnsetSymtps):
                    continue
                dateOnsetSymtps_day = dateOnsetSymtps[0:2]
                dateOnsetSymtps_month = dateOnsetSymtps[3:5]
                dateOnsetSymtps_year = dateOnsetSymtps[6:10]
                d_dateOnsetSymtps = datetime(int(dateOnsetSymtps_year), int(dateOnsetSymtps_month), int(dateOnsetSymtps_day))
    
                dateConfirmation = covid_19_open_line_list_countryDateFiltered.iloc[item].date_confirmation 
                if isinstance(dateConfirmation,float) and math.isnan(dateConfirmation):
                    continue
                dateConfirmation_day = dateConfirmation[0:2]
                dateOnsetSymtps_month = dateConfirmation[3:5]
                dateConfirmation_year = dateConfirmation[6:10]
                d_dateConfirmation = datetime(int(dateConfirmation_year), int(dateOnsetSymtps_month), int(dateConfirmation_day))
    
                daysDifference = (d_dateConfirmation - d_dateOnsetSymtps).days
                num_of_days_from_onset_symptoms_previous_n_days.append(daysDifference)
            except:
                logger.debug("ignoring item number " + str(item) + "...")
        value = np.mean(num_of_days_from_onset_symptoms_previous_n_days)
        features.append(value)
        
        #variance_num_of_days_from_onset_symptoms_previous_n_days
        value = np.var(num_of_days_from_onset_symptoms_previous_n_days)
        features.append(value)
        
        #sd_num_of_days_from_onset_symptoms_previous_n_days
        value = np.std(num_of_days_from_onset_symptoms_previous_n_days)
        features.append(value)
        
        #number_of_reported_trips_previous_n_days
        #number_of_trips_days_previous_n_days
        number_of_reported_trips_previous_n_days = []
        number_of_trips_days_previous_n_days = []    
        for item in range(covid_19_open_line_list_countryDateFiltered.shape[0]):
            try:    
                #numberOfTrips
                number_of_reported_trips_previous_n_days.append(len(covid_19_open_line_list_countryDateFiltered.iloc[item]['travel_history_location'].split(",")))
            except:
                logger.debug("ignoring item number " + str(item) + "...")
            try:    
                #numberOfDaysOfTrips
                number_of_trips_days_previous_n_days.append(len(covid_19_open_line_list_countryDateFiltered.iloc[item]['travel_history_dates'].split(",")))
            except:
                logger.debug("ignoring item number " + str(item) + "...")
        
        value = np.sum(number_of_reported_trips_previous_n_days)
        features.append(value)
        
        value = np.sum(number_of_trips_days_previous_n_days)
        features.append(value)
        
        
        logger.debug(str(features))
        logger.info("features extraction end...")
        return features
        
    
        
        #targetDate_day = targetDate[0:2]
        #targetDate_month = targetDate[3:5]
        #targetDate_year = targetDate[6:10]
        #d_target = datetime.datetime(int(targetDate_year), int(targetDate_month), int(targetDate_day))
    
        #covid_19_open_line_list_countryFiltered[covid_19_open_line_list_countryFiltered['date_confirmation'].eq('22.01.2020')]
    
    
    # # Dataset generation
    
    # ## x data
    
    # In[132]:
    
    
    #loadData('Mainland China', 'China', 'China')
    if (country == 'China'):
        covid_19_data_countryFiltered, covid_19_open_line_list_countryFiltered = loadData('Mainland China', 'China', 'China')
    else:
        covid_19_data_countryFiltered, covid_19_open_line_list_countryFiltered = loadData(country, country, country)
    dataset = []
    
    for index, item in covid_19_data_countryFiltered.iterrows():
        targetDate = item['ObservationDate']
        out = featuresExtraction(covid_19_open_line_list_countryFiltered, covid_19_data_countryFiltered, targetDate)
        logger.info("new features vecotr generated: " + str(out))
        if out is not None:#the top row we cannot generate because of "not minimun historical data..."
            logger.info("append...")
            dataset.append(out)
        
    
    
    # ## y data
    
    # In[119]:
    
    
    future_deep_day = 7 #how many days ahead you want to predict?
                        #example: with 7, you will use the range [2020-01-26 00:00:00, 2020-02-01 00:00:00] to predict the 2020-02-08 and so on...
    y_dataset = []
    last_index = covid_19_data_countryFiltered.shape[0]-1
    newestDateCovid_19 = covid_19_data_countryFiltered.iloc[last_index]['ObservationDate']
    dataset_last_index_to_use = 0
    for item in dataset:
        asNpArray = np.array(item)
        if asNpArray.size == 1:
            logger.info("no data for this row: continue...")
            continue
        upperBoundDateInterval = asNpArray[0].split(',')[1].strip()[0:10]
        date_time = datetime.strptime(upperBoundDateInterval,"%Y-%m-%d")
        target_date = date_time + timedelta(days=future_deep_day)
        logger.info(str(asNpArray[0]) + " - " + str(upperBoundDateInterval) + " - " + str(target_date))
        if (target_date > newestDateCovid_19):
            logger.info('last available target data is: ' + str(target_date) + ' -1 day...update your dataset to go ahead...')
            break
        date_formatted = target_date.strftime("%Y-%m-%d")
        target = covid_19_data_countryFiltered[covid_19_data_countryFiltered['ObservationDate'].eq(date_formatted)]
        y_dataset.append(target)
        dataset_last_index_to_use += 1
    dataset_train = dataset[0:dataset_last_index_to_use]    
    
    
    # In[120]:
    
    
    dataset_np = np.array(dataset_train)[:,1:]
    dataset_np = np.nan_to_num(dataset_np, copy=False).astype(float)
    dataset_np = np.nan_to_num(dataset_np, copy=False).astype(float)
    
    y_confirmed = []
    y_death = []
    y_recovered = []
    for item in y_dataset:
        y_confirmed.append(float(item['Confirmed']))
        y_death.append(float(item['Deaths']))
        y_recovered.append(float(item['Recovered']))
    from sklearn import svm, linear_model
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge
    
    # # Predicted chart
    # load models
    import pickle
    logger.info("loding models from GCS...")
    dataset_bucket = storage_client.get_bucket("covai19_models")
    
    blob = dataset_bucket.get_blob('clf_confirmed_SVR.p')
    clf_confirmed = pickle.loads(blob.download_as_string())
    
    blob = dataset_bucket.get_blob('clf_deaths_SVR.p')
    clf_deaths = pickle.loads(blob.download_as_string())
    
    blob = dataset_bucket.get_blob('clf_recovered_SVR.p')
    clf_recovered = pickle.loads(blob.download_as_string())
    
    dataset_all = np.array(dataset)[:,1:]
    dataset_all = np.nan_to_num(dataset_all, copy=False).astype(float)
    dataset_all = np.nan_to_num(dataset_all, copy=False).astype(float)
    first_target_date = dataset[0][0].split(',')[1].strip()[0:10]
    first_target_date = datetime.strptime(first_target_date,"%Y-%m-%d")
    first_target_date = first_target_date + timedelta(days=future_deep_day)
    logger.info(str(first_target_date))
    
    x_plot = []
    y_confirmed_predicted_plot = []
    y_deaths_predicted_plot = []
    y_recovered_predicted_plot = []
    for item in dataset_all:
        y_confirmed_predicted_plot.append(clf_confirmed.predict([item])[0])
        y_deaths_predicted_plot.append(clf_deaths.predict([item])[0])
        y_recovered_predicted_plot.append(clf_recovered.predict([item])[0])
        x_plot.append(first_target_date.strftime("%Y-%m-%d"))
        first_target_date = first_target_date + timedelta(days=1)
        
    #date_time = datetime.strptime(upperBoundDateInterval,"%Y-%m-%d")
    
    
    # In[134]:
    
    
    zero_to_add = len(y_confirmed_predicted_plot) - len(y_confirmed)
    logger.info("Zeros to add: " + str(zero_to_add))
    for i in range(zero_to_add):
        y_confirmed.append(0)
        y_death.append(0)
        y_recovered.append(0)
    
    _COUNTRY = "ITALY"
    #get_ipython().run_line_magic('matplotlib', 'notebook')
    # Create figure and plot space
    
    
    # Plot the raw time series
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig)
    
    ax = fig.add_subplot(gs[0, :])
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_confirmed_predicted_plot,
                        '-b')
    confirmedLine.set_label('confirmed Predicted')
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_confirmed,
                        'ro')
    confirmedLine.set_label('confirmed')
    # Set title and labels for axes
    ax.set(xlabel="Date",
           title=country + " Confirmed cases PREDICTION COVID 19")
    ax.legend()
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
    ax = fig.add_subplot(gs[1, :])
    # Create figure and plot space
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_deaths_predicted_plot,
                        '-b')
    confirmedLine.set_label('Deaths Predicted')
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_death,
                        'ro')
    confirmedLine.set_label('Death')
    # Set title and labels for axes
    ax.set(xlabel="Date",
           title=country + " Death cases PREDICTION COVID 19")
    ax.legend()
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
    ax = fig.add_subplot(gs[2, :])
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_recovered_predicted_plot,
                        '-b')
    confirmedLine.set_label('Recovered Predicted')
    
    # Add x-axis and y-axis
    confirmedLine, = ax.plot(x_plot,
                        y_recovered,
                        'ro')
    confirmedLine.set_label('Recovered')
    
    # Set title and labels for axes
    ax.set(xlabel="Date",
           title=country + " Recovered cases PREDICTION COVID 19")
    
    ax.legend()
    ax.grid(True)
    
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    #plt.show()
    
    import base64
    import io
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.savefig('fig.png')
    
    logger.info("Buffer created...returning bytes array...")
    encoded_string = base64.b64encode(buf.read())
    base64_string = encoded_string.decode('utf-8')
    buf.close()
    plt.close()
        
    d = {'author': 'Simone Romano - https://www.linkedin.com/in/simoneromano92/', 'chart': base64_string, 'date': x_plot, 'confirmed': y_confirmed, 'confirmed_predicted': y_confirmed_predicted_plot, 'deaths': y_death, 'deaths_predicted': y_deaths_predicted_plot, 'recovered': y_recovered, 'recovered_predicted': y_recovered_predicted_plot}
    
    
    to_return = str(d).replace("'","\"")
    
    #upload to CS
    logger.info("Uploading to Cloud Storage...")
    bucket = storage_client.bucket('covai19_dataset')
    blob = bucket.blob(filename_to_check)
    blob.upload_from_string(to_return)

    return(jsonify(to_return))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)