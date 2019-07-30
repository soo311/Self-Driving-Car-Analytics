from solution.logger_initializer import *
from collections import Counter
from sklearn import cluster
import argparse
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.style.use('seaborn-whitegrid')

"""
solution Autopilot Data Sciece Code Challenge Solution Implementation.
"""

# LOG_DIR = '/orbital/soo/data/log'
# DATA_DIR = "/orbital/soo/data/"
LOG_DIR = '/Users/soohyunlee/Desktop/python/Self-Driving-Car-Analytics/data/log'
DATA_DIR = '/Users/soohyunlee/Desktop/python/Self-Driving-Car-Analytics/data/solution'
initialize_logger(LOG_DIR)


##############################################################################
#                             PARSE DATA
##############################################################################

class TimeseriesData(object):
    """
    Timeseries Data - base calss, stores the provided timeseries signal data
    and the users can execute the queries of interest.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def read_csvs(self, runtime=0):
        """
        Parse through the csv files and create dictionary whose key is the vin
        number and the value is the parsed dataframe.

        Args:
            runtime (int): If running for the first time, runtime must be equal
            to 0. This will create a pickle object after parsing. After the
            first time, use the runtime 1. This will read the pickle object
            instead of looping through the csv files, thus saving more time.

        """
        if runtime == 0:
            data_dict = {}
            for data in os.listdir(self.data_dir):
                if data.endswith('csv'):
                    df = self.read_csv(self.data_dir, data)
                    data_dict[data[:5]] = df
            logging.info("Completed parsing data")

            with open("data_dict.pickle", "wb") as handle:
                pickle.dump(data_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("Saving the dictionary to pickle object")
        if runtime == 1:
            try:
                with open("data_dict.pickle", "rb") as handle:
                    data_dict = pickle.load(handle)
                logging.info("Reading the data from the pickle")
            except:
                logging.warning("There is no pickle object to read data from.")
        return data_dict

    def read_csv(self, data_dir, data):
        assert data.endswith('csv'), "File has to be in .csv"
        file_path = data_dir + data
        logging.info("Reading signal data from {}".format(file_path))
        df = pd.read_csv(file_path)
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.vin = data[:5]
        df.timestamp_utc = pd.to_datetime(df.timestamp_utc)
        df['year'] = df.timestamp_utc.dt.year
        df['month'] = df.timestamp_utc.dt.month
        df['day'] = df.timestamp_utc.dt.day
        df['doy'] = df.timestamp_utc.dt.dayofyear
        df['time_spent'] = df.timestamp_utc.diff().shift(-1)
        df['maxima'] = df.sig_value[(df.sig_value > df.sig_value.shift(1)) & (
            df.sig_value > df.sig_value.shift(-1))]
        df['minima'] = df.sig_value[(df.sig_value < df.sig_value.shift(1)) & (
            df.sig_value < df.sig_value.shift(-1))]
        return df

##############################################################################
#                               QUESTION 1
##############################################################################
    def get_by_total_occurrence(self, df, n=1, top=True,
                                start_date=None, end_date=None):
        """
        Get the top N most/least common values by total occurence.

        Args:
            df         (DataFrame): DataFrame that contains full timeseries.
            n          (int): Number of results the user wants to see.
            top        (boolean): If wants to see top results, set to True,
                                if least set to False.
            start_date (str): Start date of the search range. "YYYY-MM-DD"
            end_date   (str): End date of the search range. "YYYY-MM-DD"

        Returns:
            List of top n most/least common values by total occurence.

        """
        logging.info("Getting data by the total occurence")
        df_time = df.set_index('timestamp_utc')
        if start_date:
            df_time = df_time.loc[start_date:]
        if end_date:
            df_time = df_time.loc[:end_date]
        val_count = Counter(df_time.sig_value)
        if top:
            return val_count.most_common()[:n]
        else:
            return val_count.most_common()[-n:]

    def get_by_total_time_spent(self, df, n=1, top=True,
                                start_date=None, end_date=None):
        """
        Get the top N most/least common values by total time spent.

        Args:
            df         (DataFrame): DataFrame that contains full timeseries.
            n          (int): Number of results the user wants to see.
            top        (boolean): If wants to see top results, set to True,
                                  if least set to False.
            start_date (str): Start date of the search range. "YYYY-MM-DD"
            end_date   (str): End date of the search range. "YYYY-MM-DD"

        Returns:
            List of top n most/least common values by total time spent.

        """
        logging.info("Getting data by the total time spent (v1)")
        df_time = df.set_index('timestamp_utc')
        if start_date:
            df_time = df_time.loc[start_date:]
        if end_date:
            df_time = df_time.loc[:end_date]
        df_time = df_time.groupby('sig_value').agg({'time_spent': 'sum'})
        df_time = df_time.sort_values('time_spent', ascending=False)
        if top:
            return df_time.iloc[:n].index.tolist()
        else:
            return df_time.iloc[-n:].index.tolist()

    def get_by_total_time_spent_2(self, df, n=1, top=True):
        logging.info("Getting data by the total time spent (v2)")
        df_pivot = pd.pivot_table(df, index='day', columns='sig_value',
                                  values='time_spent', aggfunc=np.sum,
                                  fill_value=0, margins=True,
                                  margins_name='Total Time Spent')
        df_unstack = df_pivot.unstack().xs(
            'Total Time Spent', level='day')[:-1]
        df_res = pd.DataFrame(df_unstack).reset_index().rename(
            columns={0: 'total_time_spent'})
        df_res = df_res.sort_values(
            'total_time_spent', ascending=False).reset_index(drop=True).reset_index(drop=True)

        if top:
            return df_res.iloc[:n].sig_value.tolist()
        else:
            return df_res.iloc[-n:].sig_value.tolist()

    def drop_consecutive_values(self, df):
        """
        Given a timeseries signal data, eliminate the consecutive values by
        keeping only the first value. This step is required to find the local
        min/max.

        Args:
            df (DataFrame): DataFrame that contains full timeseries.

        Returns:
            df (DataFrame): DataFrame without consecutive signal values.

        """
        i = 0
        shape_old = df.shape[0]
        shape_new = 0
        while shape_old != shape_new:
            df_shift = df.loc[df.sig_value.shift() != df.sig_value]
            shape_new = df_shift.shape[0]
            df = df_shift
            i += 1
            shape_old = shape_new
        df = df[(~df.maxima.isnull()) | (
            ~df.minima.isnull())].reset_index(drop=True)
        return df

    def get_cycle_info(self, df):
        """
        Given the processed dataframe, extract information about the cycles.
        This includes minima, maxima, size of a cycle given by the amplitude,
        and length of the cycle (defined as time between min/max to the next
        min/max pair).

        Args:
            df (DataFrame): DataFrame that contains full timeseries.

        Returns:
            df (DataFrame): DataFrame without consecutive signal values.

        """
        df['minima'] = df.minima.shift()
        df = df.dropna(axis=0, subset=[
                       'maxima', 'minima']).reset_index(drop=True)
        df['amplitude'] = np.abs(df.maxima.values - df.minima.values)
        df['cycle_length'] = df.timestamp_utc.diff().shift(-1).dt.total_seconds()
        return df

    def get_largest_cycle(self, df, n=1, top=True):
        """
        Find the n largest/smallest cycles determined by its amplitude.

        Args:
            df (DataFrame): DataFrame that contains full timeseries.
            n        (int): Number of results the user wants to see.
            top  (boolean): If wants to see top results, set to True,
                                if least set to False.

        Returns:
            df (DataFrame): Dataframe for the n number of largest/smallest
                            cycles. It contains maxima, minima, amplitude,
                            and cycle length information as columns.

        """
        df = self.drop_consecutive_values(df)
        df = self.get_cycle_info(df)
        df = df.sort_values(
            'amplitude', ascending=False).reset_index(drop=True)
        if top:
            return df.iloc[:n][['maxima', 'minima', 'amplitude', 'cycle_length']]
        else:
            return df.iloc[-n:][['maxima', 'minima', 'amplitude', 'cycle_length']]

##############################################################################
#                               QUESTION 2
##############################################################################

    def find_two_vehicles(self, data_dict, n=100, plot=False):
        """
        Find the 2 vehcles that are suspected to have experienced higher damage
        accrual than the other 8. Here, the problem was solved assuming high
        volatility of size of the cycle(amplitude) as it will result in a very
        abrupt change from one torque to the other.

        Avg. Complexity: O(2 N)

        Args:
            data_dict (dictionary): Dictionary of timeseries data.
            n                (int): Number of results the user wants to see.
            plot         (boolean): True if the user wants to see the clustering
                                    result plot. The image will be saved in the
                                    DATA_DIR

        Returns:
            List of 2 vehicles that are suspected to behave differently.
        """
        logging.info(
            "Finding the two vehicles with more potential damage accrual")
        car_index = []
        X = []
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        for car, df in data_dict.iteritems():
            car_index.append(car)
            data = self.get_largest_cycle(df, n)[['amplitude', 'cycle_length']]
            ax.scatter(x=data.cycle_length, y=data.amplitude, label=car)
            X.append(data.amplitude)
        ax.legend()
        ax.set_xlabel('Cycle Length (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('')
        if plot:
            fig.savefig(DATA_DIR + "scatter_plot.png")
            logging.info("Saving the scatter plot to the data directory")

        k_means = cluster.KMeans(n_clusters=2).fit(X)
        cluster_map = pd.DataFrame()
        cluster_map['vin'] = car_index
        cluster_map['data_index'] = X
        cluster_map['cluster'] = k_means.labels_

        cluster_0 = list(cluster_map[cluster_map.cluster == 0]['vin'])
        cluster_1 = list(cluster_map[cluster_map.cluster == 1]['vin'])

        if len(cluster_0) < len(cluster_1):
            return cluster_0
        else:
            return cluster_1


##############################################################################
#                             RUN THE SCRIPT
##############################################################################

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--data_dir', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    data_dir = args.data_dir if args.data_dir else DATA_DIR

    td = TimeseriesData(data_dir)
    data_dict = td.read_csvs(runtime=0)

    print "******************** PART I ********************"
    print "Q1. The 5 least common values by total occurence"
    print "A: {}".format(td.get_by_total_occurrence(df=data_dict['car_0'], n=5, top=False))
    print "\n Q2. The 3 most common values by total time spent at that value"
    print "A (v1): {}".format(td.get_by_total_time_spent(df=data_dict['car_0'], n=3, top=True))
    print "A (v2): {}".format(td.get_by_total_time_spent_2(df=data_dict['car_0'], n=3))
    print "\n Q3. 3 largest cycles with the relevant information"
    q3 = td.get_largest_cycle(df=data_dict['car_0'], n=3, top=True)
    for idx, row in q3.iterrows():
        print "{}: minima:{}, maxima:{}, amplitude:{}, cycle length:{}".format(idx+1, row.minima, row.maxima, row.amplitude, row.cycle_length)

    print "\n ******************** PART II ********************"
    print "The 2 vehicles are {}".format(td.find_two_vehicles(data_dict, plot=True))
