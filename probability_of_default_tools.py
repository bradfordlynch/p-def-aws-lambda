import StringIO, time, boto3
#import pandas as pd
import numpy as np
import scipy.optimize as sco
from scipy.stats import norm

def rolling_window(a, window):
    '''
    Use Numpy's stride tricks to create a rolling window over array a

    Args:
        a (ndarray): Numpy array of values to calculate rolling window over
        window (int): Width of window

    Returns:
        ndarray: Array of rolling values
    '''
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_pd_dataframe(key, bucket='p-def'):
    '''
    Loads CSV file of data for a single company from the file key specified.

    Args:
        key (str): Key pointing to file in S3

    Returns:
        dict: Map of column name to numpy array column index
        ndarray: Numpy array of data
    '''
    # Create S3 client and retrieve data based on key
    client = boto3.client('s3')
    obj = client.get_object(Bucket=bucket, Key=key)

    # Read in data
    raw_data = StringIO.StringIO(obj['Body'].read())

    return pd.read_csv(raw_data, index_col=0, parse_dates=True)

def get_data(key, bucket='p-def'):
    '''
    Loads CSV file of data for a single company from the file key specified.

    Args:
        key (str): Key pointing to file in S3

    Returns:
        dict: Map of column name to numpy array column index
        ndarray: Numpy array of data
    '''
    # Create S3 client and retrieve data based on key
    client = boto3.client('s3')
    obj = client.get_object(Bucket=bucket, Key=key)

    # Read in data
    raw_data = obj['Body'].read().splitlines()

    # Create map of column name to index, assuming the first column is dropped
    header_map = {key:value-1 for value,key in enumerate(raw_data[0].split(','))}

    # Discard header row and split off first column (Date column)
    dates = [row.split(',')[0] for row in raw_data[1:]]
    data = [",".join(row.split(',')[1:]) for row in raw_data[1:]]

    # Creat numpy array of data
    data = np.genfromtxt(data, delimiter=',')

    return header_map, dates, data

def save_data(data, key, bucket='p-def'):
    '''
    Writes data for a single company to a file at the specified key.

    Args:
        data (File buffer): File buffer containing data to write to file
        key (str): Key pointing to file in S3

    Returns:
        bool: True if data was successfully written to S3
    '''
    #
    s3 = boto3.resource('s3')
    res = s3.Object(bucket, key).put(Body=data)

    return res

def merge_data_to_csv(header_map, dates, corp_data, results):
    '''
    Combines arguments back into a single CSV file.

    Args:
        header_map (dict): Map of column name to numpy array column index
        dates (list): Date of each observation in the dataset
        corp_data (ndarray): Numpy array of company data
        results (ndarray): Numpy array of same length as company data containing asset value results

    Returns:
        StringIO: File buffer containined data in CSV format
    '''
    csv_file = StringIO.StringIO()

    # Convert header map to map of index to name
    i_to_name = {value+1:key for key, value in header_map.items()}

    # Add column names for new asset value columns
    i_start = max(i_to_name.keys()) + 1
    for i in range(results.shape[1]):
        i_to_name[i_start + i] = 'Va_{:d}'.format(i+1)

    # Create header row
    csv_file.write(','.join([i_to_name[i] for i in range(len(i_to_name))]) + '\n')

    # Combine company data and results and write to CSV
    np.savetxt(csv_file, np.hstack((np.array((dates)).reshape((-1,1)), corp_data, results)), fmt='%s', delimiter=',')

    # Reset file position
    csv_file.seek(0)

    return csv_file


def solve_for_asset_value(corp_data, header_map, time_horizon, min_hist_vals=252):
    '''
    Solves for a firm's asset value based on a time history of the firm's equity
    value, debt level, and risk-free rate.

    Args:
        corp_data (ndarray): Numpy array of company data (Equity value, face value of debt, and risk-free rate)
        header_map (dict): Map of column name to the data column index in corp_data
        time_horizon (list): List of time horizons (In years) to calulate asset value for
        min_hist_vals (int): Minimum number of days to use for calculating historical data

    Returns:
        ndarray: Numpy array of time-series of asset values

    '''
    import scipy.optimize as sco
    from scipy.stats import norm

    def equations(v_a, debug=False):
        d1 = (np.log(v_a/face_val_debt) + (r_f + 0.5*sigma_a**2)*T)/(sigma_a*np.sqrt(T))
        d2 = d1 - sigma_a*np.sqrt(T)

        y1 = v_e - (v_a*norm.cdf(d1) - np.exp(-r_f*T)*face_val_debt*norm.cdf(d2))

        if debug:
            print("d1 = {:.6f}".format(d1))
            print("d2 = {:.6f}".format(d2))
            print("Error = {:.6f}".format(y1))

        return y1

    # Set window width for calculating historical data
    win = 252

    # Set start point of historical data
    start_time = min_hist_vals
    timesteps = range(min_hist_vals, len(corp_data))

    # Calculate historical volatility
    ret_col = header_map['RET']
    sigma_e = np.zeros((corp_data.shape[0]))
    sigma_e[:win-1] = np.nan
    sigma_e[win-1:] = np.std(rolling_window(np.log(corp_data[:,ret_col] + 1), win), axis=-1)

    assert type(time_horizon) in [list, tuple],"time_horizon must be a list"

    # Create array for storing results
    results = np.empty((corp_data.shape[0],len(time_horizon)))

    for i, years in enumerate(time_horizon):
        T = 252*years
        # Set initial guess for firm value equal to the equity value
        results[:,i] = corp_data[:,header_map['mkt_val']]

        # Run through all days
        for i_t, t in enumerate(timesteps):
            # Check if the company is levered
            if corp_data[t,header_map['face_value_debt']] > 1e-10:
                # Company is levered, calculate probability of default
                # Calculate initial guess at sigma_a
                v_a_per = results[t-252:t,i]
                v_a_ret = np.log(v_a_per/np.roll(v_a_per,1))
                v_a_ret[0] = np.nan
                sigma_a = np.nanstd(v_a_ret)

                if i_t == 0:
                    subset_timesteps = range(t-252, t+1)
                else:
                    #subset_timesteps = corp_data.loc[t-pd.Timedelta(20,'D'):t].index
                    subset_timesteps = [t]

                # Iterate on previous values of V_a
                n_its = 0
                while n_its < 10:
                    n_its += 1
                    # Loop over timesteps, calculating Va using initial guess for sigma_a
                    for t_sub in subset_timesteps:
                        r_f = (1 + corp_data[t_sub,header_map['DGS1']])**(1.0/365) - 1
                        v_e = corp_data[t_sub,header_map['mkt_val']]
                        face_val_debt = corp_data[t_sub,header_map['face_value_debt']]
                        sol = sco.root(equations, results[t_sub,i])
                        results[t_sub,i] = sol['x'][0]

                    # Update sigma_a based on new values of Va
                    last_sigma_a = sigma_a
                    v_a_per = results[t-252:t,i]
                    v_a_ret = np.log(v_a_per/np.roll(v_a_per,1))
                    v_a_ret[0] = np.nan
                    sigma_a = np.nanstd(v_a_ret)

                    if abs(last_sigma_a - sigma_a) < 1e-3:
                        #corp_data.loc[t_sub, 'sigma_a'] = sigma_a
                        break
            else:
                # Company is unlevered, Va = Ve
                pass

    return results

def run_model(key, time_horizon=[1,2,3,4,5]):
    '''
    Apply B-S option pricing model to calculate inferred firm asset values as a
    function of time.

    Args:
        key (str): key pointing to data in S3
        time_horizon (list): List of time horizons (In Years) to calculate model over

    Returns:
        float: Time run was started (In unix time)
        float: Time run finished (In unix time)
        dict: Response from S3 write

    '''
    start = time.time()

    # Get data from S3
    h_map, dates, data = get_data(key)

    if len(dates) > 252:
        # Run the simulation
        results = solve_for_asset_value(data, h_map, time_horizon=time_horizon)

        # Merge data back into CSV
        csv_file = merge_data_to_csv(h_map, dates, data, results)

        # Save results to S3
        result_key = key.replace('merged-corp-data', 'merton-results')
        response = save_data(csv_file, result_key)
    else:
        response = False

    end = time.time()

    return start, end, response
