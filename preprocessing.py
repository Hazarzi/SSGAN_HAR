import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sklearn
from sklearn.utils import shuffle
from scipy import signal
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas import Series
from sklearn.model_selection import train_test_split


def sliding_window(x, y, window_size, step_size):
    print('Sliding all over the data! (Sliding windows)')
    array_end_Flag = False
    array_counter = 0
    append_counter = 0

    data_x_sliced = np.empty((int((len(x) / window_size) * (window_size / step_size)), window_size, np.shape(x)[1]))
    data_y_sliced = np.empty((int((len(y) / window_size) * (window_size / step_size)), window_size))

    while array_end_Flag is False:
        try:
            slice_i_y = y[array_counter * step_size: (array_counter * step_size) + window_size]
            if np.all(slice_i_y == slice_i_y[0]):
                data_y_sliced[append_counter] = slice_i_y
                data_x_sliced[append_counter] = x[array_counter * step_size: (array_counter * step_size) + window_size, :]
                array_counter += 1
                append_counter += 1
            else:
                # print("End of class", end='\r')
                array_counter += 1
                continue
        except:
            array_end_Flag = True
            break

    data_x_sliced = data_x_sliced[:append_counter]
    data_y_sliced = data_y_sliced[:append_counter]
    for i in np.arange(len(data_y_sliced)):
        data_y_sliced[i] = np.bincount(data_y_sliced[i].astype(int)).argmax()
    data_y_sliced = np.delete(data_y_sliced, np.s_[1:], 1)

    return data_x_sliced, np.ravel(data_y_sliced.astype(int))

def sliding_window_nolabel(x, window_size, step_size):
    print('Sliding all over the data! (Sliding windows)')
    array_end_Flag = False
    array_counter = 0
    append_counter = 0

    data_x_sliced = np.empty((int((len(x) / window_size) * (window_size / step_size)), window_size, np.shape(x)[1]))

    while array_end_Flag is False:
        try:
            data_x_sliced[append_counter] = x[array_counter * step_size: (array_counter * step_size) + window_size, :]
            array_counter += 1
            append_counter += 1
        except:
            array_end_Flag = True
            break

    data_x_sliced = data_x_sliced[:append_counter]

    return data_x_sliced


def del_labels(data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    return np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


NORM_MAX_THRESHOLDS = [202.0, 35.5, 47.6314, 155.532, 157.76, 45.5484, 62.2598, 61.728, 21.8452,
                       13.1222, 14.2184, 137.544, 109.181, 100.543, 38.5625, 26.386, 153.582,
                       37.2936, 23.9101, 61.9328, 36.9676, 15.5171, 5.97964, 2.94183, 80.4739,
                       39.7391, 95.8415, 35.4375, 157.232, 157.293, 150.99, 61.9509, 62.0461,
                       60.9357, 17.4204, 13.5882, 13.9617, 91.4247, 92.867, 146.651]

NORM_MIN_THRESHOLDS = [0., 0., -114.755, -104.301, -73.0384, -61.1938, -61.8086, -61.4193, -27.8044,
                       -17.8495, -14.2647, -103.941, -200.043, -163.608, 0., -29.0888, -38.1657, -57.2366,
                       -32.9627, -39.7561, -56.0108, -10.1563, -5.06858, -3.99487, -70.0627, -122.48,
                       -66.6847, 0., -155.068, -155.617, -156.179, -60.3067, -61.9064, -62.2629, -14.162,
                       -13.0401, -14.0196, -172.865, -137.908, -102.232]


def normalize(data, max_list, min_list):
    """Normalizes sensor channels to a range [0,1]

    :param data: numpy integer matrix
        Sensor data
    :param max_list: numpy integer array
        Array containing maximums values for every one of the 40 sensor channels
    :param min_list: numpy integer array
        Array containing minimum values for every one of the 40 sensor channels
    :return:
        Normalized sensor data
    """
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (2 * (data[:, i] - min_list[i]) / diffs[i]) - 1

    data[data > 1] = 1
    data[data < -1] = -1
    return data

def downsampling(data_x):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_x.shape[0], 3)

    return data_x[idx]

def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


class Subject_LISSI:

    def __init__(self):
        self.folder = "LISSI_Dataset"
        self.subject_list = {}
        self.repetition_folders = []
        for i in os.listdir(self.folder):
            self.repetition_folders = os.listdir(os.path.join(self.folder, i))
            self.subject_list[i] = self.repetition_folders
        self.header_list = [x.lower() for x in
                            ['PacketCounter', 'time', 'SampleTimeFine', 'Acc_X', 'Acc_Y', 'Acc_Z', 'VelInc_X',
                             'VelInc_Y', 'VelInc_Z', 'OriInc_q0', 'OriInc_q1', 'OriInc_q2', 'OriInc_q3',
                             'FreeAcc_X', 'FreeAcc_Y', 'FreeAcc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Mag_X', 'Mag_Y',
                             'Mag_Z', 'Roll', 'Pitch', 'Yaw']]
        self.cols_to_remove = ['time', 'sampletimefine', 'packetcounter', 'scracc_x', 'scracc_y', 'scracc_z',
                               'scrgyr_x',
                               'scrgyr_y', 'scrgyr_z', 'mag_x', 'mag_y', 'mag_z', 'gyr_x', 'gyr_y', 'gyr_z',
                               'freeacc_x', 'freeacc_y', 'freeacc_z', 'velinc_x', 'velinc_y', 'velinc_z']

    def load_subjects(self, subject_list):
        print(subject_list)
        data_y = []
        data_x = []
        label_tmp = []
        for key, value in self.subject_list.items():
            #if re.search(r'\b' + key + r'\b', subject_list):
            if key in subject_list:
                print(key)
                rep_list = []
                label_rep_list = []
                for i in value:
                    print(i)
                    tmp_list = []
                    for csv in os.listdir(os.path.join(self.folder, key, i)):
                        if 'MT' in csv and csv.endswith(".csv"):
                            print(csv)
                            csv_path = os.path.join(self.folder, key, i, csv)
                            print(csv_path)
                            print("Loading " + csv)
                            tmp = pd.read_csv(csv_path, comment="/", delimiter=",", skiprows=9, encoding='cp855')
                            print('removing cols')
                            tmp.columns = map(str.lower, tmp.columns)
                            for col in self.cols_to_remove:
                                if col in tmp.columns:
                                    tmp = tmp.drop(labels=col, axis=1)
                            print("Loaded.")
                            tmp = tmp.to_numpy()
                            tmp_list.append(tmp)

                        elif 'Rep' and "csv" in csv:
                            csv_path = os.path.join(self.folder, key, i, csv)
                            print("Loading rep read" + csv_path)
                            label_tmp = np.genfromtxt(csv_path, delimiter=',', dtype=str, comments='//', encoding='cp855')[1:, :]
                            print("Loaded rep read")

                    length_list = [len(label_tmp)]
                    for _, temp in enumerate(tmp_list):
                        length_list.append(len(temp))
                    minimum_val = min(length_list)
                    for _, temp in enumerate(tmp_list):
                        tmp_list[_] = temp[:minimum_val]
                    label_tmp = label_tmp[:minimum_val]

                    tmp_list = np.concatenate(tmp_list, axis=1)

                    rep_list.append(tmp_list)
                    label_rep_list.append(label_tmp)
                for b in rep_list:
                    print(b.shape)
                rep_list = np.vstack(rep_list)
                data_x.append(rep_list)
                label_rep_list = np.vstack(label_rep_list)
                data_y.append(label_rep_list)
        data_x = np.vstack(data_x)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_x = imp.fit_transform(data_x)
        #data_x = pd.DataFrame(data_x).fillna(method="bfill").fillna(method="ffill")
        #data_x = data_x.to_numpy()
        data_y = np.vstack(data_y)
        data_y = data_y[:, 2]

        return data_x, data_y


class Subject_PAMAP:

    def __init__(self):
        self.folder = "PAMAP2_Dataset/Protocol"
        self.subject_list = {}
        for i in os.listdir(self.folder):
            self.subject_list[i] = i
        self.cols_to_remove = [0, 16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]
        # self.cols_to_remove = [0, 7,8,9,24,25,26,41,42,43,  16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]

    def load_subjects(self, subject_list):
        tmp_list = []
        for key, value in self.subject_list.items():
            if key in subject_list:
                csv_path = os.path.join(self.folder, key)
                tmp = pd.read_csv(csv_path, delimiter=" ", skiprows=0)
                tmp.drop(tmp.columns[self.cols_to_remove], axis=1, inplace=True)
                print("Loaded.")
                tmp.fillna(method="ffill", inplace=True)
                tmp.fillna(method="bfill", inplace=True)
                tmp = tmp.to_numpy()
                tmp = downsampling(tmp)
                tmp_list.append(tmp)

        tmp_list = np.vstack(tmp_list)
        data_x = tmp_list[:, 1:]
        data_y = tmp_list[:, 0]

        #data_x, data_y = downsampling(data_x, data_y)

        data_y = data_y.astype(int)

        data_x, data_y = del_labels(data_x, data_y)

        data_y[data_y == 24] = 0
        data_y[data_y == 12] = 8
        data_y[data_y == 13] = 9
        data_y[data_y == 16] = 10
        data_y[data_y == 17] = 11

        print(data_x.shape)

        #data_x = complete_HR(data_x)
        data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)

        print(data_x.shape)

        #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        #data_x = imp.fit_transform(data_x)

        return data_x, data_y


class Subject_OPPO:

    def __init__(self):
        self.folder = "oppo_Dataset"
        self.subject_list = {}
        for i in os.listdir(self.folder):
            self.subject_list[i] = i
        # Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
        self.NB_SENSOR_CHANNELS = 113
        
        # Hardcoded names of the files defining the OPPORTUNITY challenge data. As named in the original data.
        self.OPPORTUNITY_DATA_FILES = ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                                  'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                                  'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                                  'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                                  'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                                  'OpportunityUCIDataset/dataset/S1-ADL5.dat',
                                  
                                  'OpportunityUCIDataset/dataset/S2-Drill.dat',
                                  'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                                  'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                                  'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                                  'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                                  'OpportunityUCIDataset/dataset/S2-ADL5.dat',
                                  
                                  'OpportunityUCIDataset/dataset/S3-Drill.dat',
                                  'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                                  'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                                  'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                                  'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                                  'OpportunityUCIDataset/dataset/S3-ADL5.dat',
                                  
                                  'OpportunityUCIDataset/dataset/S4-Drill.dat',
                                  'OpportunityUCIDataset/dataset/S4-ADL1.dat',
                                  'OpportunityUCIDataset/dataset/S4-ADL2.dat',
                                  'OpportunityUCIDataset/dataset/S4-ADL3.dat',
                                  'OpportunityUCIDataset/dataset/S4-ADL4.dat',
                                  'OpportunityUCIDataset/dataset/S4-ADL5.dat'
                                  ]
        
        self.NORM_MAX_THRESHOLDS = [3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                       250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                       10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                       200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                       10000,  10000,  10000,  10000,  250, ]

        self.NORM_MIN_THRESHOLDS = [-3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                           -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                           -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                           -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                           -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                           -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                           -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                           -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                           -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                           -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                           -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                           -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                           -10000, -10000, -10000, -10000, -250, ]

    def select_columns_opp(self,data):
        """Selection of the 113 columns employed in the OPPORTUNITY challenge
        :param data: numpy integer matrix
            Sensor data (all features)
        :return: numpy integer matrix
            Selection of features
        """
    
        #                     included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(data, features_delete, 1)
    
    def divide_x_y(self, data, label):
        """Segments each sample into features and label
        :param data: numpy integer matrix
            Sensor data
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer matrix, numpy integer array
            Features encapsulated into a matrix and labels as an array
        """
    
        data_x = data[:, 1:114]
        if label not in ['locomotion', 'gestures']:
                raise RuntimeError("Invalid label: '%s'" % label)
        if label == 'locomotion':
            data_y = data[:, 114]  # Locomotion label
        elif label == 'gestures':
            data_y = data[:, 115]  # Gestures label
    
        return data_x, data_y
    
    def adjust_idx_labels(self, data_y, label):
        """Transforms original labels into the range [0, nb_labels-1]
        :param data_y: numpy integer array
            Sensor labels
        :param label: string, ['gestures' (default), 'locomotion']
            Type of activities to be recognized
        :return: numpy integer array
            Modified sensor labels
        """
        if label == 'locomotion':  # Labels for locomotion are adjusted
            data_y [ data_y ==  4 ]  =  3
            data_y [ data_y ==  5 ]  =  4
        elif label == 'gestures':  # Labels for gestures are adjusted
            data_y[data_y == 406516] = 1
            data_y[data_y == 406517] = 2
            data_y[data_y == 404516] = 3
            data_y[data_y == 404517] = 4
            data_y[data_y == 406520] = 5
            data_y[data_y == 404520] = 6
            data_y[data_y == 406505] = 7
            data_y[data_y == 404505] = 8
            data_y[data_y == 406519] = 9
            data_y[data_y == 404519] = 10
            data_y[data_y == 406511] = 11
            data_y[data_y == 404511] = 12
            data_y[data_y == 406508] = 13
            data_y[data_y == 404508] = 14
            data_y[data_y == 408512] = 15
            data_y[data_y == 407521] = 16
            data_y[data_y == 405506] = 17
        return data_y
    
    def normalize(self, data, max_list, min_list):
        """Normalizes all sensor channels
        :param data: numpy integer matrix
            Sensor data
        :param max_list: numpy integer array
            Array containing maximums values for every one of the 113 sensor channels
        :param min_list: numpy integer array
            Array containing minimum values for every one of the 113 sensor channels
        :return:
            Normalized sensor data
        """
        max_list, min_list = np.array(max_list), np.array(min_list)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (2 * (data[:, i] - min_list[i]) / diffs[i]) - 1
        #     Checking the boundaries
        data[data > 1] = 1
        data[data < -1] = -1
        return data

    def process_dataset_file(self, data):
        
        # Select correct columns
        label = 'locomotion'
        data = self.select_columns_opp(data)
    
        # Colums are segmentd into features and labels
        data_x, data_y =  self.divide_x_y(data, label)
        data_y = self.adjust_idx_labels(data_y, label)
        data_y = data_y.astype(int)
        # Perform linear interpolation
        data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    
        # Remaining missing data are converted to zero
        data_x[np.isnan(data_x)] = 0
        return data_x, data_y
    
    def load_subjects(self, subject_list):
        tmp_list_X = []
        tmp_list_y = []
        for key, value in self.subject_list.items():
            if key in subject_list:
                csv_path = os.path.join(self.folder, key)
                for file in os.listdir(csv_path):
                    tmp = np.loadtxt(os.path.join(csv_path, file))
                    tmp_X, tmp_y = self.process_dataset_file(tmp)
                    tmp_list_X.append(tmp_X)
                    tmp_list_y.append(tmp_y)
                    
        tmp_list_X = np.vstack(tmp_list_X)
        #print(tmp_list_X.shape)
        tmp_list_X = self.normalize(tmp_list_X, self.NORM_MAX_THRESHOLDS, self.NORM_MIN_THRESHOLDS)
        tmp_list_y = np.hstack(tmp_list_y)

        return tmp_list_X, tmp_list_y

class Subject_UCI:

    def __init__(self):
        self.folder = "UCI_HAR_Dataset"
        self.subject_list = {}
        for i in os.listdir(self.folder):
            self.subject_list[i] = i
        # self.cols_to_remove = [0, 115]

    def load_subjects(self):
        data_list = []
        data_folder = os.listdir(os.path.join(os.getcwd(), 'UCI_HAR_Dataset/train/Inertial Signals'))
        data_filepath = os.path.join(os.getcwd(), 'UCI_HAR_Dataset/train/Inertial Signals')
        for txt in data_folder:
            tmp = np.loadtxt(os.path.join(data_filepath, txt))
            data_list.append(tmp)
        subject_data = np.hstack(data_list)
        subject_text_file = np.loadtxt(os.path.join(os.getcwd(), 'UCI_HAR_Dataset/train/subject_train.txt'))
        subject_idx_train = np.where(subject_text_file == 1)
        subject_idx_val = np.where(subject_text_file != 1)
        subject_idx_test = np.where(subject_text_file == 3)

        data_y = np.loadtxt(os.path.join(os.getcwd(), 'UCI_HAR_Dataset/train/y_train.txt'))

        data_y[data_y == 1.0] = 0
        data_y[data_y == 2.0] = 1
        data_y[data_y == 3.0] = 2
        data_y[data_y == 4.0] = 3
        data_y[data_y == 5.0] = 4
        data_y[data_y == 6.0] = 5

        train_X = subject_data[subject_idx_train]
        train_y = data_y[subject_idx_train]
        val_X = subject_data[subject_idx_val]
        val_y = data_y[subject_idx_val]
        test_X = subject_data[subject_idx_test]
        test_y = data_y[subject_idx_test]

        return train_X.reshape(len(train_X), 128, 9, 1), train_y, val_X.reshape(len(val_X), 128, 9,
                                                                                1), val_y, test_X.reshape(
            len(test_X), 128, 9, 1), test_y

class Subject_WISDM:

    def __init__(self):
        self.folder = "WISDM_Dataset"
        #self.subject_list = {}
        #for i in os.listdir(self.folder):
            #self.subject_list[i] = i
        #self.cols_to_remove = [0, 16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]
        #self.cols_to_remove = [0, 7,8,9,24,25,26,41,42,43,  16, 17, 18, 19, 33, 34, 35, 36, 50, 51, 52, 53]

        self.NORM_MAX_THRESHOLDS = [20.0, 20.0, 20.0]

        self.NORM_MIN_THRESHOLDS = [-20.0, -20.0, -20.0]


    def normalize(self, data, max_list, min_list):
        """Normalizes sensor channels to a range [0,1]

        :param data: numpy integer matrix
            Sensor data
        :param max_list: numpy integer array
            Array containing maximums values for every one of the 40 sensor channels
        :param min_list: numpy integer array
            Array containing minimum values for every one of the 40 sensor channels
        :return:
            Normalized sensor data
        """
        max_list, min_list = np.array(max_list), np.array(min_list)
        diffs = max_list - min_list
        for i in np.arange(data.shape[1]):
            data[:, i] = (2 * (data[:, i] - min_list[i]) / diffs[i]) - 1

        data[data > 1] = 1
        data[data < -1] = -1
        return data

    def load_subjects(self):
        csv_path_labeled = os.path.join(self.folder,'WISDM_at_v2.0_raw.txt')
        csv_path_unlabeled = os.path.join(self.folder,'WISDM_at_v2.0_unlabeled_raw.txt')
        tmp_labeled = pd.read_csv(csv_path_labeled, delimiter=',', low_memory=False, header=None)
        tmp_labeled[5] = tmp_labeled[5].astype(str).map(lambda x: x.rstrip(';')).astype(float)
        tmp_unlabeled = pd.read_csv(csv_path_unlabeled, delimiter=',', low_memory=False, header=None)
        tmp_unlabeled[5] = tmp_unlabeled[5].astype(str).map(lambda x: x.rstrip(';')).astype(float)
        
        tmp_labeled_train = tmp_labeled.loc[tmp_labeled[0] == 1319]
        tmp_labeled_test = tmp_labeled.loc[tmp_labeled[0] != 1319]
        
        tmp_labeled_train = tmp_labeled_train.drop([2], axis=1)
        tmp_labeled_test = tmp_labeled_test.drop([2], axis=1)
        tmp_unlabeled = tmp_unlabeled.drop([2], axis=1)
        
        tmp_labeled_train.fillna(method="ffill", inplace=True)
        tmp_labeled_train.fillna(method="bfill", inplace=True)
        
        tmp_labeled_test.fillna(method="ffill", inplace=True)
        tmp_labeled_test.fillna(method="bfill", inplace=True)
        
        tmp_unlabeled.fillna(method="ffill", inplace=True)
        tmp_unlabeled.fillna(method="bfill", inplace=True)
        
        tmp_labeled_train = tmp_labeled_train.drop([0], axis=1)
        tmp_labeled_test = tmp_labeled_test.drop([0], axis=1)
        tmp_unlabeled = tmp_unlabeled.drop([0], axis=1)
         
        tmp_labeled_train = tmp_labeled_train.to_numpy()
        tmp_labeled_test = tmp_labeled_test.to_numpy()
        tmp_unlabeled = tmp_unlabeled.to_numpy()
        
        data_x_labeled_train = tmp_labeled_train[:, -3:]
        data_y_labeled_train = tmp_labeled_train[:, 0]
        
        data_x_labeled_test = tmp_labeled_test[:, -3:]
        data_y_labeled_test = tmp_labeled_test[:, 0]
        
        data_x_unlabeled = tmp_unlabeled[:, -3:]
        data_y_unlabeled = tmp_unlabeled[:, 0]
        print(data_y_unlabeled)
        
        le = LabelEncoder()
        le.fit(data_y_labeled_train)
        data_y_labeled_train= le.transform(data_y_labeled_train)
        data_y_labeled_test = le.transform(data_y_labeled_test)

        return data_x_labeled_train, data_y_labeled_train, data_x_labeled_test, data_y_labeled_test, data_x_unlabeled, data_y_unlabeled

def load_data(dataset):

    if dataset == 'LISSI':

        acc_idx_list = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42]
        oriinc_idx_list = [3, 4, 5, 6, 13, 14, 15, 16, 23, 24, 25, 26, 33, 34, 35, 36, 43, 44, 45, 46]
        axis_idx_list = [7, 8, 9, 17, 18, 19, 27, 28, 29, 37, 38, 39, 47, 48, 49]

        #subjects_to_load = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 7',
        #                    'Subject 8', 'Subject 9','Subject 10','Subject 11', 'Subject 13',
        #                    'Subject 14', 'Subject 15', 'Subject 16']

        s = Subject_LISSI()

        train_data = ['Subject 1']
        print("train_data")
        print(train_data)
        ul_data =['Subject 1', 'Subject 2', 'Subject 3', 'Subject 7',
                            'Subject 8', 'Subject 9','Subject 10','Subject 11', 'Subject 13',
                            'Subject 14', 'Subject 15', 'Subject 16']

        validation_data = ['Subject 5', 'Subject 17' ]
        print("validation_data")
        print(validation_data)
        test_data = ['Subject 6', 'Subject 20']
        print("test_data")
        print(test_data)

        X_train_raw, y_train_f_s = s.load_subjects(train_data)
        X_train_ul_raw, y_train_ul_f_s = s.load_subjects(ul_data)
        X_val_raw, y_val_f_s = s.load_subjects(validation_data)
        X_test_raw, y_test_f_s = s.load_subjects(test_data)

        print("==================================")

        print(X_train_raw.shape)
        print(X_val_raw.shape)
        print(X_test_raw.shape)

        print("==================================")

        le = sklearn.preprocessing.LabelEncoder()
        le.fit(y_train_f_s)
        y_train_f_s = le.transform(y_train_f_s)
        y_train_ul_f_s = le.transform(y_train_ul_f_s)
        y_val_f_s = le.transform(y_val_f_s)
        y_test_f_s = le.transform(y_test_f_s)

        X_train_f = np.copy(X_train_raw)
        X_train_ul_f = np.copy(X_train_ul_raw)
        X_val_f = np.copy(X_val_raw)
        X_test_f = np.copy(X_test_raw)

        def butter_low_filter(data, cutoff=15):
            sos = signal.butter(6, cutoff, 'lp', fs=60, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        def butter_high_filter(data, cutoff=0.004):
            sos = signal.butter(6, cutoff, 'hp', fs=60, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        for acc in acc_idx_list:
            X_train_f[:, acc] = butter_high_filter(X_train_f[:, acc], cutoff=0.01)
            X_train_ul_f[:, acc] = butter_high_filter(X_train_ul_f[:, acc], cutoff=0.01)
            X_val_f[:, acc] = butter_high_filter(X_val_f[:, acc], cutoff=0.01)
            X_test_f[:, acc] = butter_high_filter(X_test_f[:, acc], cutoff=0.01)

        for axis in axis_idx_list:
            X_train_f[:, axis] = butter_high_filter(X_train_f[:, axis], cutoff=0.01)
            X_train_ul_f[:, axis] = butter_high_filter(X_train_ul_f[:, axis], cutoff=0.01)            
            X_val_f[:, axis] = butter_high_filter(X_val_f[:, axis], cutoff=0.01)
            X_test_f[:, axis] = butter_high_filter(X_test_f[:, axis], cutoff=0.01)

        for ori in oriinc_idx_list:
            X_train_f[:, ori] = butter_high_filter(X_train_f[:, ori], cutoff=0.01)
            X_train_ul_f[:, ori] = butter_high_filter(X_train_ul_f[:, ori], cutoff=0.01)                       
            X_val_f[:, ori] = butter_high_filter(X_val_f[:, ori], cutoff=0.01)
            X_test_f[:, ori] = butter_high_filter(X_test_f[:, ori], cutoff=0.01)

        for acc in acc_idx_list:
            X_train_f[:, acc] = butter_low_filter(X_train_f[:, acc], cutoff=1)
            X_train_ul_f[:, acc] = butter_low_filter(X_train_ul_f[:, acc], cutoff=1)                                  
            X_val_f[:, acc] = butter_low_filter(X_val_f[:, acc], cutoff=1)
            X_test_f[:, acc] = butter_low_filter(X_test_f[:, acc], cutoff=1)

        for axis in axis_idx_list:
            X_train_f[:, axis] = butter_low_filter(X_train_f[:, axis], cutoff=1)
            X_train_ul_f[:, axis] = butter_low_filter(X_train_ul_f[:, axis], cutoff=1)                                  
            X_val_f[:, axis] = butter_low_filter(X_val_f[:, axis], cutoff=1)
            X_test_f[:, axis] = butter_low_filter(X_test_f[:, axis], cutoff=1)

        for ori in oriinc_idx_list:
            X_train_f[:, ori] = butter_low_filter(X_train_f[:, ori], cutoff=1)
            X_train_ul_f[:, ori] = butter_low_filter(X_train_ul_f[:, ori], cutoff=1)                                  
            X_val_f[:, ori] = butter_low_filter(X_val_f[:, ori], cutoff=1)
            X_test_f[:, ori] = butter_low_filter(X_test_f[:, ori], cutoff=1)

        #scale_values = np.vstack((X_train_f, X_train_ul_f))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X_train_f)
        X_train_f_s_r = scaler.transform(X_train_f)
        X_train_ul_f_s_r = scaler.transform(X_train_ul_f)
        X_val_f_s_r = scaler.transform(X_val_f)
        X_test_f_s_r = scaler.transform(X_test_f)

        X_train_f_s_r, y_train_f_s = sliding_window(X_train_f_s_r, y_train_f_s, 90, 27)
        X_train_ul_f_s_r, y_train_ul_f_s = sliding_window(X_train_ul_f_s_r, y_train_ul_f_s, 90, 27)
        #X_train_ul_f_s_r, X_train_f_s_r, y_train_ul_f_s,  y_train_f_s  = train_test_split(X_train_f_s_r, y_train_f_s, test_size=500, stratify=y_train_f_s, shuffle=True, random_state=42)
        X_val_f_s_r, y_val_f_s = sliding_window(X_val_f_s_r, y_val_f_s, 90, 27)
        X_test_f_s_r, y_test_f_s = sliding_window(X_test_f_s_r, y_test_f_s, 90, 27)

    elif dataset == 'PAMAP2':

        acc_idx_list = [2,3,4, 5,6,7, 8,9, 10, 11,12 , 13,  15, 16, 17 , 18,19,20 ,21,22,23, 24,25,26 ,  28,29,30 ,31,32,33, 34,35,36,37,38,39]
        other_list = [0, 1, 14, 27]
        random_seed = 42

        s = Subject_PAMAP()

        print("==================================")

        print("Training with all in ul")

        print("==================================")

        X_train_raw, y_train_f_s = s.load_subjects(["subject101.dat"])
        print("Train: subject101.dat")
        # 
        X_train_ul_raw, y_train_ul_f_s = s.load_subjects(["subject102.dat", "subject103.dat", "subject104.dat", "subject107.dat","subject108.dat", "subject109.dat"])
        # 
        X_val_raw, y_val_f_s = s.load_subjects(["subject105.dat"])
        print("Val: subject101 vs others")
        X_test_raw, y_test_f_s = s.load_subjects(["subject106.dat"])
        print("Test: subject108.dat")

        X_train_f = np.copy(X_train_raw)
        X_train_ul_f = np.copy(X_train_ul_raw)
        X_val_f = np.copy(X_val_raw)
        X_test_f = np.copy(X_test_raw)

        def butter_low_filter(data, cutoff=15):
            sos = signal.butter(6, cutoff, 'lp', fs=33.3, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        def butter_high_filter(data, cutoff=0.004):
            sos = signal.butter(6, cutoff, 'hp', fs=33.3, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        for acc in acc_idx_list:
            X_train_f[:, acc] = butter_high_filter(X_train_f[:, acc], cutoff=0.001)
            X_train_ul_f[:, acc] = butter_high_filter(X_train_ul_f[:, acc], cutoff=0.001)
            X_val_f[:, acc] = butter_high_filter(X_val_f[:, acc], cutoff=0.001)
            X_test_f[:, acc] = butter_high_filter(X_test_f[:, acc], cutoff=0.001)
        
        for others in other_list:
            X_train_f[:, others] = butter_low_filter(X_train_f[:, others], cutoff=0.1)
            X_train_ul_f[:, others] = butter_low_filter(X_train_ul_f[:, others], cutoff=0.1)
            X_val_f[:, others] = butter_low_filter(X_val_f[:, others], cutoff=0.1)
            X_test_f[:, others] = butter_low_filter(X_test_f[:, others], cutoff=0.1)

        #MINMAX KULLANMA
        #scaler = sklearn.preprocessing.StandardScaler ()
        #scaler = scaler.fit(X_train_ul_f)
        #X_train_f= scaler.transform(X_train_f)
        #X_train_ul_f = scaler.transform(X_train_ul_f)
        #X_val_f = scaler.transform(X_val_f)
        #X_test_f= scaler.transform(X_test_f)
        
        X_train_f_s_r, y_train_f_s = sliding_window(X_train_f, y_train_f_s, 90, 27)
        #X_train_f_s_r, X_val_f_s_r, y_train_f_s,  y_val_f_s  = train_test_split(X_train_f_s_r, y_train_f_s, test_size=0.10, stratify=y_train_f_s, shuffle=True, random_state=42)
        X_train_ul_f_s_r, y_train_ul_f_s =  sliding_window(X_train_ul_f, y_train_ul_f_s, 90, 27)
        #X_train_ul_f_s_r = np.vstack((a, X_train_ul_f_s_r))
        #y_train_ul_f_s = np.concatenate((b, y_train_ul_f_s))
        X_val_f_s_r, y_val_f_s = sliding_window(X_val_f, y_val_f_s, 90, 27)
        #X_train_ul_f_s_r, X_val_f_s_r, y_train_ul_f_s , y_val_f_s = train_test_split(X_train_ul_f_s_r, y_train_ul_f_s, test_size=0.10, random_state=random_seed)
        X_test_f_s_r, y_test_f_s = sliding_window(X_test_f, y_test_f_s, 90, 27) 
        #_, X_test_f_s_r, _ , y_test_f_s = train_test_split(X_test_f_s_r, y_test_f_s, test_size=0.30, random_state=random_seed)


        print("==================================")

        print(X_train_f_s_r.shape)
        print(X_val_f_s_r.shape)
        print(X_test_f_s_r.shape)

        print("==================================")

    elif dataset == 'OPPO':

        s = Subject_OPPO()
        print("minmax+filter")
        random_seed = 42
        print("random_seed = 42")

        X_train_raw, y_train_f_s = s.load_subjects(["subject1"])
        X_train_ul_raw, y_train_ul_f_s = s.load_subjects(["subject2","subject3","subject4"])
        X_val_raw, y_val_f_s = s.load_subjects(["subject5"])
        X_test_raw, y_test_f_s = s.load_subjects(["subject6"])

        acc_idx_list = [i for i in range(113)]

        X_train_f = np.copy(X_train_raw)
        X_train_ul_f = np.copy(X_train_ul_raw)
        X_val_f = np.copy(X_val_raw)
        X_test_f = np.copy(X_test_raw)

        def butter_high_filter(data, cutoff=0.004):
            sos = signal.butter(6, cutoff, 'hp', fs=30, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        for acc in acc_idx_list:
            X_train_f[:, acc] = butter_high_filter(X_train_f[:, acc], cutoff=0.001)
            X_train_ul_f[:, acc] = butter_high_filter(X_train_ul_f[:, acc], cutoff=0.001)
            X_val_f[:, acc] = butter_high_filter(X_val_f[:, acc], cutoff=0.001)
            X_test_f[:, acc] = butter_high_filter(X_test_f[:, acc], cutoff=0.001)

        #scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler = scaler.fit(X_train_f)
        #X_train_f_s_r = scaler.transform(X_train_f)
        #X_train_ul_f_s_r = scaler.transform(X_train_ul_f)
        #X_val_f_s_r = scaler.transform(X_val_f)
        #X_test_f_s_r = scaler.transform(X_test_f)

        X_train_f_s_r, y_train_f_s = sliding_window(X_train_f, y_train_f_s, 90, 27)
        #X_train_ul_f_s_r, X_train_f_s_r, y_train_ul_f_s,  y_train_f_s  = train_test_split(X_train_f_s_r, y_train_f_s, test_size=200, stratify=y_train_f_s, shuffle=True, random_state=42)
        X_train_ul_f_s_r, y_train_ul_f_s =  sliding_window(X_train_ul_f_s_r, y_train_ul_f_s, 90, 27)
        X_val_f_s_r, y_val_f_s = sliding_window(X_val_f, y_val_f_s, 90, 27)
        X_test_f_s_r, y_test_f_s = sliding_window(X_test_f, y_test_f_s, 90, 27)

        #X_val_f_s_r, X_test_f_s_r, y_val_f_s, y_test_f_s = sklearn.model_selection.train_test_split(X_val_f_s_r, y_val_f_s, test_size=0.2, random_state=42)

    elif dataset == 'UCI':

        s = Subject_UCI()
        X_train_f_s_r, y_train_f_s, X_val_f_s_r, y_val_f_s, X_test_f_s_r, y_test_f_s = s.load_subjects()

        X_train_f_s_r = X_train_f_s_r.reshape((len(X_train_f_s_r) * X_train_f_s_r.shape[1], X_train_f_s_r.shape[2]))
        X_val_f_s_r = X_val_f_s_r.reshape((len(X_val_f_s_r) * X_val_f_s_r.shape[1], X_val_f_s_r.shape[2]))
        X_test_f_s_r = X_test_f_s_r.reshape((len(X_test_f_s_r) * X_test_f_s_r.shape[1], X_test_f_s_r.shape[2]))

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X_train_f_s_r)
        X_train_f_s_r = scaler.transform(X_train_f_s_r)
        X_val_f_s_r = scaler.transform(X_val_f_s_r)
        X_test_f_s_r = scaler.transform(X_test_f_s_r)

        X_train_f_s_r, y_train_f_s = sliding_window(X_train_f_s_r, y_train_f_s, 60, 20)
        X_val_f_s_r, y_val_f_s = sliding_window(X_val_f_s_r, y_val_f_s, 60, 20)
        X_test_f_s_r, y_test_f_s = sliding_window(X_test_f_s_r, y_test_f_s, 60, 20)

    elif dataset == 'WISDM':

        s = Subject_WISDM()
        X_train_raw, y_train_f_s, X_test_raw, y_test_f_s, X_val_raw, y_val_f_s = s.load_subjects()

        acc_idx_list = [i for i in range(3)]

        X_train_f = np.copy(X_train_raw)
        X_val_f = np.copy(X_val_raw)
        X_test_f = np.copy(X_test_raw)

        def butter_high_filter(data, cutoff=0.004):
            sos = signal.butter(6, cutoff, 'hp', fs=20, output='sos')
            filtered = signal.sosfiltfilt(sos, data, axis=0)
            return filtered

        for acc in acc_idx_list:
            X_train_f[:, acc] = butter_high_filter(X_train_f[:, acc], cutoff=0.001)
            X_val_f[:, acc] = butter_high_filter(X_val_f[:, acc], cutoff=0.001)
            X_test_f[:, acc] = butter_high_filter(X_test_f[:, acc], cutoff=0.001)

        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X_train_f)
        X_train_f_s_r = scaler.transform(X_train_f)
        X_val_f_s_r = scaler.transform(X_val_f)
        X_test_f_s_r = scaler.transform(X_test_f)

        X_train_f_s_r, y_train_f_s = sliding_window(X_train_f_s_r, y_train_f_s, 100, 20)
        X_val_f_s_r = sliding_window_nolabel(X_val_f_s_r, 100, 20)
        X_test_f_s_r, y_test_f_s = sliding_window(X_test_f_s_r, y_test_f_s, 100, 20)

    X_train_f_s_r, y_train_f_s = shuffle(X_train_f_s_r, y_train_f_s, random_state=42)
    X_train_ul_f_s_r, y_train_ul_f_s = shuffle(X_train_ul_f_s_r, y_train_ul_f_s, random_state=42)
    X_val_f_s_r, y_val_f_s = shuffle(X_val_f_s_r, y_val_f_s, random_state=42)
    X_test_f_s_r, y_test_f_s = shuffle(X_test_f_s_r, y_test_f_s, random_state=42)

    X_train_f_s_r = X_train_f_s_r.reshape((len(X_train_f_s_r),X_train_f_s_r.shape[1], X_train_f_s_r.shape[2], 1))
    X_train_ul_f_s_r = X_train_ul_f_s_r.reshape((len(X_train_ul_f_s_r),X_train_ul_f_s_r.shape[1], X_train_ul_f_s_r.shape[2], 1))
    X_val_f_s_r = X_val_f_s_r.reshape((len(X_val_f_s_r),X_val_f_s_r.shape[1], X_val_f_s_r.shape[2], 1))
    X_test_f_s_r = X_test_f_s_r.reshape((len(X_test_f_s_r),X_test_f_s_r.shape[1], X_test_f_s_r.shape[2], 1))

    return X_train_f_s_r, y_train_f_s, X_train_ul_f_s_r, y_train_ul_f_s, X_val_f_s_r, y_val_f_s, X_test_f_s_r, y_test_f_s
