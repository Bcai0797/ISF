import numpy as np
import torch

MAX_TIME=9999
MIN_TIME=-MAX_TIME

class StrDataSequence(torch.utils.data.Dataset):
    def __init__(self, file_name, time_list, time_margin=0, hot_list=None,shuffle=False):
        self.obsv_time = []
        self.surv_time = []
        self.feature = []
        self.time_margin = time_margin
        self.time_list = time_list
        self.max_feat_id = 0
        self.hot_list = hot_list

        with open(file_name, 'r') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line[:-1].replace(':1', '')
                items = [int(x) for x in line.split(' ')[1:]]
                if items[0] <= 0:
                    continue
                self.surv_time.append(items[0])
                self.obsv_time.append(items[1])
                self.feature.append(items[2:])

        self.obsv_time = np.asarray(self.obsv_time, np.float32)
        self.surv_time = np.asarray(self.surv_time, np.float32)
        self.feature = np.asarray(self.feature, np.int)

        print('Max Time: %d' % (max(np.max(self.obsv_time), np.max(self.surv_time))))

    def __len__(self):
        return len(self.obsv_time)

    def __getitem__(self, idx):
        return self.feature[idx], self._get_label(idx), self.obsv_time[idx], self.surv_time[idx], self._get_true_label(idx)

    def _get_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        if self.hot_list is None:
            obsv_t, surv_t = self.obsv_time[idx], self.surv_time[idx]
            l, r = (obsv_t, MAX_TIME) if obsv_t < surv_t else (surv_t - self.time_margin, surv_t + self.time_margin)
            label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        else:
            label = self.hot_list[idx]
        return label

    def _get_true_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        
        surv_t = self.surv_time[idx]
        l, r = surv_t - self.time_margin, surv_t + self.time_margin
        label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        return label

    def get_map_size(self):
        return len(self.__getitem__(0)[0])

    def get_all_labels(self):
        return np.asarray(list(map(self._get_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_true_labels(self):
        return np.asarray(list(map(self._get_true_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_observation_time(self):
        return self.obsv_time

    def get_survival_time(self):
        return self.surv_time

    def get_all_feature(self):
        return self.feature

    def get_censorship(self):
        return np.asarray(list(map(lambda x: self.obsv_time[x] < self.surv_time[x], list(range(len(self.obsv_time))))), dtype=np.int)


class NpyDataSequence(torch.utils.data.Dataset):
    def __init__(self, file_name, time_list, time_margin=0, hot_list=None,shuffle=False):
        self.obsv_time = []
        self.surv_time = []
        self.feature = []
        self.time_margin = time_margin
        self.time_list = time_list
        self.max_feat_id = 0
        self.hot_list = hot_list

        lines = np.load(file_name)
        for items in lines:
            if items[0] <= 0:
                continue
            self.surv_time.append(items[0])
            self.obsv_time.append(items[1])
            self.feature.append(items[2:])

        self.obsv_time = np.asarray(self.obsv_time, np.float32)
        self.surv_time = np.asarray(self.surv_time, np.float32)
        self.feature = np.asarray(self.feature, np.float32)

        print('Max Time: %d' % (max(np.max(self.obsv_time), np.max(self.surv_time))))

    def __len__(self):
        return len(self.obsv_time)

    def __getitem__(self, idx):
        return self.feature[idx], self._get_label(idx), self.obsv_time[idx], self.surv_time[idx], self._get_true_label(idx)

    def _get_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        if self.hot_list is None:
            obsv_t, surv_t = self.obsv_time[idx], self.surv_time[idx]
            l, r = (obsv_t, MAX_TIME) if obsv_t < surv_t else (surv_t - self.time_margin, surv_t + self.time_margin)
            label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        else:
            label = self.hot_list[idx]
        return label

    def _get_true_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        
        surv_t = self.surv_time[idx]
        l, r = surv_t - self.time_margin, surv_t + self.time_margin
        label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        return label

    def get_map_size(self):
        return len(self.__getitem__(0)[0])

    def get_all_labels(self):
        return np.asarray(list(map(self._get_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_true_labels(self):
        return np.asarray(list(map(self._get_true_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_observation_time(self):
        return self.obsv_time

    def get_survival_time(self):
        return self.surv_time

    def get_all_feature(self):
        return self.feature

    def get_censorship(self):
        return np.asarray(list(map(lambda x: self.obsv_time[x] < self.surv_time[x], list(range(len(self.obsv_time))))), dtype=np.int)

class CSVDataSequence(torch.utils.data.Dataset):
    def __init__(self, file_name, time_list, time_margin=0, hot_list=None,shuffle=False):
        self.obsv_time = []
        self.surv_time = []
        self.feature = []
        self.time_margin = time_margin
        self.time_list = time_list
        self.max_feat_id = 0
        self.hot_list = hot_list

        input_x = np.load(file_name)
        
        for x_row in input_x:
            observed_time = int(x_row[0])
            flag_dead = int(x_row[1])
            features = x_row[2:]

            if flag_dead > 0.5:
                self.surv_time.append(observed_time)
                self.obsv_time.append(observed_time+1)
            else:
                self.surv_time.append(MAX_TIME)
                self.obsv_time.append(observed_time)
            self.feature.append(features)

        self.obsv_time = np.asarray(self.obsv_time, np.float32)
        self.surv_time = np.asarray(self.surv_time, np.float32)
        self.feature = np.asarray(self.feature, np.float32)

        print('Max Time: %d' % np.max(self.obsv_time))

    def __len__(self):
        return len(self.obsv_time)

    def __getitem__(self, idx):
        return self.feature[idx], self._get_label(idx), self.obsv_time[idx], self.surv_time[idx], self._get_true_label(idx)

    def _get_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        if self.hot_list is None:
            obsv_t, surv_t = self.obsv_time[idx], self.surv_time[idx]
            l, r = (obsv_t, MAX_TIME) if obsv_t < surv_t else (surv_t - self.time_margin, surv_t + self.time_margin)
            label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        else:
            label = self.hot_list[idx]
        return label

    def _get_true_label(self, idx):
        def _is_intersected(a, b, x, y):
            assert a <= b and x < y
            return (a <= x and b >= x) or (a > x and b < y) or (a <= y and b > y)
        
        surv_t = self.surv_time[idx]
        l, r = surv_t - self.time_margin, surv_t + self.time_margin
        label = np.asarray(list(map(lambda i: 1 if _is_intersected(l, r, self.time_list[i], self.time_list[i+1]) else 0, range(len(self.time_list)-1))), dtype=np.int)
        return label

    def get_map_size(self):
        return len(self.__getitem__(0)[0])

    def get_all_labels(self):
        return np.asarray(list(map(self._get_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_true_labels(self):
        return np.asarray(list(map(self._get_true_label, list(range(len(self.obsv_time))))), dtype=np.int)

    def get_observation_time(self):
        return self.obsv_time

    def get_survival_time(self):
        return self.surv_time

    def get_all_feature(self):
        return self.feature

    def get_censorship(self):
        return np.asarray(list(map(lambda x: self.obsv_time[x] < self.surv_time[x], list(range(len(self.obsv_time))))), dtype=np.int)
