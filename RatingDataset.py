import scipy.sparse as sp
import numpy as np
from torch.utils.data import Dataset

class TestDataset(Dataset):

    def __init__(self, train_dataset, test_data_path):
        self.feature_size = train_dataset.feature_size
        self.metapath_list = train_dataset.metapath_list
        self.features = train_dataset.features

        self.type2id = train_dataset.type2id
        self.id2type = train_dataset.id2type

        self.load_test_ratings(test_data_path)

        assert len(self.test_ratings_list) == len(self.test_neg_ratings_list)

        self.max_test_item_num = 0

        for i in range(len(self.test_ratings_list)):
            length = len(self.test_ratings_list[i][1:]) + len(self.test_neg_ratings_list[i])

            if length > self.max_test_item_num:
                self.max_test_item_num = length

        print("test ratings %d, test negative rating lists %d, maximum test item number for one rating %d" % (
            len(self.test_ratings_list), len(self.test_neg_ratings_list), self.max_test_item_num))

    def load_test_ratings(self, filename):
        self.test_ratings_list = []
        self.test_neg_ratings_list = []
        with open(filename, "r") as input:
            for line in input.read().splitlines():
                tokens = line.split(",")

                # id starts from 0
                true_ratings = [int(tokens[0])]
                true_ratings.extend([i for i in list(map(int, tokens[1].split(" ")))])
                negative_ratings = [i for i in list(map(int, tokens[2].split(" ")))]

                self.test_ratings_list.append(true_ratings)
                self.test_neg_ratings_list.append(negative_ratings)

    def __len__(self):
        return len(self.test_ratings_list)

    def __getitem__(self, idx):

        negative_id_num = len(self.test_neg_ratings_list[idx])
        real_test_item_size = len(self.test_ratings_list[idx][1:]) + negative_id_num  # self.max_test_item_num

        real_test_item_sizes = np.empty(self.max_test_item_num, dtype=int)
        real_test_item_sizes.fill(real_test_item_size)

        positive_item_indices = np.empty(self.max_test_item_num, dtype=int)
        positive_item_indices.fill(negative_id_num)

        test_item_ids = [0] * self.max_test_item_num

        u = self.test_ratings_list[idx][0]

        test_item_ids[0:negative_id_num] = self.test_neg_ratings_list[idx]
        test_item_ids[negative_id_num:real_test_item_size] = self.test_ratings_list[idx][1:]
        # test_item_ids = tuple(test_item_ids)

        user_input = np.zeros(self.max_test_item_num, dtype=int)
        item_input = np.zeros(self.max_test_item_num, dtype=int)
        metapath_input_list = []

        for i in range(len(self.metapath_list)):
            # metapath_list[i]: metapath_file, path_dict, max_path_num, hop_num
            metapath_input_list.append(
                np.zeros((self.max_test_item_num, self.metapath_list[i][2], self.metapath_list[i][3],
                          self.feature_size), dtype=np.float32))

        k = 0
        # negative item ids
        for i in self.test_neg_ratings_list[idx]:
            user_input[k] = u
            item_input[k] = i

            for metapath_idx in range(len(self.metapath_list)):

                if (u, i) in self.metapath_list[metapath_idx][1]:
                    for p_i in range(len(self.metapath_list[metapath_idx][1][(u, i)])):
                        for p_j in range(len(self.metapath_list[metapath_idx][1][(u, i)][p_i])):
                            type_id = self.metapath_list[metapath_idx][1][(u, i)][p_i][p_j][0]
                            node_id = self.metapath_list[metapath_idx][1][(u, i)][p_i][p_j][1]
                            node_type = self.id2type[type_id]
                            metapath_input_list[metapath_idx][k][p_i][p_j] = self.features[node_type][node_id]

            k += 1

        # positive item ids
        for i in self.test_ratings_list[idx][1:]:
            user_input[k] = u
            item_input[k] = i

            for metapath_idx in range(len(self.metapath_list)):

                if (u, i) in self.metapath_list[metapath_idx][1]:
                    for p_i in range(len(self.metapath_list[metapath_idx][1][(u, i)])):
                        for p_j in range(len(self.metapath_list[metapath_idx][1][(u, i)][p_i])):
                            type_id = self.metapath_list[metapath_idx][1][(u, i)][p_i][p_j][0]
                            node_id = self.metapath_list[metapath_idx][1][(u, i)][p_i][p_j][1]
                            node_type = self.id2type[type_id]
                            metapath_input_list[metapath_idx][k][p_i][p_j] = self.features[node_type][node_id]

            k += 1

        # metapath_input_list[i]: metapath_file, path_dict, path_num, hop_num, feature_size
        # data = [real_test_item_size, positive_item_indices, test_item_ids, user_input, item_input]
        data = [real_test_item_sizes, positive_item_indices, np.array(test_item_ids), user_input, item_input]
        data.extend(metapath_input_list)

        return tuple(data)


class TrainDataset(Dataset):

    def __init__(self, train_data_path, metapath_file_paths, negative_num, feature_file_dict):

        self.negative_num = negative_num
        self.load_train_ratings(train_data_path)

        self.load_feature_as_map(feature_file_dict)
        self.load_metapath(metapath_file_paths)

        print("max_user_id %d, max_item_id %d, train ratings %d" % (
        self.max_user_id, self.max_item_id, self.train_rating_mat.nnz))

    def __len__(self):
        return self.train_rating_mat.nnz

    def __getitem__(self, idx):

        u, i = self.user_item_pairs[idx]
        user_input = np.zeros(self.negative_num + 1, dtype=int)
        item_input = np.zeros(self.negative_num + 1, dtype=int)
        labels = np.zeros(self.negative_num + 1, dtype=np.float32)

        counter = 0
        user_input[counter] = u
        item_input[counter] = i
        labels[counter] = 1

        # metapath: (metapath_file, path_dict, path_num, hop_num)
        metapath_input_list = []
        for metapath in self.metapath_list:
            # PyTorch uses row-wist representation
            metapath_input = np.zeros((self.negative_num + 1, metapath[2], metapath[3], self.feature_size),
                                      dtype=np.float32)

            if (u, i) in metapath[1]:
                for p_i in range(len(metapath[1][(u, i)])):
                    for p_j in range(len(metapath[1][(u, i)][p_i])):
                        type_id = metapath[1][(u, i)][p_i][p_j][0]
                        node_id = metapath[1][(u, i)][p_i][p_j][1]
                        node_type = self.id2type[type_id]
                        metapath_input[counter][p_i][p_j] = self.features[node_type][node_id]

            metapath_input_list.append(metapath_input)

        for t in range(self.negative_num):
            counter += 1

            j = np.random.randint(1, self.max_item_id + 1)
            while j in self.user_item_map[u]:
                j = np.random.randint(1, self.max_item_id + 1)

            user_input[counter] = u
            item_input[counter] = j
            labels[counter] = 0

            for list_index in range(len(metapath_input_list)):

                if (u, j) in self.metapath_list[list_index][1]:
                    for p_i in range(len(self.metapath_list[list_index][1][(u, j)])):
                        for p_j in range(len(self.metapath_list[list_index][1][(u, j)][p_i])):
                            type_id = self.metapath_list[list_index][1][(u, j)][p_i][p_j][0]
                            node_id = self.metapath_list[list_index][1][(u, j)][p_i][p_j][1]
                            node_type = self.id2type[type_id]
                            metapath_input_list[list_index][counter][p_i][p_j] = self.features[node_type][node_id]

        data = [user_input, item_input, labels]
        data.extend(metapath_input_list)
        return tuple(data)

    def load_train_ratings(self, filename):
        self.max_user_id, self.max_item_id = 0, 0

        with open(filename, "r") as input:
            for line in input.read().splitlines():
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                self.max_user_id = max(self.max_user_id, u)
                self.max_item_id = max(self.max_item_id, i)

        # id starts from 1, add one more id 0 for invalid updates
        shape = (self.max_user_id + 1, self.max_item_id + 1)

        self.train_rating_mat = sp.dok_matrix(shape, dtype=np.float32)

        self.user_item_map = {}
        self.item_user_map = {}
        self.user_item_pairs = []

        with open(filename, "r") as input:
            for line in input.read().splitlines():
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])

                self.train_rating_mat[user, item] = 1.0

                if user not in self.user_item_map:
                    self.user_item_map[user] = {}
                if item not in self.item_user_map:
                    self.item_user_map[item] = {}

                self.user_item_map[user][item] = 1.0
                self.item_user_map[item][user] = 1.0

                self.user_item_pairs.append([user, item])

    def load_feature_as_map(self, feature_file_dict):

        self.features = {}
        self.node_sizes = {}
        self.feature_size = -1
        for feature_type, feature_file_path in feature_file_dict.items():
            with open(feature_file_path) as input:
                count = 0
                for line in input.read().splitlines():
                    line = line.strip()
                    if line == "":
                        continue
                    count += 1
                    arr = line.split(',')
                    if self.feature_size == -1:
                        self.feature_size = len(arr) - 1
                    else:
                        assert (self.feature_size == (len(arr) - 1))

                self.node_sizes[feature_type] = count

        for feature_type, feature_file_path in feature_file_dict.items():
            self.features[feature_type] = np.zeros((self.node_sizes[feature_type] + 1, self.feature_size),
                                                   dtype=np.float32)
            with open(feature_file_path) as input:
                for line in input.readlines():
                    line = line.strip()
                    if line == "":
                        continue

                    arr = line.strip().split(',')
                    node_id = int(arr[0])

                    for j in range(len(arr[1:])):
                        self.features[feature_type][node_id][j] = float(arr[j + 1])

    def load_metapath(self, metapath_files):

        self.type2id = {}
        self.id2type = {}
        tmp_type_set = set()

        for metapath_file in metapath_files:
            with open(metapath_file) as input:
                for line in input.read().splitlines():
                    arr = line.split('\t')
                    for path in arr[2:]:
                        nodes = path.split(' ')[0].split('-')

                        for node in nodes:
                            tmp_type_set.add(node[0])

        for node_type in tmp_type_set:
            id = len(self.id2type)
            self.type2id[node_type] = id
            self.id2type[id] = node_type

        print("node type " + str(self.type2id))

        self.metapath_list = []

        for metapath_file in metapath_files:
            path_dict = {}
            max_path_num = 0
            hop_num = 0

            with open(metapath_file) as input:
                for line in input.read().splitlines():
                    arr = line.split('\t')
                    max_path_num = max(int(arr[1]), max_path_num)
                    hop_num = len(arr[2].strip().split('-'))

            with open(metapath_file) as input:
                for line in input.read().splitlines():
                    arr = line.strip().split('\t')
                    u, i = arr[0].split(',')
                    u, i = int(u), int(i)

                    path_dict[(u, i)] = []

                    for path in arr[2:]:
                        tmp = path.split(' ')[0].split('-')
                        node_list = []
                        for node in tmp:
                            index = int(node[1:]) - 1

                            node_list.append([self.type2id[node[0]], index])
                        path_dict[(u, i)].append(node_list)

            self.metapath_list.append((metapath_file, path_dict, max_path_num, hop_num))
