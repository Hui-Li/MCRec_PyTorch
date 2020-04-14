import time

import os
import torch

from torch import nn
from torch.utils.data import DataLoader
from EvaluateUtil import evaluate
from MCRec import MCRec
from RatingDataset import TrainDataset, TestDataset
from GPUUtil import set_device, move_to_device, move_model_to_device

if __name__ == "__main__":

    gpu_id = 0
    latent_dim = 128
    att_size = 128
    layer_size = [512, 256, 128, 64]
    learning_rate = 0.001
    regularization = 0.0001
    epochs = 30
    eval_frequency = 5
    train_batch_size = 256
    test_batch_size = 32
    negative_num = 4
    eval_processes_num = 4

    set_device(gpu_id)

    root = os.path.dirname(os.path.realpath(__file__)) + "/data/ML100K/"
    metapath_files = ["umgm.dat", "umum.dat", "uuum.dat", "ummm.dat"]
    feature_file_dict = {"u": "user_node_emb.dat", "m": "movie_node_emb.dat",
                         "g": "genre_node_emb.dat"}

    train_data_file = "train.dat"
    test_data_file = "test.dat"

    for i in range(len(metapath_files)):
        metapath_files[i] = root + metapath_files[i]

    for feature_type, feature_file_name in feature_file_dict.items():
        feature_file_dict[feature_type] = root + feature_file_name

    print("[Setting] processes for evaluation %d " % (eval_processes_num))
    print("[Setting] latent_dim: %d" % latent_dim)
    print("[Setting] att_size: %d" % att_size)
    print("[Setting] learning_rate: %f" % learning_rate)
    print("[Setting] regularization: %f" % regularization)
    print("[Setting] train_batch_size: %d" % train_batch_size)
    print("[Setting] test_batch_size: %d" % test_batch_size)
    print("[Setting] negative_num: %d" % negative_num)

    train_total_time = 0
    eval_total_time = 0

    loss_sequence = []

    start = time.perf_counter()

    print("Read data from " + root)
    train_dataset = TrainDataset(root + train_data_file, metapath_files, negative_num, feature_file_dict)
    test_dataset = TestDataset(train_dataset, root + test_data_file)

    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size)

    train_start = time.perf_counter()
    print("load data %.2f seconds" % (time.perf_counter() - start))

    metapath_list_attributes = []
    # metapath_list[i]: (metapath_file, path_dict, path_num, hop_num)
    for i in range(len(train_dataset.metapath_list)):
        metapath_list_attributes.append((train_dataset.metapath_list[i][2], train_dataset.metapath_list[i][3]))

    model = MCRec(latent_dim=latent_dim, att_size=att_size, feature_size=train_dataset.feature_size,
                  negative_num=negative_num,
                  user_num=train_dataset.max_user_id, item_num=train_dataset.max_item_id,
                  metapath_list_attributes=metapath_list_attributes,
                  layer_size=layer_size)

    move_model_to_device(model)

    topk = [1, 5, 10]

    pred_item_p_lists = []
    pred_item_r_lists = []
    pred_item_ndcg_lists = []
    pred_item_mrr_lists = []

    for _ in topk:
        pred_item_p_lists.append([])
        pred_item_r_lists.append([])
        pred_item_ndcg_lists.append([])
        pred_item_mrr_lists.append([])

    t1 = time.perf_counter()
    train_total_time += t1 - train_start

    results = evaluate(test_data_loader, model, test_dataset.max_test_item_num, eval_processes_num)

    for i in range(len(topk)):
        pred_item_p_lists[i].append(results[i][0])
        pred_item_r_lists[i].append(results[i][1])
        pred_item_ndcg_lists[i].append(results[i][2])
        pred_item_mrr_lists[i].append(results[i][3])

    model.train()

    t2 = time.perf_counter()
    eval_total_time += t2 - t1

    for i in range(len(topk)):
        k = topk[i]
        print('Init: Item Pred: P@%d = %.4f, R@%d = %.4f, NDCG@%d = %.4f, MRR@%d = %.4f' % (
            k, results[i][0], k, results[i][1], k, results[i][2], k, results[i][3]))

    print('Evaluation Time: %.2f seconds' % (t2 - t1))

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

    t3 = time.perf_counter()
    train_total_time += t3 - t2

    for epoch in range(epochs):

        t4 = time.perf_counter()

        cumulative_loss = 0
        total_step = 0

        for batch_data in train_data_loader:

            metapath_inputs = []
            for metapath_input in batch_data[3:]:
                metapath_inputs.append(move_to_device(metapath_input))

            output = model(move_to_device(batch_data[0]), move_to_device(batch_data[1]), metapath_inputs)
            loss = loss_fn(output, move_to_device(batch_data[2]).view(-1, 1))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            cumulative_loss += loss.item()
            total_step += 1

        cumulative_loss /= len(train_data_loader)

        loss_sequence.append(cumulative_loss)

        t5 = time.perf_counter()
        train_total_time += t5 - t4

        print("Epoch %d Steps %d Loss %f [%.2f seconds]" % (epoch, total_step, cumulative_loss, t5 - t4))

        if (epoch + 1) % eval_frequency == 0:
            t6 = time.perf_counter()

            results = evaluate(test_data_loader, model, test_dataset.max_test_item_num, eval_processes_num)

            for i in range(len(topk)):
                pred_item_p_lists[i].append(results[i][0])
                pred_item_r_lists[i].append(results[i][1])
                pred_item_ndcg_lists[i].append(results[i][2])
                pred_item_mrr_lists[i].append(results[i][3])

            t7 = time.perf_counter()
            eval_total_time += t7 - t6

            for i in range(len(topk)):
                k = topk[i]
                print('Item Pred: P@%d = %.4f, R@%d = %.4f, NDCG@%d = %.4f, MRR@%d = %.4f' % (
                    k, results[i][0], k, results[i][1], k, results[i][2], k, results[i][3]))

            print('Evaluation Time: %.2f seconds' % (t7 - t6))

    print("Total Time %.2f seconds, Train Time %.2f seconds, Evaluation Time %.2f seconds" % (
        time.perf_counter() - start, train_total_time, eval_total_time))

    for i in range(len(topk)):
        k = topk[i]
        print('Best Results for Item Pred: P@%d = %.4f, R@%d = %.4f, NDCG@%d = %.4f, MRR@%d = %.4f' % (
            k, max(pred_item_p_lists[i]), k, max(pred_item_r_lists[i]), k, max(pred_item_ndcg_lists[i]), k,
            max(pred_item_mrr_lists[i])))
