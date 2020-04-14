import multiprocessing
import heapq
import math
import numpy as np
import torch
from GPUUtil import move_to_device

_predictions = None
_batch_data = None


def evaluate(test_data_loader, model, max_test_item_num, eval_processes_num):

	global _predictions
	global _batch_data

	# topK = 1, 5, 10
	pred_item_ps = np.array([0.0] * 3)
	pred_item_rs = np.array([0.0] * 3)
	pred_item_ndcgs = np.array([0.0] * 3)
	pred_item_mrrs = np.array([0.0] * 3)

	model.eval()

	batch_index = 0
	# batch_data[idx] = [real_test_item_size, positive_item_indices, test_item_ids, user_input, item_input, metapath_inputs]
	for _batch_data in test_data_loader:

		metapath_inputs = []
		for metapath_input in _batch_data[5:]:
			metapath_inputs.append(move_to_device(metapath_input))

		with torch.no_grad():
			# predictions shape: batch_size * max_test_item_num

			_predictions = model(move_to_device(_batch_data[3]), move_to_device(_batch_data[4]), metapath_inputs)
			_predictions = _predictions.view(-1, max_test_item_num).cpu()

		batch_size = _predictions.shape[0]

		pool = multiprocessing.Pool(processes=eval_processes_num)
		batch_results = pool.map(eval_one_batch, range(batch_size))

		batch_pred_item_ps = np.array([0.0] * 3)
		batch_pred_item_rs = np.array([0.0] * 3)
		batch_pred_item_ndcgs = np.array([0.0] * 3)
		batch_pred_item_mrrs = np.array([0.0] * 3)

		for result in batch_results:
			batch_pred_item_ps += np.array(result[0])
			batch_pred_item_rs += np.array(result[1])
			batch_pred_item_ndcgs += np.array(result[2])
			batch_pred_item_mrrs += np.array(result[3])

		pred_item_ps += batch_pred_item_ps / len(batch_results)
		pred_item_rs += batch_pred_item_rs / len(batch_results)
		pred_item_ndcgs += batch_pred_item_ndcgs / len(batch_results)
		pred_item_mrrs += batch_pred_item_mrrs / len(batch_results)

		pool.close()
		pool.join()

		batch_index += 1

	pred_item_ps = pred_item_ps / (batch_index * 1.0)
	pred_item_rs = pred_item_rs / (batch_index * 1.0)
	pred_item_ndcgs = pred_item_ndcgs / (batch_index * 1.0)
	pred_item_mrrs = pred_item_mrrs / (batch_index * 1.0)

	# [[p_1, r_1, ndcg_1, mrr_1], [p_5, r_5, ndcg_5, mrr_5], [p_10, r_10, ndcg_10, mrr_10]]
	results = []

	for i in range(3):
		results.append((pred_item_ps[i], pred_item_rs[i], pred_item_ndcgs[i], pred_item_mrrs[i]))

	return results


def eval_one_batch(idx):

	map_item_score = {}
	# batch_data[idx] = [real_test_item_size, positive_item_indices, test_item_ids, user_input, item_input, metapath_inputs]
	real_test_item_size = _batch_data[0][idx].numpy()[0]

	test_item_ids = _batch_data[2][idx].numpy()[0:real_test_item_size]
	prediction = _predictions[idx]

	for item_idx in range(len(test_item_ids)):
		item_id = test_item_ids[item_idx]
		map_item_score[item_id] = prediction[item_idx]

	# Evaluate top rank list
	ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)

	positive_item_index = _batch_data[1][idx].numpy()[0]
	positive_item_ids = test_item_ids[positive_item_index:real_test_item_size]

	p_list = getP(ranklist, positive_item_ids)
	r_list = getR(ranklist, positive_item_ids)
	ndcg_list = getNDCG(ranklist, positive_item_ids)
	mrr_list = getMRR(ranklist, positive_item_ids)

	return p_list, r_list, ndcg_list, mrr_list


def getP(ranklist, gtItems):
	p_1, p_5, p_10 = 0, 0, 0

	counter = 0
	for item in ranklist:
		if item in gtItems:
			if counter == 0:
				p_1 += 1

			if counter < 5:
				p_5 += 1

			p_10 += 1

		counter += 1

	return (p_1 * 1.0, p_5 / 5.0, p_10 / 10.0)


def getR(ranklist, gtItems):
	r_1, r_5, r_10 = 0, 0, 0

	counter = 0
	for item in ranklist:
		if item in gtItems:
			if counter == 0:
				r_1 += 1

			if counter < 5:
				r_5 += 1

			r_10 += 1

		counter += 1

	return (r_1 * 1.0 / len(gtItems), r_5 * 1.0 / len(gtItems), r_10 * 1.0 / len(gtItems))


def getNDCG(ranklist, gtItems):
	dcg_1, dcg_5, dcg_10 = getDCG(ranklist, gtItems)
	idcg_1, idcg_5, idcg_10 = getIDCG(ranklist, gtItems)

	if idcg_1 == 0:
		ndcg_1 = 0
	else:
		ndcg_1 = dcg_1 / idcg_1

	if idcg_5 == 0:
		ndcg_5 = 0
	else:
		ndcg_5 = dcg_5 / idcg_5

	if idcg_10 == 0:
		ndcg_10 = 0
	else:
		ndcg_10 = dcg_10 / idcg_10

	return (ndcg_1, ndcg_5, ndcg_10)


def getDCG(ranklist, gtItems):
	dcg_1, dcg_5, dcg_10 = 0.0, 0.0, 0.0

	for i in range(len(ranklist)):
		item = ranklist[i]
		if item in gtItems:
			if i == 0:
				dcg_1 += 1.0 / math.log(i + 2)
			if i < 5:
				dcg_5 += 1.0 / math.log(i + 2)
			dcg_10 += 1.0 / math.log(i + 2)

	return (dcg_1, dcg_5, dcg_10)


def getIDCG(ranklist, gtItems):
	idcg_1, idcg_5, idcg_10 = 0.0, 0.0, 0.0
	i = 0
	counter = 0
	for item in ranklist:
		if item in gtItems:
			if counter == 0:
				idcg_1 += 1.0 / math.log(i + 2)
			if counter < 5:
				idcg_5 += 1.0 / math.log(i + 2)
			idcg_10 += 1.0 / math.log(i + 2)
			i += 1

		counter+=1
	return (idcg_1, idcg_5, idcg_10)


def getMRR(ranklist, gtItems):
	mrr_1, mrr_5, mrr_10 = 0.0, 0.0, 0.0
	rank = 1
	for item in ranklist:
		if item in gtItems:
			if rank == 1:
				mrr_1 += 1.0 / rank
			if rank <= 5:
				mrr_5 += 1.0 / rank
			mrr_10 += 1.0 / rank
		rank += 1

	return (mrr_1 / len(gtItems), mrr_5 / len(gtItems), mrr_10 / len(gtItems))
