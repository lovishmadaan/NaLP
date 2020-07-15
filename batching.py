import tensorflow as tf
import numpy as np

def replace_val(n_values, last_idx, role_val, arity, new_facts_indexes, new_facts_values, whole_train_facts):
	"""
	Replace values randomly to get negative samples
	"""
	rmd_dict = role_val
	for cur_idx in range(last_idx):
		role_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
		tmp_role = new_facts_indexes[last_idx + cur_idx, role_ind]
		tmp_len = len(rmd_dict[tmp_role])
		rdm_w = np.random.randint(0, tmp_len)  # [low,high)

		# Sample a random value
		times = 1
		tmp_array = new_facts_indexes[last_idx + cur_idx]
		tmp_array[role_ind+1] = rmd_dict[tmp_role][rdm_w]
		while (tuple(tmp_array) in whole_train_facts):
			if (tmp_len == 1) or (times > 2*tmp_len) or (times > 100):
				tmp_array[role_ind+1] = np.random.randint(0, n_values)
			else:
				rdm_w = np.random.randint(0, tmp_len)
				tmp_array[role_ind+1] = rmd_dict[tmp_role][rdm_w]
			times = times + 1
		new_facts_indexes[last_idx + cur_idx, role_ind+1] = tmp_array[role_ind+1]
		new_facts_values[last_idx + cur_idx] = [-1]

def replace_role(n_roles, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts):
	"""
	Replace roles randomly to get negative samples
	"""
	rdm_ws = np.random.randint(0, n_roles, last_idx)        
	for cur_idx in range(last_idx):
		role_ind = (np.random.randint(np.iinfo(np.int32).max) % arity) * 2
		# Sample a random role
		tmp_array = new_facts_indexes[last_idx + cur_idx]
		tmp_array[role_ind] = rdm_ws[cur_idx]
		while (tuple(tmp_array) in whole_train_facts):
			tmp_array[role_ind] = np.random.randint(0, n_roles)
		new_facts_indexes[last_idx + cur_idx, role_ind] = tmp_array[role_ind]
		new_facts_values[last_idx + cur_idx] = [-1]

def Batch_Loader(train_facts, values_indexes, roles_indexes, role_val, batch_size, arity, whole_train_facts):
	indexes = np.array(list(train_facts.keys())).astype(np.int32)
	values = np.array(list(train_facts.values())).astype(np.float32)
	new_facts_indexes = np.empty((batch_size*2, 2*arity)).astype(np.int32)
	new_facts_values = np.empty((batch_size*2, 1)).astype(np.float32)

	idxs = np.random.randint(0, len(values), batch_size)
	new_facts_indexes[:batch_size, :] = indexes[idxs, :]
	new_facts_values[:batch_size] = values[idxs, :]
	last_idx = batch_size

	indexes_values = {}
	for tmpkey in values_indexes:
		indexes_values[values_indexes[tmpkey]] = tmpkey
	indexes_roles = {}
	for tmpkey in roles_indexes:
		indexes_roles[roles_indexes[tmpkey]] = tmpkey
	# Copy everyting in advance
	new_facts_indexes[last_idx:(last_idx*2), :] = np.tile(
		new_facts_indexes[:last_idx, :], (1, 1))
	new_facts_values[last_idx:(last_idx*2)] = np.tile(
		new_facts_values[:last_idx], (1, 1))
	n_values = len(values_indexes)
	n_roles = len(roles_indexes)
	val_role = np.random.randint(np.iinfo(np.int32).max) % (n_values+n_roles)
	if val_role < n_values:  # 0~(n_values-1)
		replace_val(n_values, last_idx, role_val, arity, new_facts_indexes, new_facts_values, whole_train_facts)
	else:
		replace_role(n_roles, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts)
	last_idx += batch_size

	return new_facts_indexes[:last_idx, :], new_facts_values[:last_idx]

def get_neighbours(facts, sample_neighbour):
	neighbour_total_array=[]
	for k in facts:
		neighbour_array=[]
		for t in range(int(len(k)/2)):
			# print(t)
			# print(k[2*t+1])
			neigh=sample_neighbour[k[2*t+1]]
			neighbour_array.append(neigh)
		neighbour_total_array.append(neighbour_array)
	return np.array(neighbour_total_array)

def Batch_Loader_GNN(train_facts, values_indexes, roles_indexes, role_val, batch_size, arity, whole_train_facts,sample_neighbour):
	indexes = np.array(list(train_facts.keys())).astype(np.int32)
	values = np.array(list(train_facts.values())).astype(np.float32)
	new_facts_indexes = np.empty((batch_size*2, 2*arity)).astype(np.int32)
	new_facts_values = np.empty((batch_size*2, 1)).astype(np.float32)

	idxs = np.random.randint(0, len(values), batch_size)
	new_facts_indexes[:batch_size, :] = indexes[idxs, :]
	new_facts_values[:batch_size] = values[idxs, :]

	last_idx = batch_size

	indexes_values = {}
	for tmpkey in values_indexes:
		indexes_values[values_indexes[tmpkey]] = tmpkey
	indexes_roles = {}
	for tmpkey in roles_indexes:
		indexes_roles[roles_indexes[tmpkey]] = tmpkey
	# Copy everyting in advance
	new_facts_indexes[last_idx:(last_idx*2), :] = np.tile(
		new_facts_indexes[:last_idx, :], (1, 1))
	new_facts_values[last_idx:(last_idx*2)] = np.tile(
		new_facts_values[:last_idx], (1, 1))
	n_values = len(values_indexes)
	n_roles = len(roles_indexes)
	val_role = np.random.randint(np.iinfo(np.int32).max) % (n_values+n_roles)
	if val_role < n_values:  # 0~(n_values-1)
		replace_val(n_values, last_idx, role_val, arity, new_facts_indexes, new_facts_values, whole_train_facts)
	else:
		replace_role(n_roles, last_idx, arity, new_facts_indexes, new_facts_values, whole_train_facts)
	last_idx += batch_size
	neighbour_x=get_neighbours(new_facts_indexes[:last_idx, :],sample_neighbour)

	return new_facts_indexes[:last_idx, :], new_facts_values[:last_idx],neighbour_x

