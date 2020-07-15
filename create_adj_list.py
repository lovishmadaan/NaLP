# import tensorflow as tf
import numpy as np
import pickle
import time
import sys
import argparse

ISOTIMEFORMAT='%Y-%m-%d %X'
parser = argparse.ArgumentParser()       
parser.add_argument("--data_dir", dest='data_dir', type=str, help="The data dir.", default='./data')
parser.add_argument("--sub_dir", dest='sub_dir', type=str, help="The sub data dir.", default="WikiPeople")
parser.add_argument("--dataset_name", dest="dataset_name", type=str, help="The name of the dataset.", default="WikiPeople")
parser.add_argument("--bin_postfix", dest="bin_postfix", type=str, help="The new_postfix for the output bin file.", default="")
parser.add_argument("--if_permutate",dest="if_permutate", type=bool, help="If permutate for test filter.", default=False)
args = parser.parse_args() 

print("\nParameters:")
print(args)

def build_adjacency_data(folder='data/', dataset_name='WikiPeople'):
	f=open(folder + dataset_name + args.bin_postfix + ".bin", 'rb')
	data_info=pickle.load(f)
	train_facts = data_info["train_facts"]
	valid_facts = data_info["valid_facts"]
	test_facts = data_info['test_facts']
	values_indexes = data_info['values_indexes']
	roles_indexes = data_info['roles_indexes']
	role_val = data_info['role_val']
	value_inverse_indexes={}
	for k in values_indexes:
		if (values_indexes[k] not in value_inverse_indexes):
			value_inverse_indexes[values_indexes[k]]=k
	with open(folder + 'n-ary_train.json') as f:
		lines = f.readlines()
	val_corres_dict={}
	for k in values_indexes:
		if values_indexes[k] not in val_corres_dict:
			val_corres_dict[values_indexes[k]]=[]
	# exit()
	# print(value_inverse_indexes[4844])
	# print(val_corres_dict[4844])
	for _, line in enumerate(lines):
		value_arr=[]
		n_dict = eval(line)
		for k in n_dict:
			if k == 'N':
				continue
			k_ind = roles_indexes[k]
			# if k_ind not in role_val:
			#     role_val[k_ind] = []
			v = n_dict[k]
			if type(v) == str:
				v_ind = values_indexes[v]
				if v_ind not in val_corres_dict:
					val_corres_dict[v_ind]=[]
				for j in value_arr:
					val_corres_dict[v_ind].append(j)
					val_corres_dict[j].append(v_ind)
				value_arr.append(v_ind)
			else:  # Multiple values
				for val in v:
					val_ind = values_indexes[val]
					if v_ind not in val_corres_dict:
						val_corres_dict[v_ind]=[]
					for j in value_arr:
						val_corres_dict[v_ind].append(j)
						val_corres_dict[j].append(v_ind)
					value_arr.append(v_ind)
		# print(line)
		# for k in val_corres_dict:
		# 	if (len(val_corres_dict[k])>0):
		# 		print(val_corres_dict[k])
		# exit()
	# print(val_corres_dict[4844])
	max_degree=25
	val_corres_sampled_dict={}
	average_len=[]
	for k in val_corres_dict:
		# average_len.append(len(val_corres_dict[k]))
		# print(k)
		# print(len(val_corres_dict[k]))
		if len(val_corres_dict[k]) == 0:
			val_corres_sampled_dict[k] =np.array([k]*max_degree)
		elif len(val_corres_dict[k]) >= max_degree:
			val_corres_sampled_dict[k] = np.random.choice(val_corres_dict[k], max_degree, replace=False)
		elif len(val_corres_dict[k]) < max_degree:
			val_corres_sampled_dict[k] = np.random.choice(val_corres_dict[k], max_degree, replace=True)
		# print(val_corres_sampled_dict[k])
	# print(np.mean(np.array(average_len)))
	# exit()
	# print(val_corres_sampled_dict[4844])
	with open(folder + dataset_name + "_adj_list.bin", 'wb') as f:
		pickle.dump(val_corres_sampled_dict, f)
	return


if __name__ == '__main__':
	print(time.strftime(ISOTIMEFORMAT, time.localtime()))
	afolder = args.data_dir + '/'
	if args.sub_dir != '':
		afolder = args.data_dir + '/' + args.sub_dir + '/'
	build_adjacency_data(folder=afolder, dataset_name=args.dataset_name)
	print(time.strftime(ISOTIMEFORMAT, time.localtime()))