import numpy as np
import os


def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
	  print('Error: Creating directory. '+directory)


#===================================================================================
# load feature data
dataset_name = 'sift1M_base';
# Open the file in binary mode
with open('sift_base.fvecs', 'rb') as f:
    # Read the data into a NumPy feat
    feat = np.fromfile(f, dtype=np.int32) 

feat_dim = feat[0]
print(feat[0])

with open('sift_base.fvecs', 'rb') as f:
    # Read the data into a NumPy feat
    feat = np.fromfile(f, dtype=np.float32) 

graph_num_nodes = int(len(feat)/(feat_dim+1));

print(('graph load done(graph_num_node = {}, feat_dim = {}').format(graph_num_nodes, feat_dim));


createFolder('./{}'.format(dataset_name));
feat_filename = './{}/{}_b_mat.dat'.format(dataset_name, dataset_name);
feat_dim_np = np.array([feat_dim], dtype=np.uint64)
feat_2d = feat.reshape(graph_num_nodes, feat_dim+1);
#print(feat_2d[0][1])
#print(feat_2d[1][1])

# delete 'feat_dim' in column 0
feat_2d = np.delete(feat_2d, 0, axis=1);
#print(feat_2d[0][0])
#print(feat_2d[1][0])

with open(feat_filename, 'wb') as fp:
	feat_dim_np.tofile(fp, format='unit64')
	feat_2d.tofile(fp, format='float')

# pre-calculated norm
norm_filename = './{}/{}_norm.dat'.format(dataset_name, dataset_name);
ssq = np.sum(feat_2d**2, axis=1)
with open(norm_filename, 'wb') as fp:
	ssq.tofile(fp, format='float')
print(ssq[0])
print(len(ssq))

#===================================================================================
# load query data
dataset_name = 'sift1M_query';
createFolder('./{}'.format(dataset_name));
with open('sift_query.fvecs', 'rb') as f:
    # Read the data into a NumPy feat
    query = np.fromfile(f, dtype=np.float32)

query_num = int(len(query)/(feat_dim+1));
# 1d to 2d, (1, query_num*(feat_dim+1)) --> (feat_dim+1, query_num)
query_2d = query.reshape(query_num, feat_dim+1)
# delete 'feat_dim' in column 0
query_2d = np.delete(query_2d, 0, axis=1);

#print('query_2d[0][:] = {}'.format(query_2d[0][:]));
query_filename = './{}/{}_query.dat'.format(dataset_name, dataset_name);
with open(query_filename, 'wb') as fp:
	feat_dim_np.tofile(fp, format='unit64')
	query_2d.tofile(fp, format='float')

dataset_name = 'sift1M_nsg'
createFolder('./{}'.format(dataset_name));

#===================================================================================
# load graph info
with open('../sift1M.nsg', 'rb') as f:
    # Read the data into a NumPy array
    indx = np.fromfile(f, dtype=np.int32)  # Change dtype according to your data

width = indx[0]
ep_   = indx[1]
nonzero_np        = np.array([len(indx)-graph_num_nodes-2], dtype=np.uint64);
total_num_node_np = np.array([graph_num_nodes], dtype=np.uint64);
ep_np             = np.array([ep_], dtype=np.uint64);
print('width = {}, ep_ = {}\n'.format(width, ep_));
#graph_num_nodes
#nonzero

pntrb = 0;
pntre = 0;
pntrb_list = []
pntre_list = []

# delete 'width', 'ep_' in indx 
indx = indx[2:];
indx = np.array(indx, dtype=np.uint64);

k = 0;

indx_filename = './{}/{}_rows.dat'.format(dataset_name, dataset_name);
pntrb_filename = './{}/{}_pntrb.dat'.format(dataset_name, dataset_name);
pntre_filename = './{}/{}_pntre.dat'.format(dataset_name, dataset_name);

with open(indx_filename, 'wb') as fp:
	total_num_node_np.tofile(fp, format='uint64');
	nonzero_np.tofile(fp, format='uint64');
	ep_np.tofile(fp, format='uint64');

	# indx[k]       indx[k+1:k+num_indx+1]
	# neighbor_num, neighbor_list         
	for jj in range(0, graph_num_nodes):
		if jj%100000==0:
			print('{}/{}-th iteration for indx'.format(jj,graph_num_nodes))	
		pntrb = pntre;
		num_indx  = int(indx[k]);
		pntre = pntre + num_indx;
		indx_slice = indx[k+1:k+num_indx+1]
		if jj == 0:
			print('k = {}, indx[{}] = {}, indx = {}'.format(k, k, indx[k], indx[k+1:k+num_indx+1]))
		indx_slice.tofile(fp, format='uint64')
		pntrb_list.append(pntrb);
		pntre_list.append(pntre);
		k = k + num_indx+1

# pntrb
with open(pntrb_filename, 'wb') as fp:
	pntrb_list_np = np.array(pntrb_list, dtype=np.uint64)
	pntrb_list_np.tofile(fp, format='uint64')

# pntre
with open(pntre_filename, 'wb') as fp:
	pntre_list_np = np.array(pntre_list, dtype=np.uint64)
	pntre_list_np.tofile(fp, format='uint64')
