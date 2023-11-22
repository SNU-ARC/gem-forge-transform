#from ogb.graphproppred import PygGraphPropPredDataset
#from torch_geometric.data import DataLoader
#
## Download and process data at './dataset/ogbg_molhiv/'
#dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')


from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import time
import os


num_split = 4;

start_time = time.time()

dataset_name = 'ogbn-papers100M'#  #  #'ogbn-arxiv' 
dataset = DglNodePropPredDataset(name = dataset_name, root = 'dataset/')

#split_idx = dataset.get_idx_split()
#train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
print(graph)
feat = graph.ndata['feat'] # floatTensor(float32) type
adj_tensors = graph.adj_tensors('csr');
print(adj_tensors)
print(len(adj_tensors));
print(len(adj_tensors[0]));
print(len(adj_tensors[1]));
print(len(adj_tensors[2]));

indptr = adj_tensors[0];
indptr = indptr.tolist();

indx = adj_tensors[1];
#indx = indx.tolist();
indx = np.array(indx, dtype=np.uint64)
#print(graph)
#print(dir(graph))

graph_num_nodes = graph.num_nodes()
graph_num_edges = graph.num_edges()
feat_dim        = feat[0].size(dim=0)


print(('graph load done(graph_num_node = {}, graph_num_edges = {}, feat_dim = {}').format(graph_num_nodes, graph_num_edges, feat_dim));

sub_graph_num_nodes, remainder = divmod(graph_num_nodes, num_split);
print('sub_graph_num_nodes = {}'.format(sub_graph_num_nodes));
if(remainder !=0):
	sub_graph_num_nodes += 1;
	dummy_sub_graph_idx = list(range(remainder, num_split));
	print('dummy_sub_graph_idx = {}'.format(dummy_sub_graph_idx));

start_node_idx = sub_graph_num_nodes*0
end_node_idx   = sub_graph_num_nodes*1

indx_list_0  =  []
num_idx_list = np.array(range(0, len(indx)), dtype=np.uint64)

row_list_size  = []
pntrb_list_size= []
pntre_list_size= []

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
	  print('Error: Creating directory. '+directory)


for ii in range(0, num_split):
	add_dummy_node = 0;
	if num_split == 1:
		createFolder('./{}'.format(dataset_name));
	else:
		createFolder('./{}_{}'.format(dataset_name,ii));

	if num_split == 1:
		feat_filename = './{}/{}_b_mat.dat'.format(dataset_name, dataset_name);
	else:
		feat_filename = './{}_{}/{}_{}_b_mat.dat'.format(dataset_name, ii, dataset_name, ii);
		
	print('start writing feat on '+feat_filename)

	start_node_idx = sub_graph_num_nodes*ii;
	print('{} vs {}'.format(sub_graph_num_nodes*(ii+1), graph_num_nodes))
	if(sub_graph_num_nodes*(ii+1) >= graph_num_nodes):
		end_node_idx = graph_num_nodes;
		add_dummy_node = 1;
	else:
		end_node_idx = sub_graph_num_nodes*(ii+1)

	with open(feat_filename, 'wb') as fp:
#print('start_node_idx = {}, end_node_idx = {}'.format(start_node_idx, end_node_idx))
		# add dimension
		partial_feat_np = np.array([feat_dim], dtype=np.uint64)
		partial_feat_np.tofile(fp, format='uint64')
		for jj in range(start_node_idx, end_node_idx):
			if jj%100000==0:
				print('split-{}::{}/{}-th iteration for feature'.format(ii, jj,graph_num_nodes))
			partial_feat = feat[ii][:];
			partial_feat_np = np.array(partial_feat, dtype=np.float32)
			partial_feat_np.tofile(fp, format='float')

		if(add_dummy_node):
			for jj in range(graph_num_nodes, sub_graph_num_nodes*(ii+1)):
#print('{}-th iteration, from {} to {}'.format(jj, graph_num_nodes+1, sub_graph_num_nodes*(ii+1)));
				partial_feat_np = np.zeros(shape=(feat_dim,),dtype=np.float32)
				partial_feat_np.tofile(fp, format='float')
	
	if num_split == 1:
		indx_filename = './{}/{}_rows.dat'.format(dataset_name, dataset_name);
		pntrb_filename = './{}/{}_pntrb.dat'.format(dataset_name, dataset_name);
		pntre_filename = './{}/{}_pntre.dat'.format(dataset_name, dataset_name);
		val_filename   = './{}/{}_val.dat'.format(dataset_name, dataset_name);
	else:
		indx_filename  = './{}_{}/{}_{}_rows.dat'.format (dataset_name, ii, dataset_name, ii);
		pntrb_filename = './{}_{}/{}_{}_pntrb.dat'.format(dataset_name, ii, dataset_name, ii);
		pntre_filename = './{}_{}/{}_{}_pntre.dat'.format(dataset_name, ii, dataset_name, ii);
		val_filename   = './{}_{}/{}_{}_val.dat'.format  (dataset_name, ii, dataset_name, ii);
		
	pntrb = 0;
	pntre = 0;
	pntrb_list = []
	pntre_list = []
	total_find_res = np.logical_and(indx >= start_node_idx, indx <end_node_idx);
	filtered_idx_list     = indx[total_find_res] - start_node_idx;
#print(filtered_num_idx_list)
#	arr = [n for n in filtered_num_idx_list if (n >= indptr[0]) & (n < indptr[0+1])]

	with open(indx_filename, 'wb') as fp:	
		for jj in range(0, graph_num_nodes):
			if jj%100000==0:
				print('split-{}::{}/{}-th iteration for indx'.format(ii,jj,graph_num_nodes))
			pntrb = pntre;
			node_indx = indx[indptr[jj]:indptr[jj+1]]
			find_res  = total_find_res[indptr[jj]:indptr[jj+1]]
			pntre += np.count_nonzero(find_res);
			pntrb_list.append(pntrb);
			pntre_list.append(pntre);

			if jj == 1011:
				a = filtered_idx_list[pntrb_list[1011]:pntre_list[1011]]+start_node_idx
				a = a.tolist();
				indx_list_0.append(a)

		nonzero = len(filtered_idx_list);
		filtered_idx_list = np.insert(filtered_idx_list, 0, nonzero);
		filtered_idx_list = np.insert(filtered_idx_list, 0, graph_num_nodes);
		filtered_idx_list.tofile(fp, format='uint64');


#	print('indx_list = {}'.format(filtered_idx_list[pntrb_list[1011]:pntre_list[1011]]+start_node_idx))

	# pntrb
	with open(pntrb_filename, 'wb') as fp:
		pntrb_list_np = np.array(pntrb_list, dtype=np.uint64)
		pntrb_list_np.tofile(fp, format='uint64')

	# pntre
	with open(pntre_filename, 'wb') as fp:
		pntre_list_np = np.array(pntre_list, dtype=np.uint64)
		pntre_list_np.tofile(fp, format='uint64')

	# val 
	rand_val = np.random.rand(nonzero).astype('f');
	with open(val_filename, 'wb') as fp:
	    rand_val.tofile(fp, format='float')


#print('pntrb = {}'.format(pntrb));
#print('pntre = {}'.format(pntre));
	print('{}-th iteration, len(indx_list ) = {}'.format(ii, nonzero))
	print('{}-th iteration, len(pntrb_list) = {} min = {}, max = {}'.format(ii, len(pntrb_list), np.min(pntrb_list), np.max(pntrb_list)))
	print('{}-th iteration, len(pntre_list) = {} min = {}, max = {}'.format(ii, len(pntre_list), np.min(pntre_list), np.max(pntre_list)))

	row_list_size.append(nonzero)
	pntrb_list_size.append(len(pntrb_list))
	pntre_list_size.append(len(pntre_list))


print('original indx[0] = {}, split indx[0] = {}'.format(indx[indptr[1011]:indptr[1011+1]], indx_list_0))
print('indptr[1011] = {}, indptr[1011+1]] = {}'.format(indptr[1011], indptr[1011+1]))
print("---{}s seconds---".format(time.time()-start_time))

graphinfo_filename = dataset_name + '_info.txt';
with open(graphinfo_filename, 'w') as fp:
	fp.writelines(('total_graph_num_nodes = {}\n').format(graph_num_nodes));
	fp.writelines(('total_graph_num_edges = {}\n').format(graph_num_edges));
	fp.writelines(('feat_dim = {}\n').format(feat_dim));
	fp.writelines(('sub_graph_num_nodes = {}\n').format(sub_graph_num_nodes));
	fp.writelines(('sub_graph_num_edges = {}\n').format(graph_num_edges));

	fp.writelines('//===============================\n');
	fp.writelines(('static const uint64_t feat_dim              = {}\n').format(feat_dim));
	fp.writelines(('static const uint64_t total_graph_num_node  = {}\n').format(graph_num_nodes));
	fp.writelines(('static const uint64_t total_graph_num_edges = {}\n').format(graph_num_edges));
	fp.writelines(('static const uint64_t sub_graph_num_nodes   = {}\n').format(sub_graph_num_nodes));
	for ii in range(0,num_split):
		fp.writelines(('//============{}-data=============\n').format(ii));
		fp.writelines(('static const uint64_t row_list_size_{}   = {};\n').format(ii, row_list_size[ii]));
		fp.writelines(('static const uint64_t pntrb_list_size_{} = {};\n').format(ii, pntrb_list_size[ii]));
		fp.writelines(('static const uint64_t pntre_list_size_{} = {};\n').format(ii, pntre_list_size[ii]));
		fp.writelines(('static const uint64_t data_list_size_{}  = {};\n').format(ii, sub_graph_num_nodes));
		fp.writelines('//===============================\n');
		
	for ii in range(0,num_split):
		fp.writelines(('INDEXTYPE* indx_{} = (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  row_list_size_{}    * sizeof(INDEXTYPE));\n').format(ii, ii))
		fp.writelines(('INDEXTYPE* pntrb_{}= (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  pntrb_list_size_{}  * sizeof(INDEXTYPE));\n').format(ii, ii))
		fp.writelines(('INDEXTYPE* pntre_{}= (INDEXTYPE*) aligned_alloc(CACHE_LINE_SIZE,  pntrb_list_size_{}  * sizeof(INDEXTYPE));\n').format(ii, ii))
		fp.writelines(('VALUETYPE* val_{}  = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  data_list_size_{}   * sizeof(VALUETYPE));\n').format(ii, ii))
		fp.writelines(('VALUETYPE* b_{}    = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  data_list_size_{}   * feat_dim * sizeof(VALUETYPE));\n').format(ii,ii))
		fp.writelines(('VALUETYPE* c_{}    = (VALUETYPE*) aligned_alloc(CACHE_LINE_SIZE,  total_graph_num_node* feat_dim * sizeof(VALUETYPE));\n\n').format(ii,ii))

	fp.writelines(('INDEXTYPE ldc = feat_dim;\n'))
	fp.writelines(('INDEXTYPE ldb = feat_dim;\n'))
	fp.writelines(('INDEXTYPE m   = sub_graph_num_node;\n'))
	fp.writelines(('INDEXTYPE k   = feat_dim;\n\n'))

	fp.writelines('===============================\n');
	fp.writelines(('run_time = {}s seconds---\n').format(time.time()-start_time))



