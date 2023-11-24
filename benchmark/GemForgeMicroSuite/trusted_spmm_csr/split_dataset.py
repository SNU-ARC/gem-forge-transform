import numpy as np

# configuratin
dim_vector_hvd      = 128;

# fixed configuration from cs4.mtx
filename            = "./harvard.mtx"
prefix              = "harvard_"

pairs = {} 
count_nnz = 0;
with open(filename, 'r') as fp:
	for line in fp:
		if line[0] == "%":
			print("%%%%")
		else:
			line = line.split('\n');
			item_list = line[0].split(' ')
			if len(item_list) == 3 :
				M = int(item_list[0])
				N = item_list[1]
				NNZ = item_list[2]
			else:
				row = int(item_list[0])-1;
				col = int(item_list[1])-1;
				res = pairs.get(row, 'NONE')
				if res  == 'NONE':
					res = []
					res.append(col)
					pairs[row] = res
					count_nnz += 1
				else:					
					if col not in res:
						res.append(col)
						pairs[row] = res
						count_nnz += 1
				# sym
				res = pairs.get(col, 'NONE')
				if res  == 'NONE':
					res = []
					res.append(row)
					pairs[col] = res
					count_nnz += 1
				else:					
					if row not in res:
						res.append(row)
						pairs[col] = res
						count_nnz += 1


#print(len(pairs.keys()))
sorted_pairs = dict(sorted(pairs.items()))
sorted_pairs_items = list(sorted_pairs.items());

rows = []
cols = []

pntrb = []
pntre = []
c_pntrb = 0 
c_pntre = 0 

p_pntrb = 0 
p_pntre = 0 
for key, value in sorted_pairs_items:
    #print("key = " + str(key) + ": value = " + str(value))
	# initial
	if p_pntre == 0:		
		c_pntrb = 0
	else:
		c_pntrb = p_pntre

	c_pntre = c_pntrb + len(value)

	for v in value:
		rows.append(v)
		cols.append(key)
	pntrb.append(c_pntrb)
	pntre.append(c_pntre)
	
	p_pntrb = c_pntrb
	p_pntre = c_pntre

		
rows_np = np.array(rows, dtype=np.uint64)
cols_np = np.array(cols, dtype=np.uint64)
	



#for ii in cols:	
#	print(ii)
print('count_nnz={}\n'.format(count_nnz))
print(max(rows))
print(max(cols))
print(len(cols))
print(len(pntrb))
print(len(pntre))

#for ii in [451, 452, 453,454,455,456,457]:
#	print(str(pntrb[ii]) + ":" + str(pntre[ii]))
#print(len(sorted_pairs.values()))
print(len(rows_np))
rows_np = np.insert(rows_np, 0, count_nnz);
rows_np = np.insert(rows_np, 0, M);
print(len(rows_np))
print(rows_np[0])

with open(prefix+'rows.dat', 'wb') as fp:
    rows_np.tofile(fp, format='uint64')

with open(prefix+'cols.dat', 'wb') as fp:
    cols_np.tofile(fp, format='uint64')

#print(len(cols_np_unique))
#print(len(pntrb))
#print(len(pntre))

# write pntrb/pntre
pntrb_np = np.array(pntrb, dtype=np.uint64)
pntre_np = np.array(pntre, dtype=np.uint64)


#for ii in range(0,len(pntrb_np)):	
#	print(pntre_np[ii] - pntrb_np[ii])

with open(prefix+'pntrb.dat', 'wb') as fp:
    pntrb_np.tofile(fp, format='uint64')

with open(prefix+'pntre.dat', 'wb') as fp:
    pntre_np.tofile(fp, format='uint64')

# write rand_val
## random generation
rand_val = np.random.rand(count_nnz).astype('f');
with open(prefix+'val.dat', 'wb') as fp:
    rand_val.tofile(fp, format='float')

## random generation
rand_b_mat = np.random.rand(M*dim_vector_hvd).astype('f');
print(len(rand_b_mat));
print(type(rand_b_mat));
#rand_b_mat = np.insert(rand_b_mat, 0, float(dim_vector_hvd));
print(len(rand_b_mat));
print('rand_b_mat[0] = {}'.format(rand_b_mat[0]))
with open(prefix+'b_mat.dat', 'wb') as fp:
	dim_vec = np.array([dim_vector_hvd], dtype=np.uint64);
	dim_vec.tofile(fp, format='uint64')
	rand_b_mat.tofile(fp, format='float')

## random generation
rand_a_mat = np.random.rand(M*dim_vector_hvd).astype('f');
#print(rand_a_mat[0])
with open(prefix+'a_mat.dat', 'wb') as fp:
    rand_a_mat.tofile(fp, format='float')
