import numpy as np


sm_bound = 5.0;
sm_table_size = 2048;
sm_resolution = sm_table_size/(2.0 * sm_bound);
sm_table = []

for idx in range(0,sm_table_size):
	x = 2.0 * sm_bound * idx / sm_table_size - sm_bound;
	sm_value =  1.0 / (1 + np.exp(-x));
	sm_table.append(sm_value);
	print(sm_value)

sm_table_np = np.array(sm_table, dtype=np.float32);
with open('sm_table.dat', 'wb') as fp:
    sm_table_np.tofile(fp, format='float')


