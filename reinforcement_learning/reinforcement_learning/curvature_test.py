import csv
import numpy as np
import trajectory_planning_helpers as tph

KAPPA_STEP = 0.5

file_path = '/home/mina/sim_ws/src/reinforcement_learning/waypoints/test_optimal.csv'
data = np.loadtxt(file_path, delimiter=',')

path = data[ : , :2]
diff = path[ :-1] - path[1: ]

psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(path = path,
                       el_lengths = np.linalg.norm(diff, axis=1),
                       stepsize_curv_preview = 1.0,
                       stepsize_curv_review = 1.0,
                       is_closed = False)
psi_ = psi + (np.pi/2) + (2*np.pi)
psi_ = psi_ % (2*np.pi)

kappa = []
stacked_path = np.vstack((path, path, path))
stacked_diff = np.linalg.norm(stacked_path[ :-1] - stacked_path[1: ], axis=1)
idx = data.shape[0]
for i in range(data.shape[0]):
    prev_idx = idx
    cnt = 0
    # Find -(KAPPA_STEP) index
    while cnt <= KAPPA_STEP:
        prev_idx -= 1
        cnt += stacked_diff[prev_idx-1]
    prev2curr_vec = stacked_path[idx] - stacked_path[prev_idx]

    # Find +(KAPPA_STEP) index
    next_idx = idx + 1
    cnt = stacked_diff[next_idx-1]
    while cnt <= KAPPA_STEP:
        next_idx += 1
        cnt += stacked_diff[next_idx-1]
    curr2next_vec = stacked_path[next_idx] - stacked_path[idx]

    inner_product = np.dot(prev2curr_vec, curr2next_vec)
    cos_theta = inner_product / (np.linalg.norm(prev2curr_vec)*np.linalg.norm(curr2next_vec))
    if np.isnan(np.arccos(cos_theta)):
        kappa.append(0.0)
    else:
        kappa.append(np.arccos(cos_theta))
    idx += 1


with open('curvature.csv', mode='w') as file:
    writer = csv.writer(file)
    for i in range(len(kappa)):
        writer.writerow([data[i,0], data[i,1], psi_[i], kappa[i], data[i,3]])

