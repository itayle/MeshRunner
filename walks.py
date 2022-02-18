import numpy as np




def regularWalk(vertices,mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump
  return seq, jumps

def regularWalkWithSkips(vertices,mesh_extra, f0, seq_len):
  Skips_param  = 2
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, (seq_len + 1)*2):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    if i % Skips_param ==0:
      seq[i] = to_add
      jumps[i] = jump
  return seq, jumps



def RandomJumps(vertices,mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    seq[i] = np.random.randint(n_vertices)
    jumps[i] = jump
  return seq, jumps

def regularWalkWithoutJumps(vertices,mesh_extra, f0, seq_len):
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = [True]
  backward_steps = 1
  jump_prob = 0 #### 1 / 100
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = 0 #np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = False
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
    jumps[i] = jump
  return seq, jumps


def WalkInOrderlyFashion(vertices,mesh_extra, f0, seq_len):
  ## Sort Vertices : 
  lst_of = [] 
  for i in range(vertices.shape[0]):
	  lst_of.append([])
	  for j in range(vertices.shape[1]):
		  lst_of[i].append(vertices[i][j])
    # 2. lst to lst with inxs 
  lst_of_lsts_w_i = [[l,i] for i,l in enumerate(lst_of)]
    # 3. sort according to value
  lst_of_lsts_w_i.sort(key = lambda t:(t[0][0],t[0][1],t[0][2])) 
  just_idxs = [i[1] for i in lst_of_lsts_w_i]
  
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  f_index = just_idxs.index(f0)
  seq[0] = f0
  jumps[0] = True
  for i in range(1, seq_len + 1):
    f_index +=1 
    f_index =  f_index % n_vertices
    seq[i] = just_idxs[f_index]
    jumps[i] = False
  return seq, jumps
get_seq_random_walk_random_global_jumps = regularWalk
#### Walks:
#1. regularWalk
#2. regularWalkWithSkips
#3. RandomJumps
#4. regularWalkWithoutJumps
#5. WalkInOrderlyFashion












