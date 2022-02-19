import numpy as np

def regularWalkWithJumps(mesh_extra, f0, seq_len):
  vertices  = mesh_extra['n_vertices']
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.int32)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = True
  backward_steps = 1
  jump_prob = 1 / (seq_len/5)
  for i in range(1, seq_len + 1):
    this_nbrs = nbrs[seq[i - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump[i] = 0
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
        jump[i] = 1
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump[i] = 2
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
  return seq, jumps


def regularWalk(mesh_extra, f0, seq_len):
  vertices  = mesh_extra['n_vertices']
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.int32)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = True
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
      jumps[i] = 0
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
        jumps[i] = 1
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jumps[i] = 2
        visited[...] = 0
        visited[-1] = True
    visited[to_add] = 1
    seq[i] = to_add
   
  return seq, jumps

def regularWalkWithSkips(mesh_extra, f0, seq_len):
  vertices  = mesh_extra['n_vertices']
  Skips_param  = 2
  seq_len_semi = seq_len*Skips_param
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.int32)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = True
  backward_steps = 1
  jump_prob = 1 / 100
  a = False
  b = False
  c = False
  for i in range(1, (seq_len_semi + 1)):
    this_nbrs = nbrs[seq[(i//Skips_param) - 1]]
    nodes_to_consider = [n for n in this_nbrs if not visited[n]]
    jump_now = np.random.binomial(1, jump_prob)
    if len(nodes_to_consider) and not jump_now:
      to_add = np.random.choice(nodes_to_consider)
      jump = 0
      a = b
      b = c
      c = jump
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[(i//Skips_param) - backward_steps - 1]
        backward_steps += 2
        jump = 1
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = 2
        visited[...] = 0
        visited[-1] = True
      a = b
      b = c
      c = jump
       
    visited[to_add] = 1
    if i % Skips_param ==0:
      seq[i//Skips_param] = to_add
      if jump == 2 or b == 2 or c == 2 or a == 2:
        jumps[i//Skips_param] =2
      elif jump == 1 or b == 1 or c == 1 or a == 1:
        jumps[i//Skips_param] = 1
      else:
        jumps[i//Skips_param] = 0  
  return seq, jumps



def RandomJumps(mesh_extra, f0, seq_len):
  vertices  = mesh_extra['n_vertices']
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.int32)
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  seq[0] = f0
  jumps[0] = True
  backward_steps = 1
  jump_prob = 1 / 100
  for i in range(1, seq_len + 1):
    seq[i] = np.random.randint(n_vertices)
    jumps[i] = True
  return seq, jumps

def regularWalkWithoutJumps(mesh_extra, f0, seq_len):
  vertices  = mesh_extra['n_vertices']
  nbrs = mesh_extra['edges']
  n_vertices = mesh_extra['n_vertices']
  seq = np.zeros((seq_len + 1,), dtype=np.int32)
  jumps = np.zeros((seq_len + 1,), dtype=np.int32)
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
      jumps[i] = 0
      backward_steps = 1
    else:
      if i > backward_steps and not jump_now:
        to_add = seq[i - backward_steps - 1]
        backward_steps += 2
        jumps[i] = 1
      else:
        backward_steps = 1
        to_add = np.random.randint(n_vertices)
        jump = True
        visited[...] = 0
        visited[-1] = True
        jumps[i] = 2
    visited[to_add] = 1
    seq[i] = to_add

  return seq, jumps


def WalkInOrderlyFashion(mesh_extra, f0, seq_len):
  ## Sort Vertices : 
  vertices  = mesh_extra['n_vertices']
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
  jumps[0] = False
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












