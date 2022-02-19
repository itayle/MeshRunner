from numpy.testing._private.utils import tempdir
import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import math
import itertools


def softmax(x,T):
  return np.exp(x/T)/np.sum(np.exp(x/T))

def is_in_sphere(point, ref_point, radius):
  residual = ref_point - point
  return math.sqrt(np.sum(np.power(residual, 2))) < radius


def G(point_cloud, ref_point, mean_curv, sigma):
  up = 0
  down = 0
  in_sphere = 0
  for i, p in enumerate(point_cloud):
    if is_in_sphere(p, ref_point, 2*sigma):
      in_sphere+=1
      temp = math.e**(-np.sum(np.power((p-ref_point),2))/(2*sigma**2))
      up+=mean_curv[i] * temp
      down+= temp
  return up/down if down != 0 else 0

def saliency(mesh, point_cloud, sigma):
  mean_curvature = mesh.curvature()
  saliency_map = []
  for p in point_cloud:
    saliency_map.append(abs(G(point_cloud, p, mean_curvature, sigma) -
                                 G(point_cloud, p, mean_curvature, 2*sigma)))
  return saliency_map


def get_bounding_box_diagonal(vertices):
  x_max, y_max, z_max = vertices.max(axis=0)
  x_min, y_min, z_min = vertices.min(axis=0)
  diffs = vertices.max(axis=0) - vertices.min(axis=0)
  radius = diffs.max()
  diagonal = math.sqrt((x_max - x_min)**2 + (y_max - y_min)**2 + (z_max - z_min)**2)
  return diagonal


def compute_saliency(mesh_data, mem_dict):
    name = f"{mesh_data}"
    if name not in mem_dict:
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        t_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh = pv.wrap(t_mesh)

        diagonal = get_bounding_box_diagonal(vertices)
        maps = []
        for s in [2,3,4,5,6]:
            maps.append(saliency(mesh, vertices, diagonal*s* 0.04))
        saliency_array = softmax(np.mean(maps, axis=0),40)
        mem_dict[name]=saliency_array
    return mem_dict[name]
