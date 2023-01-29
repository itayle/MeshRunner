# MeshRunner - Improved classification of 3D mesh objects
Code for our [MeshRunner project](MeshRunner.pdf)


* [About](#About)
    * [Authors](#Authors)
    * [Project](#Project)
    * [Acknowledgements](#Acknowledgements)
* [Background](#MeshWalker-Background)
* [Colab notebook](#Colab-Notebook)
# About
In this work we tried to improve several aspects in [MeshWalker](https://arxiv.org/abs/2006.05353):
- Improved walk generation
  - Choose starting point by analyzing saliency
  - Apply skips, jumps and ordering
- Improved walk representation
- Attention based information exchange between walks

## Authors
The authors of this project are:
- Itay Levy
- Itamar Zimerman
- Amit Cohen


## Acknowledgements
Prof. Amit Bermano - Guided us thorought the project <br>
Alon Lahav - Help reproducing MeshWalker results <br>
Adi Mesika - Helped with visualizations


# MeshWalker Background
- Apply Deep Learning directly on meshes
- Perform random walks on the mesh’s surface and feed it to an RNN
- Use the RNN’s hidden state to represent the mesh
- Perform classification (or segmentation) over the representation

# Colab Notebook
Checkout our colab notebook for a demo of this project: <br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/itayle/MeshRunner/blob/main/MeshRunner.ipynb)
