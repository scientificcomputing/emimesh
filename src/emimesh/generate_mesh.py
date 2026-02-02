import time

from pytetwild import tetrahedralize_csg

__all__ = ["make_surfaces"]

def mesh_surfaces(csg_tree_path, eps, stop_quality, max_threads):
    start = time.time()
    mesh = tetrahedralize_csg(csg_tree_path, epsilon=eps, edge_length_r=eps*50, 
                              coarsen=True, stop_energy=stop_quality,
                              num_threads=max_threads).clean()
    print("meshing finished!")
    mesh["label"] = mesh["marker"]
    mesh.field_data['runtime'] = time.time() - start
    mesh.field_data['threads'] = max_threads
    return mesh
