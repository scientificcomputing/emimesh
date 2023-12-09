import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from pathlib import Path
import yaml
import os
def getduration(ncells, size, eps=4):
    log = f"envelopsize={eps},processing=ncells-{ncells}_expand-3_smoothiter-2_smoothradius-40_shrink-1,rawdata=minnie65_position-225182-107314-22000_mip-2_size-{size}*"
    logdir = "logs/generateMesh/"
    import fnmatch
    from datetime import datetime
    logfiles = fnmatch.filter(os.listdir(logdir), log)
    datefmt = "%Y-%m-%d %H:%M:%S"

    for logf in logfiles:
        try:
            endline = None
            startline = None
            threadline = None
            with open(logdir + logf) as l:
                for i, line in enumerate(l):
                    if 'took' in line:
                        endline = line
                    if 'TBB threads' in line:
                        threadline = line
                    if "[geogram]" in line and startline is None:
                        startline = line
            start = datetime.strptime(startline.split(".")[0][1:], datefmt)
            end = datetime.strptime(endline.split(".")[0][1:], datefmt)
        except: startline = None
    return end - start, int(threadline.split(" ")[-1][:-1])


def generate_overview_table(paramx_name, paramy_name, paramx, paramy,
                            img_str, substrings):
    
    lenx, leny = len(paramx), len(paramy)
    fig,axes = plt.subplots(leny, lenx, figsize=(lenx*4, leny*4))
    for (py, px), ax, substr in zip(product(paramy, paramx), axes.flatten(), substrings):
        ax.axis('off')
        fn = img_str.format(**{paramx_name:px, paramy_name:py})
        try: img = plt.imread(fn)
        except: continue
        ax.imshow(img[100:-20, 200:-200,:])
        ax.text(0.1, -0.05, substr, c="black", size=12,
                transform=ax.transAxes, va='top')
    
    for i, p in enumerate(paramy):
        axes[i,0].text(-0.1, 0.5, f"{p}", c="black", size=12,
                       rotation="vertical", va='center', ha='center',
                       transform=axes[i,0].transAxes)
        
    for i, p in enumerate(paramx):
        axes[0,i].text(0.5, 1, f"{p}", c="black", size=12,
                       va='center', ha='center',
                       transform=axes[0,i].transAxes)

    fig.supxlabel('Domain edge length (nm)', y=1, fontsize=14)
    fig.supylabel('Number of cells', x=0, fontsize=14)
    fig.tight_layout()
    plt.savefig("overview_table.png", dpi=300, bbox_inches="tight")


eps = 8
paramy =  [5, 10, 50, 100, 200]
paramx =  [5000, 10000, 20000, 40000, 80000]
paramx_name = "size"
paramy_name = "ncells"
processed = "ncells-{ncells}_expand-3_smoothiter-2_smoothradius-40_shrink-1"
raw = "minnie65_position-225182-107314-22000_mip-2_size-{size}"
meshing = f"eps-{eps}"
img_str = f"results/meshes/{raw}/{processed}/{meshing}/mesh.png"


stats = []
for py, px in product(paramy, paramx):
    fn = img_str.format(**{paramx_name:px, paramy_name:py})
    try:
        with open(Path(fn).parent / "meshstatistic.yml") as f:
            mesh_statistic = yaml.load(f, Loader=yaml.UnsafeLoader)
            mesh_statistic["time"], mesh_statistic["threads"] = getduration(py, px, eps=eps)
            mesh_statistic["membrane_point_share"] = mesh_statistic["npoints_membrane"] / mesh_statistic["npoints"]
            mesh_statistic["ecs_tet_share"] = mesh_statistic["ntets_ecs"] / mesh_statistic["ncompcells"]

    except FileNotFoundError:
        mesh_statistic = None
    stats.append(mesh_statistic)

substrings = [("#P {npoints:,}, #T {ncompcells:,}," + 
               "\n{ecs_tet_share:.0%} TECS, {membrane_point_share:.1%} MP," + 
               "\n{nfacets_membrane:,} MF, {ecs_share:.0%} ECS, "+
               "\ntime: {time} ({threads} thr)").format(**s)
           if s is not None else '' for s in stats]

generate_overview_table(paramx_name, paramy_name, paramx, paramy, img_str, substrings)


