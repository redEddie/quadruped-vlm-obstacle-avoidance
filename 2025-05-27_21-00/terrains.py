from __future__ import annotations

import isaaclab.terrains as terrain_gen
import inspect
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from isaaclab.terrains.trimesh import mesh_terrains_cfg as mtc
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshGapTerrainCfg,
    MeshRailsTerrainCfg,
    MeshPitTerrainCfg,
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshStarTerrainCfg,
    MeshRepeatedObjectsTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
)

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane


from dataclasses import MISSING
from typing import Literal

# import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
import random
import numpy as np
import trimesh
import torch
import random

from dataclasses import MISSING
from typing import Tuple, List

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBase
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.terrains import FlatPatchSamplingCfg

from isaaclab.utils import configclass

# ------------------------------------------------------------
# 1) 미로 생성 로직 및 Cfg 정의 (maze_terrain.py에서 가져온 것)
# ------------------------------------------------------------
def generate_maze_grid(rows: int, cols: int) -> np.ndarray:
    grid = np.ones((2*rows+1, 2*cols+1), dtype=int)
    visited = np.zeros((rows, cols), dtype=bool)
    def carve(r, c):
        visited[r, c] = True
        grid[2*r+1, 2*c+1] = 0
        for dr, dc in np.random.permutation([(1,0),(-1,0),(0,1),(0,-1)]):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                grid[r+nr+1, c+nc+1] = 0
                carve(nr, nc)
    carve(0, 0)
    return grid

def maze_terrain(
    difficulty: float,
    cfg: MazeTerrainCfg
) -> Tuple[List[trimesh.Trimesh], np.ndarray]:
    grid = generate_maze_grid(cfg.rows, cfg.cols)

    # 시작점과 끝점 통로로
    grid[1, 0] = 0
    grid[-2, -1] = 0

    meshes: List[trimesh.Trimesh] = []
    cs, ht, wt = cfg.cell_size, cfg.wall_height, cfg.wall_thickness

    # 벽 생성
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                center = (j*cs, i*cs, ht/2.0)
                dims = (cs+wt, cs+wt, ht)
                box = trimesh.creation.box(
                    dims,
                    trimesh.transformations.translation_matrix(center)
                )
                meshes.append(box)

    # 바닥
    grid_w = grid.shape[1]*cs+wt
    grid_h = grid.shape[0]*cs+wt
    floor_thickness = 0.01
    dim = (grid_w, grid_h, floor_thickness)
    pos = (grid_w/2-cs/2, grid_h/2-cs/2, -floor_thickness/2)
    floor = trimesh.creation.box(
        dim,
        trimesh.transformations.translation_matrix(pos)
    )
    meshes.append(floor)

    # 스폰 위치: (0,0) 셀 중심
    origin = np.array([cs, cs, cfg.spawn_height])
    return meshes, origin

@configclass
class MazeTerrainCfg(SubTerrainBaseCfg):
    function = maze_terrain
    rows: int = MISSING
    cols: int = MISSING
    cell_size: float = MISSING
    wall_thickness: float = 0.05
    wall_height: float = 0.5
    spawn_height: float = 0.1