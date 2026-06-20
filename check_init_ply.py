# check_init_ply.py
from argparse import ArgumentParser
from arguments import ModelParams
from scene import Scene, GaussianModel

parser = ArgumentParser()
dataset = ModelParams(parser, sentinel=True)
args = parser.parse_args(["-s", "data", "-m", "data/output", "--depths", ""])

gaussians = GaussianModel(3)
scene = Scene(dataset.extract(args), gaussians)

pcd = scene.scene_info.point_cloud  # 这里如果属性名不同我再帮你改
print("Loaded point cloud:", pcd.points.shape, pcd.colors.shape, pcd.normals.shape)
