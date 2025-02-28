import argparse
import base64
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import caffe
import cv2
import habitat
import numpy as np
from habitat import Config
from habitat.core.dataset import Dataset
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat_sim.geo import GRAVITY as GRAVITY_AXIS
from habitat_sim.geo import RIGHT as RIGHT_AXIS
from habitat_sim.utils import quat_rotate_vector
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_two_vectors,
)
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


TSV_FIELDNAMES = [
    "scanId",
    "viewpointId",
    "image_w",
    "image_h",
    "vfov",
    "raw_rgb",
    "processed_rgb",
]
VIEWPOINT_SIZE = 36
FEATURE_SIZE = 2048

# Simulator image parameters
WIDTH = 640
HEIGHT = 480
VFOV = 60
HFOV = 75  # makes a VFOV of 60

ROTATION_QUAT = quat_from_two_vectors(np.array([0, 0, 1]), np.array([0, 1, 0]))
HEADINGS = [
    quat_from_angle_axis(np.deg2rad(30.0 * h), GRAVITY_AXIS) for h in range(12)
]
ELEVATIONS = [-30.0, 0, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caffe-prototxt",
        type=str,
        default="data/caffe_models/ResNet-152-deploy.prototxt",
        help="Path to the ResNet .prototxt",
    )
    parser.add_argument(
        "--caffe-model",
        type=str,
        default="data/caffe_models/resnet152_places365.caffemodel",
        help="Path to the ResNet .caffemodel",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        default="data/data_for_training_sgm/raw_rgb",
        help="location to save the computed features",
    )
    parser.add_argument(
        "--connectivity",
        type=str,
        default="connectivity",
        help="location of MP3D connectivity folder",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default="data/scene_datasets/mp3d",
        help="location of MP3D scenes containing `{scene_id}/{scene_id}.glb`",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=9,
        help="Some fraction of viewpoint size",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to run the Caffe model and HabitatSim",
    )
    return parser.parse_args()


def get_habitat_config(gpu_id: int) -> Config:
    cfg = habitat.get_config()

    cfg.defrost()
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.SIMULATOR.AGENT_0.HEIGHT = 0.0
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = WIDTH
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = HEIGHT
    cfg.SIMULATOR.RGB_SENSOR.HFOV = HFOV
    cfg.SIMULATOR.RGB_SENSOR.POSITION = [0, 0, 0]  # offset from MP3D pose
    cfg.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
    cfg.freeze()

    return cfg


def load_viewpoints(conn: str) -> Tuple[Dict[str, List], int]:
    """Loads viewpoints into a dictionary of sceneId -> list(viewpoints)
    where each viewpoint has keys {viewpointId, pose, height}.
    """
    viewpoints = []
    with open(os.path.join(conn, "scans.txt")) as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(os.path.join(conn, f"{scan}_connectivity.json")) as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpoint_data = {
                            "viewpointId": item["image_id"],
                            "pose": item["pose"],
                            "height": item["height"],
                        }
                        viewpoints.append((scan, viewpoint_data))

    scans_to_vps = defaultdict(list)
    for scene_id, viewpoint in viewpoints:
        scans_to_vps[scene_id].append(viewpoint)

    return scans_to_vps, len(viewpoints)


def precompute_features(
    scenes_dir,
    scanId,
    viewpoints,
    habitat_config,
    image_dict,
    pbar,
    save_to
):
    def habitat_dataset_single_scene(
        scenes_dir: str, scene_id: str
    ) -> Dataset:
        dataset = Dataset()
        dataset.episodes = [
            NavigationEpisode(
                episode_id=0,
                scene_id=os.path.join(scenes_dir, scene_id, f"{scene_id}.glb"),
                goals=[NavigationGoal(position=[0.0, 0.0, 0.0], radius=3.0)],
                start_position=[0.0, 0.0, 0.0],
                start_rotation=[0.0, 0.0, 0.0, 1.0],
            )
        ]
        return dataset

    def habitat_coordinates_from_pose(pose):
        pose = np.reshape(np.asarray(pose), (4, 4))
        mp3d_sim_coords = pose[:-1, -1]
        habitat_coords = quat_rotate_vector(ROTATION_QUAT, mp3d_sim_coords)
        return habitat_coords.tolist()

    def transform_obs(obs):
        img = np.array(obs["rgb"], copy=True)
        img = img.astype(np.float32, copy=True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img -= np.array([[[103.1, 115.9, 123.2]]])  # BGR pixel mean
        blob = np.zeros((1, img.shape[0], img.shape[1], 3), dtype=np.float32)
        blob[0] = img
        return blob.transpose((0, 3, 1, 2))

    env = habitat.Env(
        config=habitat_config,
        dataset=habitat_dataset_single_scene(scenes_dir, scanId),
    )
    env.reset()

    for viewpoint in viewpoints:
        viewpointId = viewpoint["viewpointId"]
        coords = habitat_coordinates_from_pose(viewpoint["pose"])

        raw_rgb = np.empty([VIEWPOINT_SIZE, HEIGHT, WIDTH, 3], dtype=np.uint8)
        idx = 0
        for elevation in ELEVATIONS:
            for heading in HEADINGS:
                elev_rot = quat_from_angle_axis(
                    np.deg2rad(elevation), RIGHT_AXIS
                )
                rot = heading * elev_rot

                obs = env.sim.get_observations_at(
                    position=coords, rotation=rot
                )

                raw_rgb[idx] = obs["rgb"]
        
        if os.path.exists(os.path.join(save_to, scanId, f"{viewpointId}.npz")):
            print(f"Already saved {viewpointId} raw rgb")
        else:
            os.makedirs(os.path.join(save_to, scanId), exist_ok=True)

            np.savez_compressed(
                os.path.join(save_to, scanId, f"{viewpointId}.npz"),
                raw_rgb=raw_rgb,
            )

            del raw_rgb

        pbar.update()

    env.close()

    return image_dict

def save_features_npy(features_dict, save_to):
    # Save the dictionary using np.save with pickling enabled.
    np.save(save_to, features_dict, allow_pickle=True)
    print(f"Features saved to {save_to}")

def main(args):
    habitat_config = get_habitat_config(args.gpu_id)
    scans_to_vps, num_vps = load_viewpoints(args.connectivity)
    print(f"Loaded {num_vps} viewpoints")

    pbar = tqdm(total=num_vps)

    image_dict = {}
    for scan, vps in scans_to_vps.items():
        precompute_features(
            args.scenes_dir,
            scan,
            vps,
            habitat_config,
            image_dict,
            pbar,
            args.save_to,
        )

if __name__ == "__main__":
    main(parse_args())
