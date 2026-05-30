import os
from scene.dataset_readers import (
    readCamerasFromTransforms, CameraInfo,
    getNerfppNorm, BasicPointCloud, SH2RGB, storePly, fetchPly, SceneInfo
)
import numpy as np
from PIL import Image, ImageOps
from utils.graphics_utils import focal2fov, fov2focal
from pathlib import Path
import math
from glob import glob


camera_angle_x = 0.6911112070083618

def create_transform_matrix(distance):
    transform_matrix = [
        [-np.sign(distance) ,0.0,0.0,0.0],
        [.0,0.0, np.sign(distance), distance ],
        [0.0, 1.0, 0.0,0.0],
        [0.0,0.0, 0.0,1.0]
    ]
    return transform_matrix


def _compute_gap_indices(filepaths, gap_start_frac, gap_end_frac):
    if gap_start_frac < 0 or gap_end_frac < 0:
        return set()
    if not (0.0 <= gap_start_frac < gap_end_frac <= 1.0):
        raise ValueError(
            f"Invalid gap fractions: gap_start_frac={gap_start_frac}, gap_end_frac={gap_end_frac}. "
            "Expected 0 <= start < end <= 1."
        )
    num_frames = len(filepaths)
    if num_frames == 0:
        return set()
    start_rank = int(math.floor(num_frames * gap_start_frac))
    end_rank = int(math.ceil(num_frames * gap_end_frac))
    end_rank = min(num_frames, max(start_rank + 1, end_rank))
    return {idx for idx, _ in filepaths[start_rank:end_rank]}


def _frame_partition(idx, holdout_stride, holdout_offset, second_holdout_offset, gap_indices=None):
    if gap_indices and idx in gap_indices:
        return "gap"

    if holdout_stride <= 0:
        return "train"

    if ((idx - holdout_offset) % holdout_stride) == 0:
        return "primary"

    if second_holdout_offset >= 0 and ((idx - second_holdout_offset) % holdout_stride) == 0:
        return "secondary"

    return "train"


def _include_frame(idx, split, holdout_stride, holdout_offset, second_holdout_offset, test_split, gap_indices=None):
    partition = _frame_partition(idx, holdout_stride, holdout_offset, second_holdout_offset, gap_indices)

    if split == "train":
        return partition == "train"

    if split != "test":
        raise ValueError(f"Unknown split: {split}")

    if holdout_stride <= 0:
        return True

    if test_split == "primary":
        return partition == "primary"
    if test_split == "secondary":
        return partition == "secondary"
    if test_split == "gap":
        return partition == "gap"
    if test_split == "all":
        return partition in {"primary", "secondary"}

    raise ValueError(f"Unknown test_split: {test_split}")


def _keep_train_pool_rank(train_pool_rank, train_pool_stride, train_pool_offset):
    if train_pool_stride <= 1:
        return True
    return ((train_pool_rank - train_pool_offset) % train_pool_stride) == 0

def readImage(
    path,
    white_background,
    eval,
    distance,
    num_pts,
    extension=".png",
    holdout_stride=0,
    holdout_offset=0,
    second_holdout_offset=-1,
    test_split="primary",
    train_pool_stride=1,
    train_pool_offset=0,
    gap_start_frac=-1.0,
    gap_end_frac=-1.0,
):
    print("Creating Training Transform")
    train_cam_infos = CreateCamerasTransforms(
        path,
        white_background,
        [-distance],
        extension,
        split="train",
        holdout_stride=holdout_stride,
        holdout_offset=holdout_offset,
        second_holdout_offset=second_holdout_offset,
        test_split=test_split,
        train_pool_stride=train_pool_stride,
        train_pool_offset=train_pool_offset,
        gap_start_frac=gap_start_frac,
        gap_end_frac=gap_end_frac,
    )
    print("Creating Test Transform")
    test_cam_infos = CreateCamerasTransforms(
        path,
        white_background,
        [-distance],
        extension,
        split="test",
        holdout_stride=holdout_stride,
        holdout_offset=holdout_offset,
        second_holdout_offset=second_holdout_offset,
        test_split=test_split,
        train_pool_stride=train_pool_stride,
        train_pool_offset=train_pool_offset,
        gap_start_frac=gap_start_frac,
        gap_end_frac=gap_end_frac,
    )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    # Since this data set has no colmap data, we start with random points
    camera = train_cam_infos[0]
    top = distance * math.tan(camera.FovY * 0.5)
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=1.0,
    )
    return scene_info


def readMirrorImages(
    path,
    white_background,
    eval,
    distance,
    num_pts,
    extension=".png",
    holdout_stride=0,
    holdout_offset=0,
    second_holdout_offset=-1,
    test_split="primary",
    train_pool_stride=1,
    train_pool_offset=0,
    gap_start_frac=-1.0,
    gap_end_frac=-1.0,
):
    print("Creating Training Transforms")
    train_cam_infos = CreateCamerasTransforms(
        path,
        white_background,
        [-distance, distance],
        extension,
        split="train",
        holdout_stride=holdout_stride,
        holdout_offset=holdout_offset,
        second_holdout_offset=second_holdout_offset,
        test_split=test_split,
        train_pool_stride=train_pool_stride,
        train_pool_offset=train_pool_offset,
        gap_start_frac=gap_start_frac,
        gap_end_frac=gap_end_frac,
    )
    print("Creating Test Transforms")
    test_cam_infos = CreateCamerasTransforms(
        path,
        white_background,
        [-distance],
        extension,
        split="test",
        holdout_stride=holdout_stride,
        holdout_offset=holdout_offset,
        second_holdout_offset=second_holdout_offset,
        test_split=test_split,
        train_pool_stride=train_pool_stride,
        train_pool_offset=train_pool_offset,
        gap_start_frac=gap_start_frac,
        gap_end_frac=gap_end_frac,
    )

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")

    # Since this data set has no colmap data, we start with random points
    camera = train_cam_infos[0]
    top = distance * math.tan(camera.FovY * 0.5)
    aspect_ratio = camera.width / camera.height
    right = top * aspect_ratio
    print(f"Generating random point cloud ({num_pts})...")
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.uniform(low=[-right, 0, -top], high=[right, 0, top], size=(num_pts, 3))
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        maxtime=1.0,
    )

    return scene_info

def CreateCamerasTransforms(
    path: str,
    white_background,
    distances,
    extension=".png",
    split="train",
    holdout_stride=0,
    holdout_offset=0,
    second_holdout_offset=-1,
    test_split="primary",
    train_pool_stride=1,
    train_pool_offset=0,
    gap_start_frac=-1.0,
    gap_end_frac=-1.0,
):
    cam_infos = []

    filepaths = glob(f"{path}/original/*{extension}")
    num_frames = len(filepaths)
    filepaths = [(int(os.path.basename(filepath).replace(extension, "")), filepath) for filepath in filepaths]
    filepaths.sort()
    gap_indices = _compute_gap_indices(filepaths, gap_start_frac, gap_end_frac)
    train_pool_rank = -1
    for idx, original in filepaths:
        partition = _frame_partition(idx, holdout_stride, holdout_offset, second_holdout_offset, gap_indices)
        if partition == "train":
            train_pool_rank += 1

        if not _include_frame(
            idx,
            split,
            holdout_stride,
            holdout_offset,
            second_holdout_offset,
            test_split,
            gap_indices,
        ):
            continue
        if split == "train" and not _keep_train_pool_rank(train_pool_rank, train_pool_stride, train_pool_offset):
            continue
        fovx = camera_angle_x
        cam_name_init = original
        cam_name_mirror = original.replace("original", "mirror")

        for i in range(len(distances)):
            distance = distances[i]
            if i == 0:
                cam_name = cam_name_init
            if i == 1:
                cam_name = cam_name_mirror
                if not os.path.exists(cam_name):
                    # save mirror image
                    im = Image.open(cam_name_init)
                    im_flip = ImageOps.mirror(im)
                    im_flip.save(cam_name_mirror)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(create_transform_matrix(distance))
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"), dtype=np.uint8)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            # Keep preprocessing in float32, then drop norm_data to avoid storing
            # hundreds of full-resolution RGBA arrays across image/seg scenes.
            norm_data = im_data.astype(np.float32) / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx
            cam_infos.append(
                CameraInfo(
                    uid=i, 
                    R=R, 
                    T=T, 
                    FovY=FovY, 
                    FovX=FovX, 
                    image=image,
                    image_path=image_path, 
                    image_name=image_name, 
                    width=image.size[0],
                    height=image.size[1], 
                    time=idx,
                    mask=None,
                    norm_data=None
                )
            )
    return cam_infos
