import os
import sys
import time

import numpy as np
from tqdm import tqdm

eva_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(eva_dir)
from src.catalog.entity.dataset import Dataset
from src.catalog.mapping_manager import MappingManager
from src.catalog.sqlite_connection import SqliteConnection
from src.operations.blur import Blur
from src.operations.grayscale import Grayscale
from src.storage.loader_uadetrac import LoaderUadetrac
from src.storage.opr_manager import OperationsManager

UADETRAC = 'uadetrac'
eva_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
cache_dir = os.path.join(eva_dir, 'cache')
catalog_dir = os.path.join(eva_dir, 'src', 'catalog')

conn = SqliteConnection(os.path.join(catalog_dir, 'eva.db'))
conn.connect()

dataset = Dataset.get(conn, UADETRAC)
loader = LoaderUadetrac(dataset.loaded_width, dataset.loaded_height, cache_dir)
loader.load_cached_frames()
mmanager = MappingManager(UADETRAC)

num_images = loader.images.shape[0]
# frame_ids = np.random.choice(num_images, 1000, False)


# exp1 ---------------------------------------------------
# loader.load_cached_frames()
gray_ids = np.random.choice(num_images, 1000, False)
blur_ids = np.random.choice(num_images, 1000, False)
all_ids = np.arange(num_images)
dirty = np.unique(np.hstack((gray_ids, blur_ids)))
non_dirty = np.setdiff1d(all_ids, dirty)
read_ids = np.hstack((np.random.choice(dirty, 0, False), np.random.choice(
    non_dirty, 1000, False)))

# print(len(frame_ids_gray))
# write
w1 = []
r1 = []
for i in tqdm(range(10)):
    loader.load_cached_frames()
    s = time.time()
    frames = loader.get_frames(gray_ids)
    o1 = Grayscale()
    frames = o1.apply(frames)
    loader.update_images(frames)

    frames = loader.get_frames(blur_ids)
    o2 = Blur()
    frames = o2.apply(frames)
    loader.update_images(frames)

    e = time.time()
    w1.append(e - s)
    # print(e - s)

    # read
    s = time.time()
    loader.get_frames(read_ids)
    e = time.time()
    r1.append(e - s)

# print((e - s))

# exp2
# loader.load_cached_frames()
# print(len(frame_ids))
r2 = []
w2 = []

# write
for i in tqdm(range(10)):
    loader.load_cached_frames()
    opr_manager = OperationsManager()
    s = time.time()
    o1 = Grayscale()
    o2 = Blur()
    opr_manager.add_opr(o1, gray_ids)
    opr_manager.add_opr(o2, blur_ids)
    e = time.time()
    w2.append(e - s)
    # print((e - s))

    # read
    s = time.time()
    frames = []
    for frame_id in read_ids:
        frame = loader.get_frames([frame_id])[0]
        opr_list = opr_manager.get_opr_list(frame.id)
        for opr in opr_list:
            frame = opr.apply([frame])[0]
        frames.append(frame)
    e = time.time()
    r2.append(e - s)
    # loader.load_cached_frames()

print(r1)
print(w1)
print(r2)
print(w2)
print("%.5f %.5f %.5f %.5f" % (np.array(r1).mean(), np.array(w1).mean(),
                               np.array(r2).mean(), np.array(w2).mean()))
