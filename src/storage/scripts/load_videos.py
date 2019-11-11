import os

from src.catalog.entity.dataset import Dataset
from src.catalog.mapping_manager import MappingManager
from src.catalog.sqlite_connection import SqliteConnection
from src.storage.loader_uadetrac import LoaderUadetrac
from src.operations.grayscale import Grayscale
from src.operations.blur import Blur
from src.storage.opr_manager import OperationsManager
import time

UADETRAC = 'uadetrac'
eva_dir = os.path.dirname(os.path.dirname(os.getcwd()))
cache_dir = os.path.join(eva_dir, 'cache')
catalog_dir = os.path.join(eva_dir, 'src', 'catalog')

conn = SqliteConnection(os.path.join(catalog_dir, 'eva.db'))
conn.connect()
# conn.exec_script(os.path.join(catalog_dir, 'scripts',
# 'create_table.sql'))

# conn.exec_script(os.path.join(catalog_dir, 'scripts',
# 'create_table.sql'))

# create dataset

Dataset.delete_all(conn.conn, UADETRAC)
loaded_width = loaded_height = 300
Dataset.create(conn.conn, UADETRAC, 540, 960, loaded_height, loaded_width)
#
# # use dataset
dataset = Dataset.get(conn, UADETRAC)
# print(dataset)
#
# # get loader for dataset
loader = LoaderUadetrac(loaded_width, loaded_height, cache_dir)

# load video/frames
data_dir = os.path.join(eva_dir, 'data', 'ua_detrac',
                        'Insight-MVT_Annotation_Test')

video_dir_list = os.listdir(data_dir)
video_dir_list.sort()
for video_dir in video_dir_list[:5]:
    loader.load_images(os.path.join(data_dir, video_dir))
print("Number of images loaded: %s" % len(loader.images))

# print(loader.images.shape)
loader.save_images()

mmanager = MappingManager(UADETRAC)
mmanager.drop_mapping(conn)
mmanager.create_table(conn)
mmanager.add_frame_mapping(conn, loader.video_start_indices,
                           loader.images.shape[0])

# loader.load_cached_frames()
frame_ids = mmanager.get_frame_ids(conn, 3)
print(frame_ids)
# frames = loader.get_frames(frame_ids)
# images_list = list(images_arr)
# print(len(frames))


# opr_manager = OperationsManager()
