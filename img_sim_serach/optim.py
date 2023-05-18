import cv2
import numpy
import time
import csv
from glob import glob
from pathlib import Path
from statistics import mean

from towhee import pipe, ops, DataCollection
from towhee.types.image import Image
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Towhee parameters
MODEL = 'vgg16'
DEVICE = None # if None, use default device (cuda is enabled if available)

# Milvus parameters
HOST = '127.0.0.1'
PORT = '19530'
TOPK = 10
DIM = 512 # dimension of embedding extracted, change with MODEL
COLLECTION_NAME = 'deep_dive_image_search_' + MODEL
INDEX_TYPE = 'IVF_FLAT'
METRIC_TYPE = 'L2'

# patterns of image paths
INSERT_SRC = './train/*/*.JPEG'
QUERY_SRC = './test/*/*.JPEG'

to_insert = glob(INSERT_SRC)
to_test = glob(QUERY_SRC)

# Create milvus collection (delete first if exists)
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='path', dtype=DataType.VARCHAR, description='path to image', max_length=500, 
                    is_primary=True, auto_id=False),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': METRIC_TYPE,
        'index_type': INDEX_TYPE,
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    return collection
            
# Read images
decoder = ops.image_decode('rgb').get_op()
def read_images(img_paths):
    imgs = []
    for p in img_paths:
        img = decoder(p)
        imgs.append(img)
#         imgs.append(Image(cv2.imread(p), 'RGB'))
    return imgs

# Get ground truth
def ground_truth(path):
    train_path = str(Path(path).parent).replace('test', 'train')
    return [str(Path(x).resolve()) for x in glob(train_path + '/*.JPEG')]

# Calculate Average Precision
def get_ap(pred: list, gt: list):
    ct = 0
    score = 0.
    for i, n in enumerate(pred):
        if n in gt:
            ct += 1
            score += (ct / (i + 1))
    if ct == 0:
        ap = 0
    else:
        ap = score / ct
    return ap

# Embedding pipeline
p_embed = (
    pipe.input('img_path')
        .map('img_path', 'img', ops.image_decode('rgb'))
        .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
        .map('vec', 'vec', lambda x: x / numpy.linalg.norm(x, axis=0))
)

p_display = p_embed.output('img_path', 'img', 'vec')

DataCollection(p_display(to_insert[0])).show()

# Connect to Milvus service
connections.connect(host=HOST, port=PORT)

# Create collection
collection = create_milvus_collection(COLLECTION_NAME, DIM)
print(f'A new collection created: {COLLECTION_NAME}')

# Insert data
p_insert = (
        p_embed.map(('img_path', 'vec'), 'mr', ops.ann_insert.milvus_client(
                    host=HOST,
                    port=PORT,
                    collection_name=COLLECTION_NAME
                    ))
          .output('mr')
)

for img_path in to_insert:
    p_insert(img_path)
print('Number of data inserted:', collection.num_entities)

# Performance
collection.load()
p_search_pre = (
        p_embed.map('vec', ('search_res'), ops.ann_search.milvus_client(
                    host=HOST, port=PORT, limit=TOPK,
                    collection_name=COLLECTION_NAME))
               .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
#                .output('img_path', 'pred')
)
p_eval = (
    p_search_pre.map('img_path', 'gt', ground_truth)
                .map(('pred', 'gt'), 'ap', get_ap)
                .output('ap')
)

res = []
for img_path in to_test:
    ap = p_eval(img_path).get()[0]
    res.append(ap)

mAP = mean(res)

print(f'mAP@{TOPK}: {mAP}')


p_search_img = (
    p_search_pre.map('img_path', 'gt', ground_truth)
                .map(('pred', 'gt'), 'ap', get_ap)
                .map('pred', 'res', read_images)
                .output('img_path', 'img', 'res', 'ap')
)
DataCollection(p_search_img('./test/rocking_chair/n04099969_23803.JPEG')).show()

def get_max_object(img, boxes):
    if len(boxes) == 0:
        return img
    max_area = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        area = (x2-x1)*(y2-y1)
        if area > max_area:
            max_area = area
            max_img = img[y1:y2,x1:x2,:]
    return max_img

p_yolo = (
    pipe.input('img_path')
        .map('img_path', 'img', ops.image_decode('rgb'))
        .map('img', ('boxes', 'class', 'score'), ops.object_detection.yolov5())
        .map(('img', 'boxes'), 'object', get_max_object)
)

# Display embedding result, no need for implementation
p_display = (
    p_yolo.output('img', 'object')
)
DataCollection(p_display('./test/rocking_chair/n04099969_23803.JPEG')).show()

# Search
p_search_pre_yolo = (
    p_yolo.map('object', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE))
          .map('vec', 'vec', lambda x: x / numpy.linalg.norm(x, axis=0))
          .map('vec', ('search_res'), ops.ann_search.milvus_client(
                host=HOST, port=PORT, limit=TOPK,
                collection_name=COLLECTION_NAME))
          .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
#          .output('img_path', 'pred')
)

# Evaluate with AP
p_search_img_yolo = (
    p_search_pre_yolo.map('img_path', 'gt', ground_truth)
                     .map(('pred', 'gt'), 'ap', get_ap)
                     .map('pred', 'res', read_images)
                     .output('img', 'object', 'res', 'ap')
)
DataCollection(p_search_img_yolo('./test/rocking_chair/n04099969_23803.JPEG')).show()

NEW_MODEL = 'tf_efficientnet_b7'
OLD_DIM = 2560
NEW_DIM = 512
NEW_COLLECTION_NAME = NEW_MODEL + '_' + str(NEW_DIM)

numpy.random.seed(2023)
projection_matrix = numpy.random.normal(scale=1.0, size=(OLD_DIM, NEW_DIM))

def dim_reduce(vec):
    return numpy.dot(vec, projection_matrix)

connections.connect(host=HOST, port=PORT)
new_collection = create_milvus_collection(NEW_COLLECTION_NAME, NEW_DIM)
print(f'A new collection created: {NEW_COLLECTION_NAME}')


# Embedding pipeline
p_embed = (
    pipe.input('img_path')
        .map('img_path', 'img', ops.image_decode('rgb'))
        .map('img', 'vec', ops.image_embedding.timm(model_name=NEW_MODEL, device=DEVICE))
        .map('vec', 'vec', dim_reduce)
)

# Display embedding result, no need for implementation
p_display = p_embed.output('img_path', 'img', 'vec')

DataCollection(p_display(to_insert[0])).show()

# Insert pipeline
p_insert = (
        p_embed.map(('img_path', 'vec'), 'mr', ops.ann_insert.milvus_client(
                    host=HOST,
                    port=PORT,
                    collection_name=NEW_COLLECTION_NAME
                    ))
          .output('mr')
)

# Insert data
for img_path in to_insert:
    p_insert(img_path)
print('Number of data inserted:', new_collection.num_entities)

# Search pipeline
new_collection.load()
p_search_pre = (
        p_embed.map('vec', 'search_res', ops.ann_search.milvus_client(
                    host=HOST, port=PORT, limit=TOPK,
                    collection_name=NEW_COLLECTION_NAME))
               .map('search_res', 'pred', lambda x: [str(Path(y[0]).resolve()) for y in x])
#                .output('img_path', 'pred')
)

# Performance
p_eval = (
    p_search_pre.map('img_path', 'gt', ground_truth)
                .map(('pred', 'gt'), 'ap', get_ap)
                .output('ap')
)

res = []
for img_path in to_test:
    ap = p_eval(img_path).get()[0]
    res.append(ap)

mAP = mean(res)

print(f'mAP@{TOPK}: {mAP}')

