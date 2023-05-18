python -m pip install -q towhee>=0.9.0 pymilvus opencv-python pillow
curl -L https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip -O
unzip -q -o reverse_image_search.zip
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d