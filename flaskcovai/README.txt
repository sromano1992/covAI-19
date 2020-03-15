# Docker build/run/push
docker login
docker build -t sromano41/covai19ml:0.0.1 .
docker run -p 8000:8000 sromano41/covai19ml:0.0.1
docker push sromano41/covai19ml:0.0.1

docker build -t sromano41/covaifronted:0.0.1 .
docker run -p 3000:3000 sromano41/covaifronted:0.0.1
docker push sromano41/covaifronted:0.0.1
