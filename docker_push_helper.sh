docker build -f docker/DependencyDockerfile -t base:latest .
docker tag base:latest mlopskrupamay.azurecr.io/base:latest
docker push mlopskrupamay.azurecr.io/base:latest

docker build -f docker/FinalDockerfile -t digits:latest .
docker digits:latest mlopskrupamay.azurecr.io/digits:latest
docker push mlopskrupamay.azurecr.io/digits:latest