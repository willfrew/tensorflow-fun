set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

LOCAL_BUILD_DIR=$SCRIPT_DIR/build
CONTAINER_SRC_DIR=/src
CONTAINER_BUILD_DIR=/build

IMAGE=tensorflow-fun:latest

# Build
docker build . --tag $IMAGE

# Run
case "$1" in
    "build")
        CMD="python $CONTAINER_SRC_DIR/build.py";;
    "load")
        CMD="python $CONTAINER_SRC_DIR/load.py";;
    "repl")
        CMD="python";;
    *)
        CMD="python";;
esac

mkdir -p $LOCAL_BUILD_DIR

docker \
    run \
    --rm \
    --volume $LOCAL_BUILD_DIR:$CONTAINER_BUILD_DIR:rw \
    -it \
    $IMAGE \
    $CMD
