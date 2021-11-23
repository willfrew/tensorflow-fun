set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

LOCAL_BUILD_DIR=$SCRIPT_DIR/build
CONTAINER_SRC_DIR=/src
CONTAINER_BUILD_DIR=/build

IMAGE=tensorflow-fun:latest

docker build . --tag $IMAGE

function usage() {
    echo "Usage: ./run.sh (<project> build|load) | repl"
}

function error() {
    echo "Error:" $1
    usage
    exit 1
}

case "$1" in
    "repl")
        CMD="python";;
    "mnist")
        PROJ="01-mnist";;
    *)
        error "Command not recognised"
esac

if [ "$PROJ" != "" ]; then
    case "$2" in
        "build")
            CMD="python $CONTAINER_SRC_DIR/$PROJ/build.py";;
        "load")
            CMD="python $CONTAINER_SRC_DIR/$PROJ/load.py";;
        *)
            error "Command not recognised"
    esac
fi


mkdir -p $LOCAL_BUILD_DIR

docker \
    run \
    --rm \
    --volume $LOCAL_BUILD_DIR:$CONTAINER_BUILD_DIR:rw \
    -it \
    $IMAGE \
    $CMD
