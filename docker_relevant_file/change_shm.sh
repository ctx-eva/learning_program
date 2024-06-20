#!/bin/bash

set -e
CONTAINER_NAME="$1"
if [ -z "$CONTAINER_NAME" ]; then
        echo "ERROR - Usage: $0 <container-name> [shm-size]"
        exit 1
fi
if [ ! -z "$2" ]; then
        SHM_SIZE="$2"
else
        SHM_SIZE="1073741824"
fi

echotime() {
        echo [$(date "+%Y-%m-%d %H:%M:%S")] $@
}

CONTAINER_ID=$(docker inspect -f '{{ .ID }}' $CONTAINER_NAME)
HOST_CONFIG=/var/lib/docker/containers/$CONTAINER_ID/hostconfig.json

echotime "Stopping docker service"
service docker stop

echotime "Changing container's hostconfig.json"
sed -i 's/"ShmSize":[0-9]\+,/"ShmSize":'$SHM_SIZE',/' $HOST_CONFIG

echotime "Starting docker service"
service docker start
