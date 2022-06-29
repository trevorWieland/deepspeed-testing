#!/bin/bash

echo "pod started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
fi

mkdir /workspace
cd /workspace
git clone https://github.com/huggingface/transformers.git
git clone https://github.com/trevorWieland/deepspeed-testing.git
