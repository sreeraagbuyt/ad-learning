#!/bin/bash

apphome=$(pwd)
export PYTHONPATH=$apphome:${PYTHONPATH}

cmd="python $@"
#cmd="/usr/local/bin/python2.7 $@"
cmd="${cmd%"${cmd##*[![:space:]]}"}"
running=`ps -ef | egrep "$cmd" | grep -v grep | wc -l`
if [ "$running" == "0" ]; then
    $cmd
fi

