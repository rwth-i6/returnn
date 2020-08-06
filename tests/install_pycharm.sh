#!/usr/bin/env bash

set -xe

name="pycharm-community-2020.2"
fn="$name.tar.gz"

wget -c "https://download.jetbrains.com/python/$fn"

tar -xzvf $fn | tail
test -d $name
rm $fn
mv $name pycharm
