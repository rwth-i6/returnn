#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script copied from here and adapted:
# https://github.com/apache/flink/blob/02d30ace69dc18555a5085eccf70ee884e73a16e/tools/azure-pipelines/free_disk_space.sh

# See here for discussion: https://github.com/rwth-i6/returnn/issues/1770

echo "=============================================================================="
echo "Freeing up disk space on CI system"
echo "=============================================================================="

echo "Space usage before installing dependencies:"
df -h
echo "Home realpath: $(realpath ~)"
echo "Listing space usage of home directory:"
time du -h -d1 ~
time du -h -d1 ~/actions-runner
time du -h -d1 ~/work

echo "Listing 100 largest packages:"
dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n | tail -n 100

echo "Removing large packages"
sudo apt-get remove -y 'php.*'
sudo apt-get remove -y azure-cli google-cloud-sdk hhvm microsoft-edge-stable google-chrome-stable firefox powershell mono-devel
sudo apt-get autoremove -y
sudo apt-get clean
df -h
echo "Removing large directories"

# deleting 15GB
rm -rf /usr/share/dotnet/

echo "Space usage after cleanup:"
df -h
