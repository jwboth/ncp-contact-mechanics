#!/bin/bash

# Fetch current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# PorePy directory
POREPY_DIR=$HOME/porepy

# Switch to PorePy directory
cd $POREPY_DIR

# Checkout the develop branch at commit f3f14e14a06f5e8245a378502d8c02edebe537a7
git pull origin develop
git checkout f3f14e14a06f5e8245a378502d8c02edebe537a7

# Get back to the original directory
cd $DIR
