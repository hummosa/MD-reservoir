#! /bin/bash
rsync -v ~/code/MD/* ahummos@openmind7.mit.edu:~/code/MD
rsync -vr ahummos@openmind7.mit.edu:~/code/MD/* ~/code/MD

