#!/bin/bash

python3 -c "import deeplake; deeplake.copy('hub://haritsahm/camvid_train', 'data/camvid_train', verbose=True)"
python3 -c "import deeplake; deeplake.copy('hub://haritsahm/camvid_val', 'data/camvid_val', verbose=True)"
python3 -c "import deeplake; deeplake.copy('hub://haritsahm/camvid_test', 'data/camvid_test', verbose=True)"
