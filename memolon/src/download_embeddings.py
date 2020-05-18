#!/usr/bin/env python

import wget
import pandas as pd
import gzip
import os
import os.path

from memolon.src import utils
import memolon.src.constants as cs

language_table = utils.get_language_table()
iso_codes = language_table.index.tolist()

# downloading the .vec.gz-files
url_root = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/'
for iso in iso_codes:
	print(iso)
	file = 'cc.{}.300.vec.gz'.format(iso)
	target_path = str(cs.EMBEDDINGS / file)
	if not os.path.isfile(target_path):
		print('downloading {}'.format(file))
		url = url_root+file
		wget.download(url, out=target_path)
	else:
		print('{} already exists'.format(file))
