#!/usr/bin/env python

import sys
import numpy as np

output_filename = sys.argv[1]

results = []
for i in xrange(2, len(sys.argv)):
    results.append (np.load (sys.argv[i]))

np.save (output_filename, np.array(results))
