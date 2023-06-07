#!/usr/bin/env python3

import json
import sys

for line in open(sys.argv[1], 'r'):
    record = json.loads(line.strip())
    print(f"<|input|>"+record['prompt']+"<|response|>"+record['response'])