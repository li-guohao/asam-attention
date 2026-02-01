#!/usr/bin/env python3
import os
token = os.getenv('GITHUB_TOKEN')
if token:
    print('Token found in environment variable')
    print('First 20 chars:', token[:20])
    print('Starts with ghp_:', token.startswith('ghp_'))
    print('Starts with github_pat_:', token.startswith('github_pat_'))
    print('Total length:', len(token))
else:
    print('Token NOT found in environment variable')
