#!/usr/bin/env python3
with open('.env', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('GITHUB_TOKEN='):
            token = line.split('=', 1)[1].strip().strip('"""')
            print('.env file Token:')
            print('  First 20 chars:', token[:20])
            print('  Starts with ghp_:', token.startswith('ghp_'))
            print('  Starts with github_pat_:', token.startswith('github_pat_'))
            print('  Total length:', len(token))
            break
