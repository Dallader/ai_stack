#!/usr/bin/env python3
import json
import subprocess
import os

# Path to the host_settings.json
settings_path = os.path.join(os.path.dirname(__file__), 'agents', 'agent2_ticket', 'host_settings.json')

# Read the settings
with open(settings_path, 'r') as f:
    settings = json.load(f)

# Check the start_necessary flag
if settings.get('start_necessary', False):
    # Stop all services first, then start only necessary services
    subprocess.run(['docker-compose', 'down'])
    cmd = ['docker-compose', '--profile', 'necessary', 'up', '-d']
else:
    # Start all services
    cmd = ['docker-compose', 'up', '-d']

# Run the command
subprocess.run(cmd)