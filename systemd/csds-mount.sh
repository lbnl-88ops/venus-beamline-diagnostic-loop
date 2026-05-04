#!/bin/bash
# SSHFS mount script for CSDS data
# This script checks if the mount point is active and attempts to mount if not.

MOUNT_POINT="/data/csds"
REMOTE="rehak@ecris-beamline.dhcp.lbl.gov:/data/csds"

# Options:
# allow_other: allow other users to access the mount
# reconnect: handle connection drops
# ServerAlive*: keep-alive packets
# idmap=user: map remote user to local user
OPTS="allow_other,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3"

echo "Starting CSDS mount monitor..."

while true; do
    if ! mountpoint -q "$MOUNT_POINT"; then
        echo "$(date): Mount point $MOUNT_POINT is not mounted. Attempting to mount..."
        # Note: Running as rehak, using local SSH keys
        sshfs "$REMOTE" "$MOUNT_POINT" -o $OPTS
        if [ $? -eq 0 ]; then
            echo "$(date): Successfully mounted $MOUNT_POINT"
        else
            echo "$(date): Failed to mount $MOUNT_POINT. Retrying in 60 seconds."
        fi
    fi
    sleep 60
done
