#!/bin/bash
# run-cmd.sh - Execute a command in Plan 9 via QEMU
#
# Usage: ./run-cmd.sh "command to execute"
#
# Returns the output of the command

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISK_IMAGE="$SCRIPT_DIR/9front.qcow2"
SHARED_IMG="$SCRIPT_DIR/shared.img"
SHARED_DIR="$SCRIPT_DIR/shared"

# Source OS helper functions
source "$SCRIPT_DIR/os-helper.sh"

# Get platform-specific QEMU settings
QEMU="$(get_qemu_path)"
ACCEL="$(get_qemu_accel)"

if [ -z "$1" ]; then
    echo "Usage: $0 \"command to execute\""
    exit 1
fi

CMD="$1"

# Check prerequisites
if ! command -v "$QEMU" &> /dev/null; then
    echo "Error: QEMU not found at $QEMU"
    exit 1
fi

if [ ! -f "$DISK_IMAGE" ]; then
    echo "Error: Disk image not found at $DISK_IMAGE"
    exit 1
fi

# Ensure shared directory exists
mkdir -p "$SHARED_DIR"

# Create expect script on the fly
EXPECT_SCRIPT=$(mktemp /tmp/run-cmd.XXXXXX.exp)
cat > "$EXPECT_SCRIPT" << EXPECT_EOF
#!/usr/bin/expect -f

set timeout 180
set disk_image [lindex \$argv 0]
set shared_img [lindex \$argv 1]
set cmd [lindex \$argv 2]
set qemu [lindex \$argv 3]
set accel [lindex \$argv 4]

log_user 0

spawn \$qemu \\
    -m 512 \\
    -cpu max \\
    -accel \$accel \\
    -drive file=\$disk_image,format=qcow2,if=virtio \\
    -drive file=\$shared_img,format=raw,if=virtio \\
    -display none \\
    -serial mon:stdio

# Wait for boot prompt and accept default
expect {
    -re "bootargs.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: waiting for bootargs"
        exit 1
    }
}

# Wait for user prompt
expect {
    -re "user\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: waiting for user prompt"
        exit 1
    }
}

# Wait for rc shell prompt
expect {
    -re "term%" {
        # At shell
    }
    -re ";" {
        # At shell (different prompt)
    }
    timeout {
        puts "TIMEOUT: waiting for shell"
        exit 1
    }
}

# Small delay to ensure shell is ready
sleep 1

# Mount the FAT shared disk at /mnt/host
# Use /dev/sdG0/data for raw FAT images (no partition table)
send "dossrv -f /dev/sdG0/data shared\r"
expect {
    -re "term%" { }
    -re ";" { }
    timeout { }
}
sleep 1
send "mount -c /srv/shared /mnt/host\r"
expect {
    -re "term%" { }
    -re ";" { }
    timeout { }
}
sleep 1

# Execute the command
send "\$cmd\r"

# Capture output until next prompt
log_user 1
expect {
    -re "term%" {
        # Command completed
    }
    -re "\n;" {
        # Command completed (different prompt)
    }
    timeout {
        puts "TIMEOUT: waiting for command completion"
    }
}
log_user 0

# Shutdown
send "fshalt\r"
sleep 2
send "\001x"
sleep 1

exit 0
EXPECT_EOF

chmod +x "$EXPECT_SCRIPT"

# Run the expect script and filter output
# Escape special characters in CMD for sed
CMD_ESCAPED=$(printf '%s\n' "$CMD" | sed 's/[[\.*^$()+?{|]/\\&/g')
OUTPUT=$("$EXPECT_SCRIPT" "$DISK_IMAGE" "$SHARED_IMG" "$CMD" "$QEMU" "$ACCEL" 2>/dev/null | \
    grep -v "^spawn " | \
    grep -v "^ $CMD_ESCAPED$" | \
    sed '/^[[:space:]]*$/d')

# Print output (remove prompt line at end)
echo "$OUTPUT" | grep -v "^term%" | grep -v "^;$"

# Cleanup
rm -f "$EXPECT_SCRIPT"

# Kill any remaining QEMU processes from this script
pkill -f "qemu-system.*$DISK_IMAGE" 2>/dev/null

exit 0
