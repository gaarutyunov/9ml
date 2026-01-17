#!/bin/bash
# install-9front.sh - Download and install 9front into a QCOW2 image
#
# Usage: ./install-9front.sh
#
# Downloads the 9front ISO and installs it into 9front.qcow2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QCOW2_FILE="$SCRIPT_DIR/9front.qcow2"
ISO_URL="http://9front.org/iso/9front-11321.amd64.iso.gz"
ISO_FILE="$SCRIPT_DIR/9front.iso"

source "$SCRIPT_DIR/os-helper.sh"

QEMU="$(get_qemu_path)"
ACCEL="$(get_qemu_accel)"

if [ -f "$QCOW2_FILE" ]; then
    echo "9front.qcow2 already exists at $QCOW2_FILE"
    exit 0
fi

echo "Downloading 9front ISO..."
if [ ! -f "$ISO_FILE" ]; then
    curl -L -o "${ISO_FILE}.gz" "$ISO_URL"
    gunzip "${ISO_FILE}.gz"
fi

echo "Creating QCOW2 disk image..."
qemu-img create -f qcow2 "$QCOW2_FILE" 2G

echo "Installing 9front (this takes a few minutes)..."

# Create expect script for automated installation
EXPECT_SCRIPT=$(mktemp /tmp/install-9front.XXXXXX.exp)
cat > "$EXPECT_SCRIPT" << 'EXPECT_EOF'
#!/usr/bin/expect -f

set timeout 300
set qemu [lindex $argv 0]
set accel [lindex $argv 1]
set qcow2 [lindex $argv 2]
set iso [lindex $argv 3]

log_user 1

spawn $qemu \
    -m 1024 \
    -cpu max \
    -accel $accel \
    -drive file=$qcow2,format=qcow2,if=virtio \
    -cdrom $iso \
    -boot d \
    -display none \
    -serial mon:stdio

# Wait for bootargs prompt - accept default
expect {
    -re "bootargs.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: waiting for bootargs"
        exit 1
    }
}

# Wait for user prompt - accept default (glenda)
expect {
    -re "user\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: waiting for user"
        exit 1
    }
}

# Wait for shell prompt
expect {
    -re "term%" { }
    -re ";" { }
    timeout {
        puts "TIMEOUT: waiting for shell"
        exit 1
    }
}

sleep 2

# Start installer
send "inst/start\r"

# Configfs - use default (cwfs64x)
expect {
    -re "Configfs.*\\\[cwfs64x\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: configfs"
        exit 1
    }
}

# Partdisk - select sdF0 (virtio disk)
expect {
    -re "Partdisk.*\\\[.*\\\]" {
        send "sdF0\r"
    }
    timeout {
        puts "TIMEOUT: partdisk"
        exit 1
    }
}

# Install mbr - yes
expect {
    -re "Install mbr.*\\\[y\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: mbr"
        exit 1
    }
}

# Plan 9 partition - use whole disk
expect {
    -re "Plan 9 partition.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: plan9 partition"
        exit 1
    }
}

# Prep partition - accept default
expect {
    -re "Prep.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: prep"
        exit 1
    }
}

sleep 1

# Wait for partdisk done
expect {
    -re "done" { }
    -re "Prepdisk" { send "\r" }
    timeout { }
}

sleep 1

# Prepdisk - may not appear, handle both cases
expect {
    -re "Prepdisk.*\\\[.*\\\]" {
        send "\r"
    }
    -re "Mountfs" {
        # Already at mountfs
    }
    timeout { }
}

# Fmtfs - format the filesystem
expect {
    -re "Fmtfs.*\\\[.*\\\]" {
        send "\r"
    }
    -re "Mountfs" { }
    timeout { }
}

# Ream - yes to format
expect {
    -re "Ream.*\\\[yes\\\]" {
        send "\r"
    }
    -re "Mountfs" { }
    timeout { }
}

sleep 2

# Mountfs
expect {
    -re "Mountfs.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: mountfs"
        exit 1
    }
}

# Configdist - local
expect {
    -re "Configdist.*\\\[local\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: configdist"
        exit 1
    }
}

# Mountdist - /dev/sdG0/data (cdrom)
expect {
    -re "Mountdist.*\\\[.*\\\]" {
        send "/dev/sdG0/data\r"
    }
    timeout {
        puts "TIMEOUT: mountdist"
        exit 1
    }
}

# Copydist
expect {
    -re "Copydist.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: copydist"
        exit 1
    }
}

# Wait for copy to complete (can take a while)
expect {
    -re "Bootsetup" {
        # Copy done
    }
    timeout {
        puts "TIMEOUT: waiting for copy"
        exit 1
    }
}

# Bootsetup
expect {
    -re "Bootsetup.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: bootsetup"
        exit 1
    }
}

sleep 2

# Finish
expect {
    -re "Finish.*\\\[.*\\\]" {
        send "\r"
    }
    timeout {
        puts "TIMEOUT: finish"
        exit 1
    }
}

sleep 2

# Exit installer and halt
expect {
    -re "term%" {
        send "fshalt\r"
    }
    -re ";" {
        send "fshalt\r"
    }
    timeout { }
}

sleep 3

# Quit QEMU
send "\001x"
sleep 1

exit 0
EXPECT_EOF

chmod +x "$EXPECT_SCRIPT"

"$EXPECT_SCRIPT" "$QEMU" "$ACCEL" "$QCOW2_FILE" "$ISO_FILE"

rm -f "$EXPECT_SCRIPT"

echo ""
echo "9front installed successfully to $QCOW2_FILE"
ls -lh "$QCOW2_FILE"
