#!/bin/bash
# os-helper.sh - OS detection and platform-specific functions
#
# Source this file in other scripts:
#   source "$(dirname "$0")/os-helper.sh"

detect_os() {
    case "$(uname -s)" in
        Darwin) echo "macos" ;;
        Linux)  echo "linux" ;;
        *)      echo "unknown" ;;
    esac
}

get_qemu_path() {
    case "$(detect_os)" in
        macos)
            if [ -x "/opt/local/bin/qemu-system-x86_64" ]; then
                echo "/opt/local/bin/qemu-system-x86_64"
            elif [ -x "/usr/local/bin/qemu-system-x86_64" ]; then
                echo "/usr/local/bin/qemu-system-x86_64"
            else
                echo "qemu-system-x86_64"
            fi
            ;;
        linux)
            echo "qemu-system-x86_64"
            ;;
        *)
            echo "qemu-system-x86_64"
            ;;
    esac
}

get_qemu_accel() {
    case "$(detect_os)" in
        macos)
            # Check if HVF is available
            if sysctl -n kern.hv_support 2>/dev/null | grep -q 1; then
                echo "hvf"
            else
                echo "tcg"
            fi
            ;;
        linux)
            # Check if KVM is available
            if [ -e /dev/kvm ] && [ -r /dev/kvm ] && [ -w /dev/kvm ]; then
                echo "kvm"
            else
                echo "tcg"
            fi
            ;;
        *)
            echo "tcg"
            ;;
    esac
}

mount_fat_image() {
    local img="$1"
    local mount_point="$2"

    case "$(detect_os)" in
        macos)
            hdiutil attach -imagekey diskimage-class=CRawDiskImage \
                -mountpoint "$mount_point" "$img" > /dev/null
            ;;
        linux)
            sudo mount -o loop,uid=$(id -u),gid=$(id -g) "$img" "$mount_point"
            ;;
        *)
            echo "Error: Unsupported OS for mounting" >&2
            return 1
            ;;
    esac
}

unmount_fat_image() {
    local mount_point="$1"

    case "$(detect_os)" in
        macos)
            hdiutil detach "$mount_point" > /dev/null
            ;;
        linux)
            sudo umount "$mount_point"
            ;;
        *)
            echo "Error: Unsupported OS for unmounting" >&2
            return 1
            ;;
    esac
}

create_fat_image() {
    local img="$1"
    local size_mb="${2:-128}"

    case "$(detect_os)" in
        macos)
            hdiutil create -size "${size_mb}m" -fs MS-DOS -volname SHARED "${img%.img}"
            mv "${img%.img}.dmg" "$img"
            ;;
        linux)
            dd if=/dev/zero of="$img" bs=1M count="$size_mb" 2>/dev/null
            mkfs.vfat -n SHARED "$img" > /dev/null
            ;;
        *)
            echo "Error: Unsupported OS for creating FAT image" >&2
            return 1
            ;;
    esac
}
