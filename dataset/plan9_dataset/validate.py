"""Validate dataset examples in Plan 9 QEMU VM."""

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .export import Example, load_examples


# Paths relative to 9ml repo root
QEMU_DIR = "qemu"
DISK_IMAGE = "9front.qcow2"
SHARED_IMAGE = "shared.img"

# Validation timeout
DEFAULT_TIMEOUT = 30


@dataclass
class ValidationResult:
    """Result of validating a single example."""

    example_index: int
    instruction: str
    category: str
    code_type: str  # "c", "rc", "asm", "unknown"
    valid: bool
    error: str = ""
    output: str = ""


@dataclass
class ValidationReport:
    """Report of validation results."""

    total: int = 0
    valid: int = 0
    invalid: int = 0
    skipped: int = 0
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        self.total += 1
        if result.valid:
            self.valid += 1
        elif result.error == "skipped":
            self.skipped += 1
        else:
            self.invalid += 1


def extract_code(response: str) -> str:
    """Extract just the code portion from a response, removing prose text.

    Many responses have explanatory text after the code that should not be compiled.
    """
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Detect start of code
        if not in_code:
            # Code starts with shebang, fn, #include, TEXT, or mkfile include
            if (stripped.startswith("#!") or stripped.startswith("fn ") or
                stripped.startswith("#include") or stripped.startswith("TEXT ") or
                stripped.startswith("</$objtype")):
                in_code = True
                code_lines.append(line)
                blank_count = 0
                continue
            # Also start if we see C function signatures
            if re.match(r"^(void|int|char|long|ulong|static)\s*$", stripped) or \
               re.match(r"^(void|int|char|long|ulong|static)\s+\w+\s*\(", stripped):
                in_code = True
                code_lines.append(line)
                blank_count = 0
                continue
        else:
            # Track blank lines
            if stripped == "":
                blank_count += 1
                if blank_count <= 1:
                    code_lines.append(line)
                continue

            # Stop at prose markers (lines starting with capital letters or common prose patterns)
            if blank_count >= 2:
                # Check if this looks like prose
                if re.match(r"^[A-Z][a-z]+[:\.]", stripped) or \
                   stripped.startswith("Usage:") or \
                   stripped.startswith("Note:") or \
                   stripped.startswith("This ") or \
                   stripped.startswith("The ") or \
                   re.match(r"^-\s+", stripped):  # Bullet point
                    break

            blank_count = 0
            code_lines.append(line)

    return "\n".join(code_lines).rstrip()


def detect_code_type(response: str) -> str:
    """Detect the type of code in a response."""
    response = response.strip()

    # Check for rc script
    if response.startswith("#!/bin/rc") or response.startswith("fn "):
        return "rc"

    # Check for Plan 9 C code
    if "#include <u.h>" in response or "#include <libc.h>" in response:
        return "c"

    # Check for function-like C patterns
    if re.search(r"\bvoid\s+\w+\s*\(", response) or re.search(r"\bint\s+\w+\s*\(", response):
        return "c"

    # Check for assembly
    if response.startswith("TEXT ") or "MOVQ" in response or "ADDPS" in response:
        return "asm"

    # Check for mkfile
    if "</$objtype/mkfile" in response or "OFILES=" in response:
        return "mkfile"

    return "unknown"


class FATDisk:
    """FAT disk operations using mtools."""

    def __init__(self, path: str):
        self.path = path

    def create(self, size_mb: int = 64) -> bool:
        """Create a FAT32 disk image."""
        try:
            # Remove existing
            if os.path.exists(self.path):
                os.unlink(self.path)

            # Create sparse file
            subprocess.run(
                ["truncate", "-s", f"{size_mb}M", self.path],
                check=True,
                capture_output=True,
            )

            # Zero first MB
            subprocess.run(
                ["dd", "if=/dev/zero", f"of={self.path}", "bs=1M", "count=1", "conv=notrunc"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Format as FAT32
            subprocess.run(
                ["mformat", "-i", self.path, "-F", "::"],
                check=True,
                capture_output=True,
            )

            return True
        except subprocess.CalledProcessError:
            return False

    def copy_to(self, src_path: str, dest_name: str) -> bool:
        """Copy a file to the disk."""
        try:
            subprocess.run(
                ["mcopy", "-i", self.path, src_path, f"::{dest_name}"],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def copy_content(self, content: str, dest_name: str, retries: int = 3) -> bool:
        """Write content directly to a file on the disk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            for attempt in range(retries):
                # Delete existing file first
                self.delete(dest_name)
                time.sleep(0.1)  # Small delay to let disk settle

                if self.copy_to(temp_path, dest_name):
                    return True
                time.sleep(0.5)  # Wait before retry

            return False
        finally:
            os.unlink(temp_path)

    def read_file(self, filename: str) -> str | None:
        """Read a file from the disk."""
        try:
            result = subprocess.run(
                ["mcopy", "-i", self.path, f"::{filename}", "-"],
                capture_output=True,
            )
            if result.returncode == 0:
                return result.stdout.decode("utf-8", errors="replace")
            return None
        except subprocess.CalledProcessError:
            return None

    def delete(self, filename: str) -> bool:
        """Delete a file from the disk."""
        try:
            subprocess.run(
                ["mdel", "-i", self.path, f"::{filename}"],
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return True  # File may not exist


class Plan9VM:
    """Plan 9 QEMU VM for validation using pexpect."""

    def __init__(self, disk_image: str, shared_image: str, debug: bool = False):
        self.disk_image = disk_image
        self.shared_image = shared_image
        self.debug = debug
        self.child = None
        self.booted = False

    def start(self) -> bool:
        """Start the VM using pexpect."""
        import pexpect

        disk_arg = f"file={self.disk_image},format=qcow2,if=virtio"
        shared_arg = f"file={self.shared_image},format=raw,if=virtio"

        cmd = (
            f"qemu-system-x86_64 -m 512 -smp 4 -cpu host -accel kvm "
            f"-drive {disk_arg} -drive {shared_arg} "
            f"-display none -serial mon:stdio"
        )

        self.child = pexpect.spawn(cmd, encoding="utf-8", timeout=120)
        if self.debug:
            self.child.logfile = open("/tmp/qemu_debug.log", "w")

        return self.child is not None

    def expect(self, pattern: str, timeout: int = 30) -> bool:
        """Wait for pattern in output."""
        import pexpect

        if not self.child:
            return False
        try:
            self.child.expect(pattern, timeout=timeout)
            return True
        except pexpect.TIMEOUT:
            return False
        except pexpect.EOF:
            return False

    def sendln(self, text: str) -> None:
        """Send text with newline."""
        if self.child:
            self.child.sendline(text)

    def send(self, text: str) -> None:
        """Send text."""
        if self.child:
            self.child.send(text)

    def run_command(self, cmd: str, timeout: int = 30) -> str:
        """Run a command and wait for prompt."""
        import pexpect

        if not self.child:
            return ""

        self.sendln(cmd)
        try:
            self.child.expect("term%", timeout=timeout)
            return self.child.before
        except (pexpect.TIMEOUT, pexpect.EOF):
            return ""

    def clear_output(self) -> None:
        """Clear any pending output from the terminal buffer."""
        import pexpect

        if not self.child:
            return
        # Send a simple command and wait for prompt to sync
        self.sendln("echo SYNC")
        try:
            self.child.expect("SYNC", timeout=5)
            self.child.expect("term%", timeout=5)
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

    def boot(self) -> bool:
        """Boot the VM and get to shell prompt."""
        if self.booted:
            return True

        if not self.start():
            return False

        # Wait for bootargs prompt
        if not self.expect("bootargs", timeout=60):
            return False
        self.sendln("")

        # Wait for user prompt
        if not self.expect("user", timeout=60):
            return False
        self.sendln("")

        # Wait for shell
        if not self.expect("term%", timeout=60):
            return False

        # Mount shared disk
        self.sendln("dossrv -f /dev/sdG0/data shared")
        time.sleep(2)
        self.sendln("mount -c /srv/shared /mnt/host")
        time.sleep(1)
        self.sendln("cd /mnt/host")

        if not self.expect("term%", timeout=10):
            return False

        self.booted = True
        return True

    def shutdown(self) -> None:
        """Shutdown the VM."""
        if self.child:
            self.child.close(force=True)
            self.child = None
        self.booted = False


def make_compilable_c(code: str) -> str:
    """Make a C code snippet compilable by adding necessary boilerplate."""
    # If it's already a complete program, return as-is
    if "void\nmain" in code or "void main" in code or "int\nmain" in code:
        return code

    # If it already has headers, just return
    if "#include <u.h>" in code:
        return code

    # Add standard headers
    headers = "#include <u.h>\n#include <libc.h>\n\n"

    # Check if it's just a function definition
    if re.match(r"^\s*(void|int|char|long|ulong)\s*\n?\w+\s*\(", code):
        return headers + code

    # Wrap in a dummy main
    return headers + "void\nmain(int argc, char **argv)\n{\n" + code + "\n\texits(nil);\n}\n"


def validate_c_code(code: str, disk: FATDisk, vm: Plan9VM) -> tuple[bool, str]:
    """Validate C code by compiling in Plan 9.

    Returns (valid, error_message).
    """
    # Make code compilable
    full_code = make_compilable_c(code)

    # Write code to disk
    if not disk.copy_content(full_code, "test_validate.c"):
        return False, "Failed to write code to disk"

    # Compile in VM with unique marker
    marker = f"DONE_{int(time.time())}"
    output = vm.run_command(f"6c -w test_validate.c >[2=1]; echo {marker}", timeout=30)

    # Check for errors in output
    if output:
        # Look for common error patterns
        if "error" in output.lower() or "undefined" in output.lower() or "syntax" in output.lower():
            # Extract error message
            lines = output.split("\n")
            errors = [l for l in lines if "error" in l.lower() or "undefined" in l.lower()]
            return False, "\n".join(errors[:3]) if errors else "Compilation error"

    # Clean up
    vm.run_command("rm -f test_validate.c test_validate.6", timeout=5)

    return True, ""


def validate_rc_code(code: str, disk: FATDisk, vm: Plan9VM) -> tuple[bool, str]:
    """Validate rc script syntax in Plan 9.

    Returns (valid, error_message).
    """
    # Write code to disk
    if not disk.copy_content(code, "test_validate.rc"):
        return False, "Failed to write code to disk"

    # Syntax check with rc -n (parse only, don't execute)
    marker = f"DONE_{int(time.time())}"
    output = vm.run_command(f"rc -n test_validate.rc >[2=1]; echo {marker}", timeout=10)

    # Check for errors
    if output:
        if "error" in output.lower() or "syntax" in output.lower():
            lines = output.split("\n")
            errors = [l for l in lines if "error" in l.lower() or "syntax" in l.lower()]
            return False, "\n".join(errors[:3]) if errors else "Syntax error"

    # Clean up
    vm.run_command("rm -f test_validate.rc", timeout=5)

    return True, ""


def validate_example(
    example: Example, index: int, disk: FATDisk, vm: Plan9VM
) -> ValidationResult:
    """Validate a single example."""
    response = example.response

    # Detect code type
    code_type = detect_code_type(response)

    # Extract just the code portion (remove prose text)
    code = extract_code(response)
    if not code:
        code = response  # Fallback to full response if extraction failed

    # Clear any pending output before validation
    vm.clear_output()

    if code_type == "unknown":
        return ValidationResult(
            example_index=index,
            instruction=example.instruction[:50] + "...",
            category=example.category,
            code_type=code_type,
            valid=True,  # Can't validate, assume OK
            error="skipped",
            output="No code detected",
        )

    # Validate based on type
    if code_type == "c":
        valid, error = validate_c_code(code, disk, vm)
    elif code_type == "rc":
        valid, error = validate_rc_code(code, disk, vm)
    else:
        # mkfile, asm - skip for now
        return ValidationResult(
            example_index=index,
            instruction=example.instruction[:50] + "...",
            category=example.category,
            code_type=code_type,
            valid=True,
            error="skipped",
            output=f"Validation not implemented for {code_type}",
        )

    return ValidationResult(
        example_index=index,
        instruction=example.instruction[:50] + "...",
        category=example.category,
        code_type=code_type,
        valid=valid,
        error=error,
    )


def validate_examples(
    examples_path: Path, qemu_dir: Path, categories: list[str] | None = None, debug: bool = False
) -> ValidationReport:
    """Validate all examples from the dataset.

    Args:
        examples_path: Path to examples.json
        qemu_dir: Path to qemu directory with disk images
        categories: Optional list of categories to validate (None = all)
        debug: Enable debug logging

    Returns:
        ValidationReport with results
    """
    report = ValidationReport()

    # Load examples
    examples = load_examples(examples_path)
    if not examples:
        return report

    # Set up paths
    disk_image = qemu_dir / DISK_IMAGE
    shared_image = qemu_dir / SHARED_IMAGE

    if not disk_image.exists():
        raise FileNotFoundError(f"Disk image not found: {disk_image}")

    # Create fresh shared disk
    disk = FATDisk(str(shared_image))
    if not disk.create(64):
        raise RuntimeError("Failed to create shared disk")

    # Start VM
    vm = Plan9VM(str(disk_image), str(shared_image), debug=debug)
    try:
        print("Booting Plan 9 VM...")
        if not vm.boot():
            raise RuntimeError("Failed to boot VM")
        print("VM booted successfully")

        # Validate each example
        for i, example in enumerate(examples):
            # Skip if category filter specified
            if categories and example.category not in categories:
                continue

            print(f"Validating example {i}: {example.category}...")
            result = validate_example(example, i, disk, vm)
            report.add(result)

            if not result.valid and result.error != "skipped":
                print(f"  INVALID: {result.error[:60]}")
            else:
                print(f"  OK")

    finally:
        print("Shutting down VM...")
        vm.shutdown()

    return report


def quick_validate_syntax(examples_path: Path) -> ValidationReport:
    """Quick syntax validation without QEMU (local checks only).

    Validates:
    - C code has matching braces
    - rc scripts have valid structure
    - No obvious syntax errors
    """
    report = ValidationReport()

    examples = load_examples(examples_path)
    if not examples:
        return report

    for i, example in enumerate(examples):
        response = example.response
        code_type = detect_code_type(response)

        errors = []

        if code_type == "c":
            # Check brace matching
            open_braces = response.count("{")
            close_braces = response.count("}")
            if open_braces != close_braces:
                errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")

            # Check parenthesis matching
            open_parens = response.count("(")
            close_parens = response.count(")")
            if open_parens != close_parens:
                errors.append(f"Mismatched parens: {open_parens} open, {close_parens} close")

            # Check for common issues
            if "NULL" in response and "nil" not in response:
                errors.append("Uses NULL instead of nil")

        elif code_type == "rc":
            # Check brace matching for rc
            open_braces = response.count("{")
            close_braces = response.count("}")
            if open_braces != close_braces:
                errors.append(f"Mismatched braces: {open_braces} open, {close_braces} close")

            # Check parenthesis in if/for/while
            open_parens = response.count("(")
            close_parens = response.count(")")
            if open_parens != close_parens:
                errors.append(f"Mismatched parens: {open_parens} open, {close_parens} close")

        valid = len(errors) == 0
        error_str = "; ".join(errors) if errors else ""
        if code_type == "unknown" and not errors:
            error_str = "skipped"

        report.add(
            ValidationResult(
                example_index=i,
                instruction=example.instruction[:50] + "...",
                category=example.category,
                code_type=code_type,
                valid=valid if code_type != "unknown" else True,
                error=error_str,
            )
        )

    return report
