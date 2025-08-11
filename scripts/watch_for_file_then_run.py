#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description='Watch for a file to appear (and stabilize), then run a command once.')
    ap.add_argument('--path', required=True, help='Path to watch for (file)')
    ap.add_argument('--command', nargs='+', required=True, help='Command to execute when file is ready')
    ap.add_argument('--interval', type=float, default=10.0, help='Polling interval (seconds)')
    ap.add_argument('--stable-seconds', type=float, default=60.0, help='Require mtime unchanged for this many seconds')
    ap.add_argument('--timeout', type=float, default=0.0, help='Optional timeout in seconds (0 = no timeout)')
    args = ap.parse_args()

    target = Path(args.path)
    start = time.time()
    last_mtime = None
    stable_since = None

    print(f'Watching for {target} ...')
    while True:
        now = time.time()
        if args.timeout and (now - start) > args.timeout:
            print('Timeout expired; exiting with code 2')
            sys.exit(2)
        if target.exists():
            mtime = target.stat().st_mtime
            if last_mtime is None or mtime != last_mtime:
                last_mtime = mtime
                stable_since = now
            elif (now - stable_since) >= args.stable_seconds:
                # File is stable; run command
                print(f'File detected and stable; running: {" ".join(args.command)}')
                proc = subprocess.run(args.command)
                sys.exit(proc.returncode)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
