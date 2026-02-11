
import os
import sys
import asyncio
import argparse
from telegram import Bot

def daemonize():
    """Fork the process and detach from the terminal."""
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit first parent
    except OSError as e:
        sys.stderr.write(f"fork #1 failed: {e.errno} ({e.strerror})\n")
        sys.exit(1)

    os.setsid() # Decouple from parent environment

    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit second parent
    except OSError as e:
        sys.stderr.write(f"fork #2 failed: {e.errno} ({e.strerror})\n")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, 'r') as si:
        os.dup2(si.fileno(), sys.stdin.fileno())
    with open(os.devnull, 'a+') as so:
        os.dup2(so.fileno(), sys.stdout.fileno())
    with open(os.devnull, 'a+') as se:
        os.dup2(se.fileno(), sys.stderr.fileno())


async def send_to_telegram(bot, chat_id, message, topic_id=None):
    try:
        if topic_id:
            await bot.send_message(chat_id=chat_id, text=message, message_thread_id=topic_id)
        else:
            await bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

async def watch_logs(token, chat_id, log_file_path, interval, topic_id=None):
    bot = Bot(token=token)

    # Ensure the file exists before starting
    if not os.path.exists(log_file_path):
        print(f"Warning: {log_file_path} not found. Waiting for it to be created...")
        while not os.path.exists(log_file_path):
            await asyncio.sleep(1)

    with open(log_file_path, 'r') as f:
        # Move to the end of the file to ignore old logs
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                await send_to_telegram(bot, chat_id, 'Health Ping Noti', topic_id)
            else:
                clean_line = line.strip()
                if clean_line:
                    # You can filter logs here (e.g., only send if "Reward" is in line)
                    await send_to_telegram(bot, chat_id, clean_line, topic_id)
            
            await asyncio.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Harvest logs and send them to Telegram.")
    
    # Flags
    parser.add_argument("-t", "--token", required=True, help="Telegram Bot Token")
    parser.add_argument("-c", "--chat_id", required=True, help="Telegram Chat ID")
    parser.add_argument("-f", "--file", required=True, help="Path to the log file to watch")
    parser.add_argument("-i", "--interval", type=float, default=5, help="Check interval in seconds (default: 300)")
    parser.add_argument("-T", "--topic_id", type=int, help="Telegram Topic ID (optional, for forums)")

    args = parser.parse_args()

    # Convert to absolute path to avoid issues if daemon changes cwd (though we skipped chdir /)
    abs_log_path = os.path.abspath(args.file)

    print("--- Telegram Harvester Started ---")
    print(f"Target File: {abs_log_path}")
    print(f"Chat ID:     {args.chat_id}")
    if args.topic_id:
        print(f"Topic ID:    {args.topic_id}")
    print("Running in background...")

    daemonize()

    try:
        asyncio.run(watch_logs(args.token, args.chat_id, abs_log_path, args.interval, args.topic_id))
    except KeyboardInterrupt:
        print("\nStopping harvester...")

if __name__ == "__main__":
    main()