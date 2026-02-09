import time
import os
import asyncio
import argparse
from telegram import Bot

async def send_to_telegram(bot, chat_id, message):
    try:
        await bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

async def watch_logs(token, chat_id, log_file_path, interval):
    bot = Bot(token=token)
    print("--- Telegram Harvester Started ---")
    print(f"Target File: {log_file_path}")
    print(f"Chat ID:     {chat_id}")
    
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
                await asyncio.sleep(interval)
                continue
            
            clean_line = line.strip()
            if clean_line:
                # You can filter logs here (e.g., only send if "Reward" is in line)
                await send_to_telegram(bot, chat_id, clean_line)

def main():
    parser = argparse.ArgumentParser(description="Harvest logs and send them to Telegram.")
    
    # Flags
    parser.add_argument("-t", "--token", required=True, help="Telegram Bot Token")
    parser.add_argument("-c", "--chat_id", required=True, help="Telegram Chat ID")
    parser.add_argument("-f", "--file", required=True, help="Path to the log file to watch")
    parser.add_argument("-i", "--interval", type=float, default=0.5, help="Check interval in seconds (default: 0.5)")

    args = parser.parse_args()

    try:
        asyncio.run(watch_logs(args.token, args.chat_id, args.file, args.interval))
    except KeyboardInterrupt:
        print("\nStopping harvester...")

if __name__ == "__main__":
    main()