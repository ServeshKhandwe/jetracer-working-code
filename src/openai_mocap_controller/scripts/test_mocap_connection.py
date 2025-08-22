#!/usr/bin/env python3

import asyncio
import websockets
import json
import sys

async def test_mocap_connection(server_ip="192.168.64.147", port=8765):
    """Test connection to mocap server and display received data"""
    uri = f"ws://{server_ip}:{port}"
    print(f"Testing connection to mocap server at {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            print("Receiving data (press Ctrl+C to stop):")
            print("-" * 50)
            
            count = 0
            while count < 10:  # Show first 10 messages
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if "objects" in data and "Player" in data["objects"]:
                        player = data["objects"]["Player"]
                        print(f"Frame {count + 1}:")
                        print(f"  Position: ({player['x']:.1f}, {player['y']:.1f}, {player['z']:.1f}) mm")
                        print(f"  Quaternion: ({player['qx']:.3f}, {player['qy']:.3f}, {player['qz']:.3f}, {player['qw']:.3f})")
                        print(f"  Timestamp: {data['timestamp']}")
                        print()
                        
                    count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
                except KeyboardInterrupt:
                    break
                    
            print("✓ Mocap connection test completed successfully!")
            
    except websockets.exceptions.ConnectionRefused:
        print("✗ Connection refused. Is the mocap server running?")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False
        
    return True

def main():
    server_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.64.147"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    print("OpenAI Mocap Controller - Connection Test")
    print("=" * 50)
    
    try:
        asyncio.run(test_mocap_connection(server_ip, port))
    except KeyboardInterrupt:
        print("\nTest interrupted by user")

if __name__ == '__main__':
    main()