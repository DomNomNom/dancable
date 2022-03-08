import sys


import asyncio
import websockets
import subprocess

# from dancable import dancable

PORT = 8444

# create handler for each connection
async def handler(websocket, path):
    url = await websocket.recv()

    # Sanitizing
    prefix = 'https://www.youtube.com/playlist?'
    if not url.startswith(prefix):
        await websocket.send(f'Error: URL must start with {prefix!r}')
        return

    with subprocess.Popen([sys.executable, '-u', 'dancable.py', url],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          bufsize=1,
                          universal_newlines=True) as process:
        for line in process.stdout:
            line = line.rstrip()
            print(line)
            await websocket.send(line)

start_server = websockets.serve(handler, "localhost", PORT)

asyncio.get_event_loop().run_until_complete(start_server)
print(f'serving on port {PORT}...')
asyncio.get_event_loop().run_forever()
