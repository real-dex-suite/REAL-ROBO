import asyncio
import websockets
import numpy as np

async def point_cloud_server(websocket, path):
    print("Client connected")
    try:
        while True:
            # 生成 1024x3 的随机点云
            points = np.random.rand(1024, 3).astype('float32').flatten()
            await websocket.send(points.tobytes())  # 发送二进制数据
            await asyncio.sleep(0.1)  # 每 0.1 秒发送一次数据
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

start_server = websockets.serve(point_cloud_server, "0.0.0.0", 8080)  # 监听所有接口

if __name__ == "__main__":
    print("Starting WebSocket server on ws://localhost:8080")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()