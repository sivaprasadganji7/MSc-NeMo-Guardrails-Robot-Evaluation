from furhat_remote_api import FurhatRemoteAPI
import time

ROBOT_IP = "127.0.0.1"   # Your robot's IP
furhat = FurhatRemoteAPI(ROBOT_IP)

print("Listening... (will block until speech is detected)")
start = time.time()
result = furhat.listen()
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.2f} seconds")
print(f"Result object: {result}")
if result:
    print(f"Success: {result.success}, Message: '{result.message}'")