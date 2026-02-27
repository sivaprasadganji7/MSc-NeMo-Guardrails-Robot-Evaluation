import time
from furhat_remote_api import FurhatRemoteAPI

# Replace with your robot's IP
ROBOT_IP = "127.0.0.1"  # Example, change to your robot's IP

# Initialize connection
furhat = FurhatRemoteAPI(ROBOT_IP)

# Test speaking
print("Saying hello...")
furhat.say(text="Hello! I am your movie bot. Connection successful.")
time.sleep(2)
furhat.say(text="Let's start building the movie expert system.")

print("Test complete. If you heard the robot, it's working!")