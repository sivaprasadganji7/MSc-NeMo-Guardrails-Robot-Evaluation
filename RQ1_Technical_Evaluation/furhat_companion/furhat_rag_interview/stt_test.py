from furhat_remote_api import FurhatRemoteAPI

furhat = FurhatRemoteAPI("127.0.0.1")  # add port if needed

furhat.say(text="Microphone test. Please say: hello Furhat.")
res = furhat.listen()
print("RAW RESPONSE:", res)
print("MESSAGE:", getattr(res, "message", None))
furhat.say(text=f"I heard: {getattr(res, 'message', '')}")
