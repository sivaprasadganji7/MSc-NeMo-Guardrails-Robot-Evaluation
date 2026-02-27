import swagger_client

class FurhatRemoteAPI(swagger_client.FurhatApi):
    def __init__(self, host):
        configuration = swagger_client.Configuration()
        configuration.host = "http://" + host + ":54321"
        super().__init__(swagger_client.ApiClient(configuration))
    attend = swagger_client.FurhatApi.furhat_attend_post
    say = swagger_client.FurhatApi.furhat_say_post
    say_stop = swagger_client.FurhatApi.furhat_say_stop_post
    set_face = swagger_client.FurhatApi.furhat_face_post
    set_visibility = swagger_client.FurhatApi.furhat_visibility_post
    set_voice = swagger_client.FurhatApi.furhat_voice_post
    get_voices = swagger_client.FurhatApi.furhat_voices_get
    gesture = swagger_client.FurhatApi.furhat_gesture_post
    get_gestures = swagger_client.FurhatApi.furhat_gestures_get
    set_led = swagger_client.FurhatApi.furhat_led_post
    listen = swagger_client.FurhatApi.furhat_listen_get
    listen_stop = swagger_client.FurhatApi.furhat_listen_stop_post
    get_users = swagger_client.FurhatApi.furhat_users_get

from swagger_client.models.basic_param import BasicParam
from swagger_client.models.frame import Frame
from swagger_client.models.gesture import Gesture
from swagger_client.models.gesture_definition import GestureDefinition
from swagger_client.models.location import Location
from swagger_client.models.rotation import Rotation
from swagger_client.models.status import Status
from swagger_client.models.user import User
from swagger_client.models.voice import Voice
