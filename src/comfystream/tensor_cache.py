# TODO: add better frame management, improve eviction policy fifo might not be the best, skip alternate frames instead
# TODO: also make the tensor_cache solution backward compatible for when not using process pool -- after the multi process solution is stable
image_inputs = None
image_outputs = None

audio_inputs = None
audio_outputs = None


def init_tensor_cache(image_inputs_proxy, image_outputs_proxy, audio_inputs_proxy=None, audio_outputs_proxy=None):
    print("init_tensor_cache")
    global image_inputs, image_outputs, audio_inputs, audio_outputs
    image_inputs = image_inputs_proxy
    image_outputs = image_outputs_proxy
    audio_inputs = audio_inputs_proxy
    audio_outputs = audio_outputs_proxy
    