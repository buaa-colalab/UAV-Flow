from pynput import keyboard

key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    'ctrl':False,
    '1': False,
    '2': False,
    'head_up': False,
    'head_down': False,
    'e': False
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True
        if key ==keyboard.Key.ctrl_l:
            key_state['ctrl'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
        if key ==keyboard.Key.ctrl_l:
            key_state['ctrl'] = False
def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['i']:
        action[0][1] = 200
    if key_state['k']:
        action[0][1] = -200
    if key_state['j']:
        action[0][0] = -30
    if key_state['l']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['ctrl']:
        action[2] = 2
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2
    if key_state['e']:
        action[2] = 5

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action
