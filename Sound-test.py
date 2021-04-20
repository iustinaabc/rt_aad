import os
from audio import AudioPlayer, LRBalancer

# Volume parameters
volumeThreshold = 50  # in percentage
volLeft = 100  # Starting volume in percentage
volRight = 100  # Starting volume in percentage

""" SET-UP Headphones """
device_name = "sysdefault"
control_name = "Headphone Jack Sense"
# control_name = 'Master'
cardindex = 0

wav_fn = os.path.join(os.path.expanduser('~/Music'), 'Creep.wav')

# Playback
ap = AudioPlayer()

print(ap.list_devices())

ap.set_device(device_name, cardindex)
ap.init_play(wav_fn)
ap.play()

# Audio Control
lr_bal = LRBalancer()
lr_bal.set_control(control_name, device_name, cardindex)

lr_bal.set_volume_left(volLeft)
lr_bal.set_volume_right(volRight)
