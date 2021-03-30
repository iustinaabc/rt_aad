    # TODO: these are the ALSA related sound settings, to be replaced with
    # by your own audio interface building block. Note that the volume
    # controls are also ALSA specific, and need to be changed
#    """ SET-UP Headphones """
#    device_name = 'sysdefault'
#    control_name = 'Headphone+LO'
#    cardindex = 0
#
#    wav_fn = os.path.join(os.path.expanduser('~/Desktop'), 'Pilot_1.wav')
#
#    # Playback
#    ap = AudioPlayer()
#
#    ap.set_device(device_name, cardindex)
#    ap.init_play(wav_fn)
#    ap.play()
#
#    # Audio Control
#    lr_bal = LRBalancer()
#    lr_bal.set_control(control_name, device_name, cardindex)
#
#    lr_bal.set_volume_left(volLeft)
#    lr_bal.set_volume_right(volRight)


# # Faded gain control towards left or right, stops when one channel falls below the volume threshold
# # Validation: previous decision is the same as this one
# print(lr_bal.get_volume())
# if all(np.array(lr_bal.get_volume()) > volumeThreshold) and previousLeftOrRight == leftOrRight:
#     print("---Controlling volume---")
#     if leftOrRight == -1.:
#         if volLeft != 100:
#             lr_bal.set_volume_left(100)
#             volLeft = 100
#         print("Right Decrease")
#         volRight = volRight - 5
#         lr_bal.set_volume_right(volRight)

#     elif leftOrRight == 1.:
#         if volRight != 100:
#             lr_bal.set_volume_right(100)
#             volRight = 100
#         print("Left Decrease")
#         volLeft = volLeft - 5
#         lr_bal.set_volume_left(volLeft)