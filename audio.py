#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 17:14:35 2020

@author: Jasper Wouters
"""

import os
import time
import multiprocessing
import numpy as np

import wave
import alsaaudio as audio

from pylsl import StreamInfo, StreamOutlet


class AudioPlayer:
    START_MARKER = 'PlaybackStarted'
    STOP_MARKER = 'PlaybackStopped'

    """ 
    An audio player class based on ALSA (Linux only).
    """
    def __init__(self):
        self._device = None  # PCM playback device
        self._playback_daemon = None
        self._wav_fh = None  # wav file
        self._period_size = None  # period size for playback
        self._marker_outlet = None

    def list_devices(self):
        """
        List the available ALSA playback devices.

        """
        return audio.pcms(audio.PCM_PLAYBACK)

    def set_device(self, device_name, cardindex):
        """
        Set the playback device.

        Parameters
        ----------
        device_name : str
            Name of playback device.

        """
        self._device = audio.PCM(type=audio.PCM_PLAYBACK,
                                 device=device_name,
                                 cardindex=cardindex)

    def init_play(self, wav_fn):
        """
        Initialise the audio player for wav playback. Requires that a playback
        device has been set.

        Parameters
        ----------
        wav_fn : str
            Full filename to wav.

        """
        # open wav file
        self._wav_fh = wave.open(wav_fn, 'rb')

        # initialize playback device
        self._device.setchannels(self._wav_fh.getnchannels())
        self._device.setrate(self._wav_fh.getframerate())

        # 8bit is unsigned in wav files
        if self._wav_fh.getsampwidth() == 1:
            self._device.setformat(audio.PCM_FORMAT_U8)
        # otherwise we assume signed data, little endian
        elif self._wav_fh.getsampwidth() == 2:
            self._device.setformat(audio.PCM_FORMAT_S16_LE)
        elif self._wav_fh.getsampwidth() == 3:
            self._device.setformat(audio.PCM_FORMAT_S24_3LE)
        elif self._wav_fh.getsampwidth() == 4:
            self._device.setformat(audio.PCM_FORMAT_S32_LE)
        else:
            raise ValueError('Unsupported format')

        # TODO: this division is related to buffersize (?), parameterise
        # increasing the divisor, increases cpu usage, which likely reduced latency?
        self._period_size = self._wav_fh.getframerate() // 8
        self._device.setperiodsize(self._period_size)

        # init playback daemon
        self._playback_daemon = multiprocessing.Process(target=self.__low_level_playback)
        self._playback_daemon.daemon = True # playback gets killed when main terminates

        # init lsl marker communication
        info = StreamInfo('AudioPlayerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
        self._marker_outlet = StreamOutlet(info)

    def __low_level_playback(self):
        # low level audio playback
        data = self._wav_fh.readframes(self._period_size)
        while data:
            # Read data from stdin
            self._device.write(data)
            data = self._wav_fh.readframes(self._period_size)

        print('AudioPlayer has reached the end of the recording.')

    def pause(self, yesorno):
        if yesorno:
            self._device.pause(True)
        else:
            self._device.pause(False)

    def play(self):
        """
        Play audio for which the audio player has been initialised.

        """
        # TODO starting this process might involve unacceptable delay due to
        # copying of the AudioPlayer object (?)
        self._playback_daemon.start()
        # communicate that playback start to LSL
        self._marker_outlet.push_sample([self.START_MARKER])

    def stop(self):
        """
        Stop audio play back

        """
        self._playback_daemon.terminate()
        # communicate that playback stopped to LSL
        self._marker_outlet.push_sample([self.STOP_MARKER])
        self._playback_daemon.join()


class LRBalancer:
    """ Object to control Left-Right balance
    """
    _LEFT = 0
    _RIGHT = 1

    def __init__(self, ):
        self._control = None

    def list_controls(self, device_name):
        """
        List the available controls for the given device name.

        Parameters
        ----------
        device_name : str
            Device name for which to display the available controls.

        """
        print(audio.mixers())

    def set_control(self, control_name, device_name, cardindex):
        """
        Attach a desired control to this balancer object

        Parameters
        ----------
        control_name : str
            Name of the desired control.
        device_name : str
            Name of the device to control.

        """
        self.control = audio.Mixer(control=control_name,
                                   device=device_name,
                                   cardindex=cardindex)

    # TODO refactor everything related to left and right as parameter
    def set_volume_left(self, volume):
        """
        Alter the volume of the left channel

        Parameters
        ----------
        volume : float
            Volume between 0-100.

        """
        self.control.setvolume(self.__perceptual_conversion(volume),
                               self._LEFT)

    def get_volume(self):
        """
        Get the volume of both channels

        """
        return self.control.getvolume()

    def set_volume_right(self, volume):
        """
        Alter the volume of the right channel

        Parameters
        ----------
        volume : float
            Volume between 0-100.

        """
        self.control.setvolume(self.__perceptual_conversion(volume),
                               self._RIGHT)

    IN = 0
    OUT = 1

    def fade_left(self, in_out, duration, nb_steps=100):
        """
        Fade the left channel in or out.

        Parameters
        ----------
        in_out : LRBalance.IN or LRBalance.OUT
            Direction of fading.
        duration : float
            Duration in seconds over which to fade.
        nb_steps : int
            Number of steps over whcih to fade

        """
        for volume in self.__create_volume_steps(in_out, nb_steps):
            self.set_volume_left(volume)
            self.__pause(duration, nb_steps)

    def fade_right(self, in_out, duration, nb_steps=100):
        """
        Fade the right channel in or out.

        Parameters
        ----------
        in_out : LRBalance.IN or LRBalance.OUT
            Direction of fading.
        duration : float
            Duration in seconds over which to fade.
        nb_steps : int
            Number of steps over whcih to fade

        """
        for volume in self.__create_volume_steps(in_out, nb_steps):
            self.set_volume_right(volume)
            self.__pause(duration, nb_steps)

    def __create_volume_steps(self, in_out, nb_steps):
        steps = np.linspace(0, 100, nb_steps)

        if in_out == self.OUT:
            steps = steps[::-1]

        return steps

    def __pause(self, duration, nb_steps):
        time.sleep(duration / nb_steps)

    def __perceptual_conversion(self, volume, dynamic_range=50):
        # info: https://github.com/larsimmisch/pyalsaaudio/issues/8
        dB = lambda x: 2*np.log10(x)
        vol = lambda d,x: dB((1-x)/d+x)/dB(d)+1

        p = volume / 100
        v = vol(dynamic_range, p)
        # a,b = mixer_range
        a, b = (0, 100)
        return int(a*(1-v)+b*v)


def main():
    # playback parameters (tested on Ubuntu 16.04 with Intel HDA audio card)
    device_name = 'sysdefault'
    # control_name = 'Headphone'
    # control_name = "Headphone Jack Sense"
    control_name = 'Master'
    cardindex = 0
    wav_fn = os.path.join(os.path.expanduser('~/Music'),
                          'travel.wav')
    vol_left = 100
    vol_right = 100

    ap = AudioPlayer()

    ap.list_devices()

    ap.set_device(device_name, cardindex)
    ap.init_play(wav_fn)

    lr_bal = LRBalancer()
    lr_bal.set_control(control_name, device_name, cardindex)

    lr_bal.set_volume_left(vol_left)
    lr_bal.set_volume_right(vol_right)

    # wait a little for the LSL receivers to set up
    time.sleep(5)

    ap.play()

    time.sleep(10)
    print("85 // 100")
    vol_left -= 15
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("70 // 100")
    vol_left -= 15
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("55 // 100")
    vol_left -= 15
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("40 // 100")
    vol_left -= 15
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("25 // 100")
    vol_left = 25
    lr_bal.set_volume_left(vol_left)
    time.sleep(10)
    print("50 // 85")
    vol_left += 25
    vol_right -= 15
    lr_bal.set_volume_right(vol_right)
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("75 // 70")
    vol_left += 25
    vol_right -= 15
    lr_bal.set_volume_right(vol_right)
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("100 // 55")
    vol_left += 25
    vol_right -= 15
    lr_bal.set_volume_right(vol_right)
    lr_bal.set_volume_left(vol_left)
    time.sleep(5)
    print("100 // 25")
    vol_right = 25
    lr_bal.set_volume_right(vol_right)
    time.sleep(5)

    ap.stop()

    # time.sleep(5)
    # vol_left = 50
    # lr_bal.set_volume_left(vol_left)
    # time.sleep(5)
    # vol_left -= 5
    # lr_bal.set_volume_left(vol_left)
    # time.sleep(5)
    # vol_left -= 10
    # lr_bal.set_volume_left(vol_left)
    # time.sleep(15)
    # vol_left -= 10
    # lr_bal.set_volume_left(vol_left)
    # time.sleep(5)
    # lr_bal.fade_right(LRBalancer.OUT, 5)
    # time.sleep(5)
    # lr_bal.fade_left(LRBalancer.OUT, 5)
    # time.sleep(5)
    # lr_bal.fade_right(LRBalancer.IN, 5)
    # time.sleep(5)
    # lr_bal.fade_left(LRBalancer.IN, 5)
    # time.sleep(5)

    # ap.pause(True)
    # input("enter")
    # ap.pause(False)
    # time.sleep(10)

    ap.stop()

    time.sleep(2)

    print('leaving main')


if __name__ == '__main__':
    main()
