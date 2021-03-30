'''
   ##PLOTTING EEG EMULATION##
   i = 0
   samples = []
   timesamples = list(np.linspace(0, 1, 120))

   labels=[]
   for nummers in range(1,25):
       labels.append('Channel ' + str(nummers))

   while True:
       sample, notused = EEG_inlet.pull_sample()
       # print(timestamp)
       sample = np.transpose(sample)  # rows = 24 channels , columns = 7200 time instances
       y = butter_bandpass_filter(sample, 12, 30, 120, order=8)
       y = np.transpose(y)
       samples.append(y)
       i += 1
       if i == 7200:
           break
   samples = samples[200:320] # 120 x 24
   plt.figure("EEG emulation, for all channels")
   plt.title("EEG emulation, for all channels")
   plt.plot(timesamples, samples)
   plt.ylabel("EEG amplitude (Volt)")
   plt.xlabel("time (seconds)")
   plt.legend(labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
   plt.show()
   plt.close()
   ### from this point errors, because of the transposes:
   samples = np.transpose(samples)
   for i in range(24):
       mean = np.mean(samples[i])
       for j in range(120):
           samples[i][j] = samples[i][j]-mean
   plt.figure("EEG emulation, for channels 1 to 6 - MEAN ")
   plt.title("EEG emulation, for channels 1 to 6 minus DC-value")
   plt.plot(timesamples, np.transpose(samples[:6]))
   plt.ylabel("EEG amplitude (Volt)")
   plt.xlabel("time (seconds)")
   plt.legend(labels[:6], bbox_to_anchor=(1.0, 0.5), loc="center left")
   plt.show()
   plt.close()
   plt.figure("EEG emulation, for channel 7 to 12")
   plt.title("EEG emulation, for channel 7 to 12")
   plt.plot(timesamples, np.transpose(samples[6:12]))
   plt.ylabel("EEG amplitude (Volt)")
   plt.xlabel("time (seconds)")
   plt.legend(labels[6:12], bbox_to_anchor=(1.0, 0.5), loc="center left")
   plt.show()
   plt.close()
   plt.figure("EEG emulation, for channel 13 to 18")
   plt.title("EEG emulation, for channel 13 to 18")
   plt.plot(timesamples, np.transpose(samples[13:18]))
   plt.ylabel("EEG amplitude (Volt)")
   plt.xlabel("time (seconds)")
   plt.legend(labels[13:18], bbox_to_anchor=(1.0, 0.5), loc="center left")
   plt.show()
   plt.close()
   plt.figure("EEG emulation, for channel 19 to 24")
   plt.title("EEG emulation, for channel 19 to 24")
   plt.plot(timesamples, np.transpose(samples[19:24]))
   plt.ylabel("EEG amplitude (Volt)")
   plt.xlabel("time (seconds)")
   plt.legend(labels[19:24], bbox_to_anchor=(1.0, 0.5), loc="center left")
   plt.show()
   plt.close()
   '''
'''
##REALTIME EEG EMULATION PLOT##:
plt.figure("Realtime EEG emulation")

x = []
samples = []
i = 0

labels=[]
for nummers in range(1, 25):
    labels.append('Channel ' + str(nummers))

while True:
    sample, notused = EEG_inlet.pull_sample()
    sample = np.transpose(sample)  #
    y = butter_bandpass_filter(sample, 12, 30, 120, order=8)
    y = np.transpose(y)
    x.append(float(i/120))
    samples.append(y)

    if len(samples) < 120:
        # FOR ONLY ONE CHANNEL:
        plt.plot(x, np.transpose(np.transpose(samples)[2]))
        # FOR ALL CHANNELS:
        #plt.plot(x, samples)
    else:
        # FOR ONLY ONE CHANNEL:
        plt.plot(x[-120:],np.transpose(np.transpose(samples[-120:])[2]))
        # FOR ALL CHANNELS:
        #plt.plot(x[-120:],samples[-120:])
    plt.ylabel("EEG amplitude (Volt)")
    plt.xlabel("time (seconds)")
    plt.title("Realtime EEG emulation")
    plt.legend(labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
    plt.draw()
    plt.pause(1/(240))
    i+=1
    plt.clf()

'''