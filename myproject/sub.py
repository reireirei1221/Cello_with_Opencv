#file name is sub.py
#サブプロセスで実行されるファイル
#CMD上で python sub.py name というコマンドでname.wavファイルが再生される

import sys
import wave
import pyaudio

args = sys.argv

wf = wave.open(str(args[1])+".wav", "r")
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# チャンク単位でストリームに出力し音声を再生
chunk = 1024
data = wf.readframes(chunk)
while data != b'':
    stream.write(data)
    data = wf.readframes(chunk)
stream.close()
p.terminate()