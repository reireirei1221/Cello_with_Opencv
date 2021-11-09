#from winsound import Beep
#from msvcrt import getch
import sys
#import getch
import os
from playsound import playsound

# 半音のズレが何倍か定義
onestep_pitch = 2 ** (1.0/12.0)
# 音を鳴らす時間をミリ秒で定義
duration = 300

# 音を鳴らす関数を定義
def play_pitch(frequency, duration):
    # Beep(frequency, duration)
    os.system('beep -f %s -l %s' % (frequency,duration))
    # playsound("piano.wav")
# 半音上げ下げする関数を定義
def down_pitch(base_pitch):
    return int(round(base_pitch / onestep_pitch))
def up_pitch(base_pitch):
    return int(round(base_pitch * onestep_pitch))

# 各音程の周波数を定義
A4 = 440
Ais4 = up_pitch(A4)
H4 = up_pitch(Ais4)
C5 = up_pitch(H4)
Cis5 = up_pitch(C5)
D5 = up_pitch(Cis5)
Dis5 = up_pitch(D5)
E5 = up_pitch(Dis5)

Gis4 = down_pitch(A4)
G4 = down_pitch(Gis4)
Fis4 = down_pitch(G4)
F4 = down_pitch(Fis4)
E4 = down_pitch(F4)
Dis4 = down_pitch(E4)
D4 = down_pitch(Dis4)
Cis4 = down_pitch(D4)
C4 = down_pitch(Cis4)
H3 = down_pitch(C4)
Ais3 = down_pitch(H3)
A3 = down_pitch(Ais3)

# キーボードと音程を関連づける。キーボードの"d"がC4、つまりドの音など
pitchs = {}
pitchs["a"] = A3
pitchs["w"] = Ais3
pitchs["s"] = H3
pitchs["d"] = C4
pitchs["r"] = Cis4
pitchs["f"] = D4
pitchs["t"] = Dis4
pitchs["g"] = E4
pitchs["h"] = F4
pitchs["u"] = Fis4
pitchs["j"] = G4
pitchs["i"] = Gis4
pitchs["k"] = A4
pitchs["o"] = Ais4
pitchs["l"] = H4
pitchs[";"] = C5
pitchs["@"] = Cis5
pitchs[":"] = D5
pitchs["["] = Dis5
pitchs["]"] = E5

while True:
    # 入力されたキーを認識する
    #bytes_keyboard = getch()
    bytes_keyboard = sys.stdin.read(1)
    # バイト列から文字列に変換する
    #str_keyboard = bytes_keyboard.decode("utf-8")
    # 文字列を小文字に揃える
    #pitch = str_keyboard.lower()
    pitch = bytes_keyboard.lower()
    print("音階", pitch)
    # 押したキーが辞書の中に存在するとき、音を鳴らす
    if pitch in pitchs:
        play_pitch(pitchs[pitch], duration)
    # 終了させるときはqを押す
    elif pitch == 'q':
        break
