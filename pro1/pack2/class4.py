# 가수 관련 클래스

class SingerType: 
# 가수면 기본적으로 가져야할 속성과 행위가 있다고 가정
    title_song = '화이팅 코리아'

    def sing(self):
        msg = '노래는 '
        print(msg, self.title_song + ' 랄랄라 ~~~~')